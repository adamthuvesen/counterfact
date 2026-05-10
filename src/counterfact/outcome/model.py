"""Outcome model fitting (logistic regression + bootstrap).

The model takes a corpus of `Run` objects and fits `P(success | features)` via
scikit-learn's logistic regression, then bootstraps coefficients and a held-out
empirical distribution of feature rows for use by `intervene`.

Featurization is one-hot per (decision_type, chosen_action) for any decision
whose type has at least one declared intervention kind in the taxonomy. That
restricts attention to the controllable surface — exactly the variables for
which counterfactual queries are meaningful.

Bootstrap coefficient uncertainty is reported separately from identifiability
uncertainty about the causal estimand: the two answer different questions and
must not be conflated in the surfaced output.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from counterfact.errors import InsufficientOutcomeSupportError, UnsupportedOutcomeError
from counterfact.outcome.binary import binary_outcome_value
from counterfact.schema import Run


@dataclass
class OutcomeModel:
    """A fitted outcome model + cached training-distribution rows for g-formula."""

    feature_names: list[str]
    coefficients: np.ndarray
    intercept: float
    bootstrap_coefs: np.ndarray  # shape (B, n_features)
    bootstrap_intercepts: np.ndarray  # shape (B,)
    train_X: np.ndarray
    train_y: np.ndarray
    train_n: int
    feature_index: dict[str, int] = field(default_factory=dict)
    outcome_kind: str = "binary"

    @property
    def bootstrap_ci(self) -> dict[str, tuple[float, float]]:
        """Return per-feature 95% bootstrap CI for the coefficient."""
        lo = np.percentile(self.bootstrap_coefs, 2.5, axis=0)
        hi = np.percentile(self.bootstrap_coefs, 97.5, axis=0)
        return {name: (float(lo[i]), float(hi[i])) for i, name in enumerate(self.feature_names)}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Sigmoid of (X @ coef + intercept). Shape: (n,)."""
        z = X @ self.coefficients + self.intercept
        result: np.ndarray = 1.0 / (1.0 + np.exp(-z))
        return result


def _assert_binary(runs: list[Run]) -> None:
    for run in runs:
        kind = run.outcome.kind
        if kind != "binary":
            raise UnsupportedOutcomeError(
                f"v0 only supports binary outcomes; got kind={kind!r} on run {run.run_id!r}"
            )


def _intervenable_feature_keys(run: Run) -> list[str]:
    """Return the one-hot `decision_type::action` keys present in `run`.

    Delegates to `taxonomy.extract_features` so feature-key naming has a
    single source of truth shared with the rest of the library.
    """
    from counterfact.taxonomy import extract_features

    keys: list[str] = []
    for step in run.steps:
        for d in step.decisions:
            feats = extract_features(d, run)
            key = feats.get("feature_key")
            if key is not None:
                keys.append(key)
    return keys


def _featurize(runs: list[Run]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Walk each run once, building feature names and per-run rows together."""
    per_run_keys: list[list[str]] = [_intervenable_feature_keys(r) for r in runs]
    seen: set[str] = set()
    for keys in per_run_keys:
        seen.update(keys)
    names = sorted(seen)
    index = {n: i for i, n in enumerate(names)}

    X = np.zeros((len(runs), len(names)), dtype=float)
    for i, keys in enumerate(per_run_keys):
        for key in keys:
            X[i, index[key]] = 1.0
    y = np.array([1 if binary_outcome_value(r) else 0 for r in runs], dtype=int)
    return X, y, names, index


def _fit_lr(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        lr.fit(X, y)
    return lr


def fit_outcome_model(
    traces: Iterable[Run],
    *,
    n_bootstrap: int = 200,
    seed: int | None = 0,
) -> OutcomeModel:
    """Fit a logistic regression with bootstrap over training rows."""
    runs = list(traces)
    _assert_binary(runs)
    if not runs:
        raise ValueError("fit_outcome_model requires at least one trace")
    if n_bootstrap < 1:
        # `intervene` percentiles over an empty bootstrap distribution; NumPy
        # also rejects negative shapes. Fail at the public boundary with a
        # clear message rather than at a low-level allocation site.
        raise ValueError(f"n_bootstrap must be >= 1; got {n_bootstrap!r}")

    X, y, names, index = _featurize(runs)
    if X.shape[1] == 0:
        raise InsufficientOutcomeSupportError(
            "fit_outcome_model requires at least one intervenable decision feature; "
            "observed no decisions with supported interventions and chosen_action values. "
            "Collect traces with model_call, tool_call, retry, or other intervenable "
            "decisions before fitting an outcome model."
        )
    classes = set(y.tolist())
    if len(classes) < 2:
        observed = sorted(classes)
        raise InsufficientOutcomeSupportError(
            "fit_outcome_model requires at least two outcome classes for binary "
            f"outcomes; observed classes={observed}. Collect or construct traces "
            "with both pass and fail outcomes before fitting an outcome model."
        )
    base = _fit_lr(X, y)

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    boot_coefs = np.zeros((n_bootstrap, X.shape[1]), dtype=float)
    boot_intercepts = np.zeros(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        # require at least one of each class for the boot model to fit
        if len(set(y[idx])) < 2:
            # fallback: use the base coefs/intercept for this bootstrap draw
            boot_coefs[b] = base.coef_[0]
            boot_intercepts[b] = base.intercept_[0]
            continue
        bm = _fit_lr(X[idx], y[idx])
        boot_coefs[b] = bm.coef_[0]
        boot_intercepts[b] = bm.intercept_[0]

    return OutcomeModel(
        feature_names=names,
        coefficients=base.coef_[0].copy(),
        intercept=float(base.intercept_[0]),
        bootstrap_coefs=boot_coefs,
        bootstrap_intercepts=boot_intercepts,
        train_X=X,
        train_y=y,
        train_n=len(runs),
        feature_index=index,
        outcome_kind="binary",
    )
