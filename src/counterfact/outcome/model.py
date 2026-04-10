"""Outcome model fitting (logistic regression + bootstrap).

The model takes a corpus of `Run` objects and fits `P(success | features)` via
scikit-learn's logistic regression, then bootstraps coefficients and a held-out
empirical distribution of feature rows for use by `intervene`.

Featurization is one-hot per (decision_type, chosen_action) for any decision
whose type has at least one declared intervention kind in the taxonomy. That
restricts attention to the controllable surface — exactly the variables for
which counterfactual queries are meaningful.

Per design.md D3: bootstrap coefficient uncertainty is reported separately
from identifiability uncertainty about the causal estimand.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from counterfact.errors import UnsupportedOutcomeError
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
        return 1.0 / (1.0 + np.exp(-z))


def _assert_binary(runs: list[Run]) -> None:
    for run in runs:
        kind = run.outcome.kind
        if kind != "binary":
            raise UnsupportedOutcomeError(
                f"v0 only supports binary outcomes; got kind={kind!r} on run {run.run_id!r}"
            )


def _intervenable_decisions(run: Run) -> list[tuple[str, str]]:
    """Yield (decision_type, chosen_action) for decisions whose type has interventions."""
    from counterfact.taxonomy import valid_interventions

    out: list[tuple[str, str]] = []
    for step in run.steps:
        for d in step.decisions:
            if not valid_interventions(d.decision_type):
                continue
            if d.chosen_action is None:
                continue
            out.append((d.decision_type, d.chosen_action))
    return out


def _build_feature_index(runs: list[Run]) -> tuple[list[str], dict[str, int]]:
    """Discover the feature names by scanning the training corpus."""
    seen: set[str] = set()
    for run in runs:
        for dt, action in _intervenable_decisions(run):
            seen.add(f"{dt}::{action}")
    names = sorted(seen)
    return names, {n: i for i, n in enumerate(names)}


def _row_for_run(run: Run, index: dict[str, int]) -> np.ndarray:
    row = np.zeros(len(index), dtype=float)
    for dt, action in _intervenable_decisions(run):
        key = f"{dt}::{action}"
        if key in index:
            row[index[key]] = 1.0
    return row


def _featurize(runs: list[Run]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    names, index = _build_feature_index(runs)
    X = np.vstack([_row_for_run(r, index) for r in runs])
    y = np.array([1 if r.outcome.value else 0 for r in runs], dtype=int)
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
    schema: object | None = None,
    outcome: str = "success",
    n_bootstrap: int = 200,
    seed: int | None = 0,
) -> OutcomeModel:
    """Fit a logistic regression with bootstrap over training rows."""
    runs = list(traces)
    _assert_binary(runs)
    if not runs:
        raise ValueError("fit_outcome_model requires at least one trace")

    X, y, names, index = _featurize(runs)
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
