"""Outcome model fitting (logistic regression).

The model takes a corpus of `Run` objects and fits `P(success | features)` via
scikit-learn's logistic regression. It keeps the validated corpus so
`intervene` can build an estimand-specific design (treatment arm plus
pre-treatment covariates) instead of reusing this corpus-wide fit.

Featurization is one-hot per (decision_type, chosen_action) for any decision
whose type has at least one declared intervention kind in the taxonomy. That
restricts attention to the controllable surface — exactly the variables for
which counterfactual queries are meaningful.
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
from counterfact.outcome.features import decision_feature_values
from counterfact.schema import Run


@dataclass
class OutcomeModel:
    """A corpus-wide outcome fit plus the validated runs it was fit on."""

    feature_names: list[str]
    coefficients: np.ndarray
    intercept: float
    train_X: np.ndarray
    train_y: np.ndarray
    train_n: int
    runs: list[Run]
    seed: int | None = 0
    feature_index: dict[str, int] = field(default_factory=dict)
    outcome_kind: str = "binary"

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


def _intervenable_decisions(run: Run) -> list[tuple[str, str]]:
    """Yield feature-family/value pairs for intervenable decisions."""
    out: list[tuple[str, str]] = []
    for step in run.steps:
        for d in step.decisions:
            out.extend(decision_feature_values(d))
    return out


def _featurize(runs: list[Run]) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Walk each run once, building feature names and per-run rows together."""
    per_run_pairs: list[list[tuple[str, str]]] = [_intervenable_decisions(r) for r in runs]
    seen: set[str] = set()
    for pairs in per_run_pairs:
        for family, value in pairs:
            seen.add(f"{family}::{value}")
    names = sorted(seen)
    index = {n: i for i, n in enumerate(names)}

    X = np.zeros((len(runs), len(names)), dtype=float)
    for i, pairs in enumerate(per_run_pairs):
        for family, value in pairs:
            key = f"{family}::{value}"
            if key in index:
                X[i, index[key]] = 1.0
    y = np.array([1 if binary_outcome_value(r) else 0 for r in runs], dtype=int)
    return X, y, names, index


def fit_logistic(
    X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        lr.fit(X, y, sample_weight=sample_weight)
    return lr


def fit_outcome_model(
    traces: Iterable[Run],
    *,
    seed: int | None = 0,
) -> OutcomeModel:
    """Fit a corpus-wide logistic regression; `seed` drives intervene's cross-fit folds."""
    runs = list(traces)
    _assert_binary(runs)
    if not runs:
        raise ValueError("fit_outcome_model requires at least one trace")

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
    base = fit_logistic(X, y)

    return OutcomeModel(
        feature_names=names,
        coefficients=base.coef_[0].copy(),
        intercept=float(base.intercept_[0]),
        train_X=X,
        train_y=y,
        train_n=len(runs),
        runs=runs,
        seed=seed,
        feature_index=index,
        outcome_kind="binary",
    )
