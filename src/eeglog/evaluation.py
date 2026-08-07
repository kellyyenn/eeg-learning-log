"""Honest evaluation — the point of the whole project.

Five rules, enforced here:

1. All trials from one subject live in exactly one of {train, test}.
2. Every data-learning step is fit *inside* the fold (we pass whole pipelines to
   ``cross_validate``, so sklearn does this for us).
3. The headline number is cross-subject (Leave-One-Subject-Out), reported as
   mean ± std across held-out subjects.
4. We report imbalance-aware metrics (balanced accuracy, Cohen's kappa, macro-F1)
   and a confusion matrix, not raw accuracy alone.
5. Every number is tagged with its protocol so a within-session score is never
   mistaken for the headline.

The :func:`leakage_demo` function makes the classic mistake on purpose and shows
the inflated score next to the honest one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
)


@dataclass
class EvalResult:
    model: str
    protocol: str
    metric: str
    scores: np.ndarray  # per-fold

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    @property
    def std(self) -> float:
        return float(np.std(self.scores))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.model} [{self.protocol}] {self.metric}: {self.mean:.3f} ± {self.std:.3f}"


def loso_splits(groups):
    """Yield train/test indices with Leave-One-Subject-Out. Honest headline CV."""
    n = len(groups)
    return LeaveOneGroupOut().split(np.zeros(n), np.zeros(n), groups=groups)


def evaluate_loso(model, X, y, groups, metric: str = "balanced_accuracy", model_name="model"):
    """Score a pipeline cross-subject (LOSO). Returns an :class:`EvalResult`.

    The model is cloned and refit per fold by ``cross_val_predict``, so every
    data-learning step inside it sees train-only data — no leakage.
    """
    cv = LeaveOneGroupOut()
    y_pred = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=1)
    # Per-subject scores (so we can report mean ± std across subjects).
    scores = []
    for subj in np.unique(groups):
        mask = groups == subj
        scores.append(_score(metric, y[mask], y_pred[mask]))
    return EvalResult(model_name, "cross-subject (LOSO)", metric, np.asarray(scores))


def evaluate_within_subject(
    model, X, y, groups, n_splits: int = 5, metric: str = "balanced_accuracy", model_name="model"
):
    """Score within each subject (Stratified K-fold per subject), then pool folds.

    This is the *inflated* protocol shown for contrast — never the headline.
    """
    scores = []
    for subj in np.unique(groups):
        mask = groups == subj
        Xs, ys = X[mask], y[mask]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred = cross_val_predict(model, Xs, ys, cv=cv, n_jobs=1)
        scores.append(_score(metric, ys, y_pred))
    return EvalResult(model_name, "within-subject", metric, np.asarray(scores))


def _score(metric: str, y_true, y_pred) -> float:
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric == "kappa":
        return cohen_kappa_score(y_true, y_pred)
    if metric == "f1_macro":
        return f1_score(y_true, y_pred, average="macro")
    raise ValueError(f"Unknown metric {metric!r}")


def confusion(model, X, y, groups):
    """LOSO out-of-fold confusion matrix (labels sorted) for error analysis."""
    cv = LeaveOneGroupOut()
    y_pred = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=1)
    labels = np.unique(y)
    return confusion_matrix(y, y_pred, labels=labels), labels


def leaderboard(results: list[EvalResult]) -> pd.DataFrame:
    """Tidy mean ± std table from a list of EvalResults, sorted by mean desc."""
    rows = [
        {
            "model": r.model,
            "protocol": r.protocol,
            "metric": r.metric,
            "mean": r.mean,
            "std": r.std,
            "score": f"{r.mean:.3f} ± {r.std:.3f}",
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    return df.sort_values("mean", ascending=False).reset_index(drop=True)


def leakage_demo(make_csp_lda, X, y, groups) -> dict:
    """Show the CSP-fit-on-all-data trap vs the honest in-fold score.

    ``make_csp_lda`` is a zero-arg factory returning a fresh CSP+LDA pipeline.

    Returns a dict with both numbers. The "leaky" score fits CSP on the *entire*
    dataset (peeking at test trials) before CV; the "honest" score fits CSP
    inside each fold. The leaky number is reliably, misleadingly higher.
    """
    from sklearn.base import clone
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from .features import make_csp

    # --- Leaky: transform with CSP fit on ALL data, then CV just the LDA. ---
    csp = make_csp()
    X_leaky = csp.fit_transform(X, y)  # <-- peeks at every trial, incl. test folds
    leaky = evaluate_loso(
        LinearDiscriminantAnalysis(), X_leaky, y, groups, model_name="CSP(leaky)+LDA"
    )

    # --- Honest: CSP refit inside each fold. ---
    honest = evaluate_loso(clone(make_csp_lda()), X, y, groups, model_name="CSP+LDA")

    return {
        "leaky_mean": leaky.mean,
        "honest_mean": honest.mean,
        "inflation": leaky.mean - honest.mean,
        "leaky": leaky,
        "honest": honest,
    }
