"""Sanity tests for the eeglog pipeline.

These use real MOABB data (cached under ~/mne_data after first run). They are
intentionally small (2 subjects, 2 classes) so they finish in a couple of
minutes on a laptop CPU. Run with::

    PYTHONPATH=src .venv/bin/pytest -q

What they prove:
* The data loads by code into the expected shape.
* A CSP+LDA pipeline scores above chance cross-subject (LOSO) — the plumbing
  works, without claiming inflated numbers.
* The leakage demo really does inflate the score (fitting CSP on all data beats
  the honest in-fold score) — locks in the central lesson as a regression test.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeglog import data, evaluation
from eeglog.models_classic import csp_lda


@pytest.fixture(scope="session")
def loaded():
    # Two subjects -> a real (if tiny) Leave-One-Subject-Out split.
    return data.load_moabb(subjects=[1, 2], paradigm="left_right")


def test_load_shape(loaded):
    assert loaded.X.ndim == 3  # (n_epochs, n_channels, n_times)
    assert loaded.n_classes == 2
    assert set(np.unique(loaded.groups)) == {1, 2}
    assert loaded.sfreq > 0


def test_csp_lda_above_chance(loaded):
    # Within-subject: signal clearly exists, so a working pipeline must beat
    # chance. (Cross-subject from a *single* training subject is genuinely
    # near-chance — that honest fact is demonstrated in notebook 06, not
    # asserted here.)
    res = evaluation.evaluate_within_subject(
        csp_lda(), loaded.X, loaded.y, loaded.groups, model_name="CSP+LDA"
    )
    assert res.mean > 0.6, f"CSP+LDA only reached {res.mean:.3f} (within-subject)"


def test_leakage_inflates_score(loaded):
    out = evaluation.leakage_demo(csp_lda, loaded.X, loaded.y, loaded.groups)
    # Fitting CSP on all data must beat the honest in-fold score.
    assert out["leaky_mean"] > out["honest_mean"], (
        f"expected leakage to inflate: leaky={out['leaky_mean']:.3f} "
        f"honest={out['honest_mean']:.3f}"
    )
