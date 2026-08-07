"""Classical model pipelines for motor-imagery decoding.

Each factory returns a *complete* ``sklearn`` estimator: feature extraction and
classifier in one ``Pipeline``. That packaging is the whole point — when the
pipeline is handed to a grouped CV splitter, every data-learning step (CSP,
covariances, scaler) is fit on the training fold only. No leakage by
construction.

Pipelines provided
------------------
* ``csp_lda``          : CSP + LDA               — the classic MI baseline.
* ``fbcsp_svm``        : Filter-bank CSP + SVM    — multi-band, usually stronger.
* ``riemann_ts_lr``    : Cov + TangentSpace + LR  — Riemannian, often the best.
* ``riemann_mdm``      : Cov + MDM                — parameter-free Riemannian.
* ``bandpower_rf``     : log band power + RandomForest — feature-engineering contrast.

All take ``sfreq`` where needed and a ``random_state`` for reproducibility.
"""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import BandPowerFeatures, FBCSP, make_covariances, make_csp


def csp_lda(n_components: int = 6) -> Pipeline:
    """CSP spatial filtering + Linear Discriminant Analysis."""
    return make_pipeline(make_csp(n_components=n_components), LinearDiscriminantAnalysis())


def fbcsp_svm(sfreq: float, n_components: int = 4, random_state: int = 42) -> Pipeline:
    """Filter-bank CSP + RBF SVM (scaled features)."""
    return make_pipeline(
        FBCSP(sfreq=sfreq, n_components=n_components),
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state),
    )


def riemann_ts_lr(estimator: str = "lwf", random_state: int = 42) -> Pipeline:
    """Covariances -> tangent space -> Logistic Regression (Riemannian)."""
    from pyriemann.tangentspace import TangentSpace

    return make_pipeline(
        make_covariances(estimator),
        TangentSpace(metric="riemann"),
        LogisticRegression(max_iter=1000, random_state=random_state),
    )


def riemann_mdm(estimator: str = "lwf") -> Pipeline:
    """Covariances -> Minimum Distance to Mean (parameter-free Riemannian)."""
    from pyriemann.classification import MDM

    return make_pipeline(
        make_covariances(estimator),
        MDM(metric=dict(mean="riemann", distance="riemann")),
    )


def bandpower_rf(sfreq: float, random_state: int = 42) -> Pipeline:
    """Log band power + Random Forest — a feature-engineering contrast model."""
    return make_pipeline(
        BandPowerFeatures(sfreq=sfreq),
        StandardScaler(),
        RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
    )


def all_classical(sfreq: float, random_state: int = 42) -> dict[str, Pipeline]:
    """Return every classical pipeline keyed by name (for the leaderboard)."""
    return {
        "CSP+LDA": csp_lda(),
        "FBCSP+SVM": fbcsp_svm(sfreq, random_state=random_state),
        "Riemann+TS+LR": riemann_ts_lr(random_state=random_state),
        "Riemann+MDM": riemann_mdm(),
        "BandPower+RF": bandpower_rf(sfreq, random_state=random_state),
    }
