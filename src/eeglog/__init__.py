"""eeglog — a hands-on EEG/BCI motor-imagery learning toolkit.

The package holds the reusable mechanics (loading, preprocessing, features,
models, evaluation). The numbered notebooks in ``notebooks/`` are the narrative
"learning log" and import from here.

Design rule that runs through everything: anything that *learns* from data
(scaler, ICA, CSP, covariances, a net) must be fit on training data only,
inside the cross-validation fold. See ``eeglog.evaluation``.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Reproducibility: one seed used across the project.
RANDOM_STATE = 42

__all__ = ["RANDOM_STATE", "__version__"]
