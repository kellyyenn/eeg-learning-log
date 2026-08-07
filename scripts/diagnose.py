"""Is the pipeline broken, or is 2-subject LOSO just genuinely hard?

Within-subject is the control: if CSP+LDA can't hit ~0.4-0.6 kappa within a
subject on IV-2a, something is wrong with the data/pipeline, not the protocol.
"""
import sys, time, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import logging; logging.getLogger("mne").setLevel(logging.ERROR)
import mne; mne.set_log_level("ERROR")

import numpy as np
from eeglog.data import load_moabb
from eeglog.models_classic import csp_lda, riemann_ts_lr
from eeglog.evaluation import evaluate_within_subject, evaluate_loso

d = load_moabb(subjects=[1, 2], paradigm="motor_imagery", fmin=8.0, fmax=30.0)
print("X", d.X.shape, "y", np.unique(d.y, return_counts=True))
print("groups", np.unique(d.groups, return_counts=True))
print("metadata cols:", list(d.metadata.columns))
print(d.metadata.head())
print("sessions:", d.metadata["session"].unique())
print("X scale: mean=%.3e std=%.3e" % (d.X.mean(), d.X.std()))

for name, mk in [("CSP+LDA", csp_lda), ("Riemann+TS+LR", riemann_ts_lr)]:
    t = time.time()
    r = evaluate_within_subject(mk(), d.X, d.y, d.groups, metric="kappa", model_name=name)
    print(f"WITHIN  {r}  per-subj={np.round(r.scores,3)}  ({time.time()-t:.0f}s)")
