"""Full benchmark on BCI Competition IV-2a (BNCI2014_001), 9 subjects, 4 classes.

Protocols
---------
within-subject : StratifiedKFold(5) inside each subject, scored per subject.
cross-subject  : Leave-One-Subject-Out. THE HEADLINE. Reported mean +/- std
                 across the 9 held-out subjects.

Leakage rules enforced
----------------------
* Classical models are whole sklearn Pipelines handed to CV, so CSP / covariance
  / scaler are all fit on the training fold only.
* Deep models are REBUILT FROM SCRATCH inside every fold. (Passing an
  instantiated skorch classifier to cross_val_predict would let sklearn's
  clone() deep-copy already-trained weights between folds.)
* Deep-model normalization is per-trial, per-channel z-scoring -- stateless, so
  it cannot carry information across the split.

Usage
-----
    python scripts/run_benchmark.py --phase classical
    python scripts/run_benchmark.py --phase deep
    python scripts/run_benchmark.py --phase report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import logging

logging.getLogger("mne").setLevel(logging.ERROR)
import mne

mne.set_log_level("ERROR")

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

SEED = 42
FMIN, FMAX = 8.0, 30.0  # mu + beta, where motor-imagery ERD/ERS lives
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
TAG = ""  # set from --dataset/--tag; suffixes every output file

# Dataset presets. BNCI2014_001 is BCI Competition IV-2a (the canonical
# benchmark) but its only host, lampx.tugraz.at, is frequently down -- so
# PhysionetMI (PhysioNet, 109 subjects, rock-solid mirror) is the fallback that
# still gives a genuine many-subject cross-subject split.
DATASETS = {
    "iv2a": dict(name="BNCI2014_001", subjects=list(range(1, 10))),
    "physionet": dict(name="PhysionetMI", subjects=list(range(1, 21))),
}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load(dataset="iv2a", subjects=None, fmin=FMIN, fmax=FMAX):
    from eeglog.data import load_moabb

    cfg = DATASETS[dataset]
    subs = subjects or cfg["subjects"]
    t = time.time()
    d = load_moabb(
        dataset=cfg["name"], subjects=subs, paradigm="motor_imagery",
        fmin=fmin, fmax=fmax,
    )
    print(f"[data] {cfg['name']} {d.X.shape} n_subj={len(np.unique(d.groups))} "
          f"classes={list(np.unique(d.y))} sfreq={d.sfreq} "
          f"in {time.time()-t:.0f}s band={fmin}-{fmax}Hz", flush=True)
    return d


def run_cross_session(d):
    """THE official BCI Competition IV-2a protocol: train on session T, test on E.

    Every published kappa for this dataset is this number, so it is the only
    protocol here that is directly comparable to the literature. It is also
    strictly honest -- the test session is a different day's recording, so there
    is no trial-level shuffling to inflate it.
    """
    from sklearn.base import clone

    from eeglog.models_classic import all_classical

    sess = np.asarray(d.metadata["session"])
    train_s, test_s = "0train", "1test"
    if train_s not in set(sess):
        print("[cross-session] dataset has no T/E sessions, skipping", flush=True)
        return []

    rows, preds = [], {}
    for name, mdl in all_classical(d.sfreq, random_state=SEED).items():
        yp = np.empty_like(d.y)
        for s in np.unique(d.groups):
            tr = (d.groups == s) & (sess == train_s)
            te = (d.groups == s) & (sess == test_s)
            m = clone(mdl).fit(d.X[tr], d.y[tr])
            yp[te] = m.predict(d.X[te])
            rows.append(dict(model=name, protocol="cross-session (T->E, official)",
                             subject=int(s), **scores_for(d.y[te], yp[te])))
        preds[name] = yp
        ks = [r["kappa"] for r in rows if r["model"] == name]
        print(f"[cross-session] {name:16s} kappa={np.mean(ks):.3f} "
              f"per-subj={np.round(ks,3)}", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS / f"crosssession_per_subject{TAG}.csv", index=False)
    te_mask = sess == test_s
    np.savez_compressed(RESULTS / f"crosssession_preds{TAG}.npz",
                        y_true=d.y[te_mask], groups=d.groups[te_mask],
                        **{k: v[te_mask] for k, v in preds.items()})
    print(f"[cross-session] saved -> results/crosssession_per_subject{TAG}.csv", flush=True)
    return rows


def scores_for(y_true, y_pred):
    return {
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# classical
# --------------------------------------------------------------------------- #
def run_classical(d):
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict

    from eeglog.models_classic import all_classical

    models = all_classical(d.sfreq, random_state=SEED)
    rows, preds = [], {}

    for name, mdl in models.items():
        # ---- cross-subject (LOSO) ----
        t = time.time()
        yp = cross_val_predict(
            clone(mdl), d.X, d.y, groups=d.groups, cv=LeaveOneGroupOut(), n_jobs=1
        )
        preds[name] = yp
        for s in np.unique(d.groups):
            m = d.groups == s
            rows.append(
                dict(model=name, protocol="cross-subject (LOSO)", subject=int(s),
                     **scores_for(d.y[m], yp[m]))
            )
        loso_k = cohen_kappa_score(d.y, yp)
        print(f"[classical] {name:16s} LOSO pooled kappa={loso_k:.3f} ({time.time()-t:.0f}s)", flush=True)

        # ---- within-subject ----
        t = time.time()
        for s in np.unique(d.groups):
            m = d.groups == s
            cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
            ypw = cross_val_predict(clone(mdl), d.X[m], d.y[m], cv=cv, n_jobs=1)
            rows.append(
                dict(model=name, protocol="within-subject", subject=int(s),
                     **scores_for(d.y[m], ypw))
            )
        print(f"[classical] {name:16s} within done ({time.time()-t:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS / f"classical_per_subject{TAG}.csv", index=False)
    np.savez_compressed(
        RESULTS / f"classical_loso_preds{TAG}.npz",
        y_true=d.y, groups=d.groups, **{k: v for k, v in preds.items()}
    )
    print(f"[classical] saved -> results/classical_per_subject{TAG}.csv", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# deep
# --------------------------------------------------------------------------- #
def zscore_per_trial(X):
    """Stateless per-trial, per-channel standardization. Cannot leak."""
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True) + 1e-7
    return (X - mu) / sd


DEEP_BUILDERS = {
    "EEGNet": "build_eegnet",
    "ShallowConvNet": "build_shallow",
    "DeepConvNet": "build_deep",
    "TinyLSTM": "build_lstm",
    "TinyTransformer": "build_transformer",
}


def _splits(d, protocol):
    """Yield (label, train_idx, test_idx) for the requested protocol."""
    idx = np.arange(len(d.y))
    if protocol == "loso":
        for tr, te in LeaveOneGroupOut().split(idx, d.y, groups=d.groups):
            yield f"heldout=S{int(np.unique(d.groups[te])[0])}", tr, te
    elif protocol == "crosssession":
        sess = np.asarray(d.metadata["session"])
        for s in np.unique(d.groups):
            tr = idx[(d.groups == s) & (sess == "0train")]
            te = idx[(d.groups == s) & (sess == "1test")]
            yield f"S{int(s)} T->E", tr, te
    elif protocol == "within":
        for s in np.unique(d.groups):
            sub = idx[d.groups == s]
            cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
            for k, (a, b) in enumerate(cv.split(sub, d.y[sub])):
                yield f"S{int(s)} fold{k+1}", sub[a], sub[b]
    else:
        raise ValueError(protocol)


PROTOCOL_NAMES = {
    "loso": "cross-subject (LOSO)",
    "crosssession": "cross-session (T->E, official)",
    "within": "within-subject",
}


def run_deep(d, decimate=2, max_epochs=250, protocol="crosssession", patience=40):
    from eeglog import models_deep as md

    X = zscore_per_trial(d.X)[:, :, ::decimate]
    classes = np.unique(d.y)
    y_int = np.searchsorted(classes, d.y).astype(np.int64)
    n_chans, n_times = X.shape[1], X.shape[2]
    folds = list(_splits(d, protocol))
    print(f"[deep] input {X.shape} (decimate={decimate}) classes={list(classes)} "
          f"protocol={protocol} n_folds={len(folds)}", flush=True)

    rows, preds = [], {}
    for name, builder in DEEP_BUILDERS.items():
        yp = np.full_like(y_int, -1)
        t0 = time.time()
        for i, (label, tr, te) in enumerate(folds):
            t = time.time()
            # fresh model every fold -- no weight carry-over between folds
            clf = getattr(md, builder)(n_chans, n_times, len(classes),
                                       max_epochs=max_epochs, patience=patience,
                                       random_state=SEED)
            clf.fit(X[tr], y_int[tr])
            yp[te] = clf.predict(X[te])
            print(f"[deep] {name:16s} {i+1}/{len(folds)} {label:14s} "
                  f"kappa={cohen_kappa_score(y_int[te], yp[te]):.3f} "
                  f"({time.time()-t:.0f}s)", flush=True)
        scored = yp >= 0
        preds[name] = classes[np.where(scored, yp, 0)]
        for s in np.unique(d.groups):
            m = (d.groups == s) & scored
            if m.sum():
                rows.append(dict(model=name, protocol=PROTOCOL_NAMES[protocol],
                                 subject=int(s), **scores_for(y_int[m], yp[m])))
        print(f"[deep] {name:16s} DONE pooled kappa="
              f"{cohen_kappa_score(y_int[scored], yp[scored]):.3f} "
              f"({time.time()-t0:.0f}s)\n", flush=True)
        # checkpoint after each model so a crash never loses finished work
        pd.DataFrame(rows).to_csv(RESULTS / f"deep_per_subject{TAG}.csv", index=False)
        np.savez_compressed(RESULTS / f"deep_loso_preds{TAG}.npz",
                            y_true=d.y[scored], groups=d.groups[scored],
                            **{k: v[scored] for k, v in preds.items()})
    print(f"[deep] saved -> results/deep_per_subject{TAG}.csv", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# leakage demo
# --------------------------------------------------------------------------- #
def run_leakage(d):
    from eeglog.evaluation import leakage_demo
    from eeglog.models_classic import csp_lda

    out = leakage_demo(csp_lda, d.X, d.y, d.groups)
    rec = {
        "leaky_balanced_accuracy": out["leaky_mean"],
        "honest_balanced_accuracy": out["honest_mean"],
        "inflation": out["inflation"],
    }
    (RESULTS / f"leakage_demo{TAG}.json").write_text(json.dumps(rec, indent=2))
    print(f"[leakage] leaky={rec['leaky_balanced_accuracy']:.3f} "
          f"honest={rec['honest_balanced_accuracy']:.3f} "
          f"inflation=+{rec['inflation']:.3f}", flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["classical", "crosssession", "deep", "leakage", "all"],
                    default="all")
    ap.add_argument("--deep-protocol", choices=["loso", "crosssession", "within"],
                    default="crosssession")
    ap.add_argument("--dataset", choices=list(DATASETS), default="iv2a")
    ap.add_argument("--subjects", type=int, default=None,
                    help="use the first N subjects of the preset")
    ap.add_argument("--tag", default=None, help="suffix for output filenames")
    ap.add_argument("--decimate", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=250)
    ap.add_argument("--patience", type=int, default=40)
    a = ap.parse_args()

    if a.tag:
        TAG = f"_{a.tag}"
    else:
        TAG = f"_{a.dataset}"
    globals()["TAG"] = TAG

    np.random.seed(SEED)
    subs = DATASETS[a.dataset]["subjects"]
    if a.subjects:
        subs = subs[: a.subjects]
    d = load(a.dataset, subs)
    if a.phase in ("classical", "all"):
        run_classical(d)
    if a.phase in ("crosssession", "all"):
        run_cross_session(d)
    if a.phase in ("leakage", "all"):
        run_leakage(d)
    if a.phase in ("deep", "all"):
        run_deep(d, decimate=a.decimate, max_epochs=a.max_epochs,
                 protocol=a.deep_protocol, patience=a.patience)
