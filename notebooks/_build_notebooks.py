"""Generate the six learning-log notebooks with nbformat.

Run from the repo root::

    .venv/bin/python notebooks/_build_notebooks.py

Keeping the notebook *content* here as plain Python makes it easy to read and
re-generate. Each notebook is short: narration (markdown) + a few cells that
import from ``src/eeglog`` and end in a visible artifact (a plot or a number).
"""

from __future__ import annotations

import pathlib

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).resolve().parent

# Every notebook starts by putting src/ on the path and enabling autoreload.
BOOT = """\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
%load_ext autoreload
%autoreload 2
import numpy as np, matplotlib.pyplot as plt
"""


def build(name: str, cells: list):
    nb = new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = HERE / name
    nbf.write(nb, out)
    print("wrote", out.relative_to(HERE.parent))


# --------------------------------------------------------------------------- #
# 01 — Load & explore
# --------------------------------------------------------------------------- #
build(
    "01_load_and_explore.ipynb",
    [
        new_markdown_cell(
            "# 01 · Load real EEG by code\n\n"
            "**Goal:** fetch a real motor-imagery dataset with *zero* manual "
            "downloads, and get a feel for what EEG data actually is.\n\n"
            "We use MOABB's `BNCI2014_001` (BCI Competition IV-2a): 9 subjects, "
            "22 EEG channels, motor imagery. MOABB downloads + caches it under "
            "`~/mne_data` on first call."
        ),
        new_code_cell(BOOT),
        new_code_cell(
            "from eeglog.data import load_moabb\n"
            "# Start with 3 subjects, left-vs-right (2-class) for speed.\n"
            "d = load_moabb(subjects=[1, 2, 3], paradigm='left_right')\n"
            "d"
        ),
        new_markdown_cell(
            "`X` is `(n_epochs, n_channels, n_times)`. `groups` holds the subject "
            "id per epoch — that's what makes subject-independent CV possible later."
        ),
        new_code_cell(
            "print('X:', d.X.shape, '| sfreq:', d.sfreq, 'Hz')\n"
            "print('classes:', sorted(set(d.y)))\n"
            "# Class balance — note any imbalance now so we pick the right metric later.\n"
            "import collections; print(collections.Counter(d.y))"
        ),
        new_code_cell(
            "# Artifact: plot one trial (a few channels) and the class counts.\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11, 3))\n"
            "t = np.arange(d.X.shape[-1]) / d.sfreq\n"
            "for ch in range(5):\n"
            "    ax[0].plot(t, d.X[0, ch] * 1e6 + ch * 30, lw=.6)\n"
            "ax[0].set(title=f'One trial ({d.y[0]})', xlabel='s', ylabel='µV (offset)')\n"
            "vals, cnts = np.unique(d.y, return_counts=True)\n"
            "ax[1].bar(vals, cnts); ax[1].set_title('class balance')\n"
            "plt.tight_layout()"
        ),
        new_markdown_cell(
            "**Checkpoint:** data loaded by code, shape understood, class balance "
            "noted. Next: do the preprocessing *by hand* on raw PhysioNet data."
        ),
    ],
)

# --------------------------------------------------------------------------- #
# 02 — Preprocess: filter, ICA, epoch
# --------------------------------------------------------------------------- #
build(
    "02_preprocess_ica_epoch.ipynb",
    [
        new_markdown_cell(
            "# 02 · Preprocess by hand — filter, denoise (ICA), epoch\n\n"
            "MOABB hides preprocessing. Here we do every step ourselves on raw "
            "PhysioNet EDF so the **order** and the **gotchas** stick:\n\n"
            "montage → resample → band-pass → notch → **ICA** → re-reference → "
            "epoch → (autoreject).\n\n"
            "Key rules: filter the *continuous* signal before epoching; fit ICA on "
            "a 1 Hz high-passed copy; anything that learns from data (ICA, "
            "autoreject) must later be fit inside CV folds (see notebook 06)."
        ),
        new_code_cell(BOOT),
        new_code_cell(
            "from eeglog.data import load_eegbci_raw\n"
            "from eeglog import preprocess as pp\n"
            "raw = load_eegbci_raw(subject=1, runs=[4, 8, 12])  # imagined L/R fist\n"
            "raw"
        ),
        new_code_cell(
            "# Before/after band-pass: PSD shows the band we keep (8–30 Hz).\n"
            "raw_clean = pp.preprocess_raw(raw, l_freq=8., h_freq=30., do_ica=True)\n"
            "fig = raw_clean.compute_psd(fmax=60).plot(show=False)"
        ),
        new_code_cell(
            "# Inspect the ICA we fitted (stashed on the cleaned raw).\n"
            "ica = raw_clean.info['temp']['ica']\n"
            "print('excluded components:', ica.exclude)\n"
            "fig = ica.plot_components(show=False) if ica.exclude else None"
        ),
        new_code_cell(
            "# Epoch the cleaned continuous data, then get (X, y).\n"
            "epochs = pp.make_epochs(raw_clean, tmin=0.5, tmax=3.5)\n"
            "X, y = pp.epochs_to_xy(epochs)\n"
            "print(X.shape, sorted(set(y)))\n"
            "fig = epochs.average().plot(show=False)"
        ),
        new_markdown_cell(
            "**Checkpoint:** raw → cleaned → epoched, by hand. You saw the band "
            "we kept, the artifact components removed, and the evoked response."
        ),
    ],
)

# --------------------------------------------------------------------------- #
# 03 — Features
# --------------------------------------------------------------------------- #
build(
    "03_features.ipynb",
    [
        new_markdown_cell(
            "# 03 · Feature families\n\n"
            "Five ways to turn epochs into something a classifier can use:\n\n"
            "- **Time:** Hjorth params, moments, entropy\n"
            "- **Frequency:** log band power (δ θ α β γ)\n"
            "- **Connectivity:** phase-locking between channels\n"
            "- **CSP / FBCSP:** spatial filters tuned to the discriminative bands\n"
            "- **Riemannian:** trial covariance matrices on the SPD manifold\n\n"
            "Each is an sklearn transformer, so it stays *inside* the pipeline → "
            "no leakage."
        ),
        new_code_cell(BOOT),
        new_code_cell(
            "from eeglog.data import load_moabb\n"
            "from eeglog import features as F\n"
            "d = load_moabb(subjects=[1], paradigm='left_right')\n"
            "X, y, sf = d.X, d.y, d.sfreq"
        ),
        new_code_cell(
            "# Time + frequency feature matrices.\n"
            "Xt = F.TimeDomainFeatures(sf).fit_transform(X)\n"
            "Xf = F.BandPowerFeatures(sf).fit_transform(X)\n"
            "print('time:', Xt.shape, '| bandpower:', Xf.shape)"
        ),
        new_code_cell(
            "# CSP spatial patterns — the classic MI artifact. Topomaps differ by class.\n"
            "import mne\n"
            "csp = F.make_csp(n_components=4).fit(X, y)\n"
            "info = mne.create_info([f'C{i}' for i in range(X.shape[1])], sf, 'eeg')\n"
            "# (For real topomaps use the dataset montage; here we just show CSP works.)\n"
            "print('CSP feature shape:', csp.transform(X).shape)"
        ),
        new_code_cell(
            "# Band power by class — alpha/beta desync should differ left vs right.\n"
            "import numpy as np\n"
            "for cls in sorted(set(y)):\n"
            "    plt.plot(Xf[y == cls].mean(0), label=cls, lw=1)\n"
            "plt.legend(); plt.title('mean log band power by class'); plt.xlabel('feature idx')"
        ),
        new_markdown_cell(
            "**Checkpoint:** five feature families, each an sklearn transformer. "
            "Next we plug them into classical models."
        ),
    ],
)

# --------------------------------------------------------------------------- #
# 04 — Classical models
# --------------------------------------------------------------------------- #
build(
    "04_classical_models.ipynb",
    [
        new_markdown_cell(
            "# 04 · Classical models\n\n"
            "Five complete pipelines (features + classifier in one estimator):\n"
            "`CSP+LDA`, `FBCSP+SVM`, `Riemann+TS+LR`, `Riemann+MDM`, `BandPower+RF`.\n\n"
            "We score them *within-subject* here for a quick read; the honest "
            "cross-subject ranking comes in notebook 06."
        ),
        new_code_cell(BOOT),
        new_code_cell(
            "from eeglog.data import load_moabb\n"
            "from eeglog.models_classic import all_classical\n"
            "from eeglog import evaluation as E\n"
            "d = load_moabb(subjects=[1, 2, 3], paradigm='left_right')\n"
            "models = all_classical(d.sfreq)\n"
            "list(models)"
        ),
        new_code_cell(
            "results = []\n"
            "for name, model in models.items():\n"
            "    r = E.evaluate_within_subject(model, d.X, d.y, d.groups, model_name=name)\n"
            "    print(r); results.append(r)"
        ),
        new_code_cell(
            "lb = E.leaderboard(results); display(lb)\n"
            "plt.barh(lb['model'], lb['mean'], xerr=lb['std']); plt.xlabel('balanced acc')\n"
            "plt.title('within-subject (NOT the headline)'); plt.gca().invert_yaxis()"
        ),
        new_markdown_cell(
            "**Checkpoint:** a leaderboard of classical models. Riemannian methods "
            "usually lead. Remember: these are *within-subject* (optimistic)."
        ),
    ],
)

# --------------------------------------------------------------------------- #
# 05 — Deep models
# --------------------------------------------------------------------------- #
build(
    "05_deep_models.ipynb",
    [
        new_markdown_cell(
            "# 05 · Deep models on CPU\n\n"
            "EEGNet, ShallowConvNet, DeepConvNet, a tiny LSTM and a tiny "
            "Transformer — all via braindecode/skorch, all trainable on CPU in "
            "minutes thanks to small inputs and few epochs.\n\n"
            "Per-channel scaling is fit on train only (do it inside a pipeline or "
            "on the train split)."
        ),
        new_code_cell(BOOT),
        new_code_cell(
            "from eeglog.data import load_moabb\n"
            "from eeglog.models_deep import build_eegnet, as_float32\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.metrics import balanced_accuracy_score\n"
            "d = load_moabb(subjects=[1], paradigm='left_right')\n"
            "X = as_float32(d.X); y = (d.y == d.y[0]).astype('int64')  # encode to 0/1\n"
            "n_ch, n_t = X.shape[1], X.shape[2]"
        ),
        new_code_cell(
            "Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)\n"
            "clf = build_eegnet(n_ch, n_t, n_classes=2, max_epochs=30)\n"
            "clf.fit(Xtr, ytr)\n"
            "print('balanced acc:', balanced_accuracy_score(yte, clf.predict(Xte)))"
        ),
        new_code_cell(
            "# Training/valid loss curve from skorch history.\n"
            "h = clf.history\n"
            "plt.plot(h[:, 'train_loss'], label='train')\n"
            "plt.plot(h[:, 'valid_loss'], label='valid')\n"
            "plt.legend(); plt.xlabel('epoch'); plt.title('EEGNet learning curve')"
        ),
        new_markdown_cell(
            "**Checkpoint:** a deep net trained on CPU with a sane learning curve. "
            "Swap in `build_shallow / build_deep` to compare CNNs. For the "
            "recurrent/attention nets, decimate the time axis first so they stay "
            "fast on CPU:\n\n"
            "```python\n"
            "from eeglog.models_deep import build_transformer, decimate_time\n"
            "Xd = decimate_time(d.X, factor=4)   # 1001 -> 251 samples\n"
            "clf = build_transformer(Xd.shape[1], Xd.shape[2], 2)\n"
            "```\n\n"
            "The honest cross-subject comparison is in notebook 06."
        ),
    ],
)

# --------------------------------------------------------------------------- #
# 06 — Honest evaluation
# --------------------------------------------------------------------------- #
build(
    "06_honest_evaluation.ipynb",
    [
        new_markdown_cell(
            "# 06 · Honest evaluation — the whole point\n\n"
            "Rules: subject-level splits, every learned step fit inside the fold, "
            "**cross-subject (LOSO)** headline as mean ± std, imbalance-aware "
            "metrics + confusion matrix, protocol tagged on every number.\n\n"
            "First we *prove* leakage inflates scores, then we report the honest "
            "leaderboard."
        ),
        new_code_cell(BOOT),
        new_code_cell(
            "from eeglog.data import load_moabb\n"
            "from eeglog.models_classic import csp_lda, all_classical\n"
            "from eeglog import evaluation as E\n"
            "d = load_moabb(subjects=[1, 2, 3], paradigm='left_right')"
        ),
        new_markdown_cell("### The leakage trap, made concrete"),
        new_code_cell(
            "demo = E.leakage_demo(csp_lda, d.X, d.y, d.groups)\n"
            "print(f\"leaky  CSP-on-all-data : {demo['leaky_mean']:.3f}\")\n"
            "print(f\"honest CSP-in-fold     : {demo['honest_mean']:.3f}\")\n"
            "print(f\"inflation              : +{demo['inflation']:.3f}\")"
        ),
        new_markdown_cell("### Within-subject vs cross-subject (same model)"),
        new_code_cell(
            "m = csp_lda()\n"
            "within = E.evaluate_within_subject(m, d.X, d.y, d.groups, model_name='CSP+LDA')\n"
            "loso = E.evaluate_loso(csp_lda(), d.X, d.y, d.groups, model_name='CSP+LDA')\n"
            "print(within); print(loso)  # cross-subject is lower — and honest"
        ),
        new_markdown_cell("### Honest headline leaderboard (LOSO, mean ± std)"),
        new_code_cell(
            "results = [E.evaluate_loso(mdl, d.X, d.y, d.groups, model_name=n)\n"
            "           for n, mdl in all_classical(d.sfreq).items()]\n"
            "lb = E.leaderboard(results); display(lb)"
        ),
        new_code_cell(
            "# Confusion matrix for the top model.\n"
            "best = all_classical(d.sfreq)[lb.iloc[0]['model']]\n"
            "cm, labels = E.confusion(best, d.X, d.y, d.groups)\n"
            "import matplotlib.pyplot as plt\n"
            "plt.imshow(cm, cmap='Blues'); plt.xticks(range(len(labels)), labels, rotation=45)\n"
            "plt.yticks(range(len(labels)), labels); plt.title(f\"LOSO confusion — {lb.iloc[0]['model']}\")\n"
            "plt.colorbar()"
        ),
        new_markdown_cell(
            "**What would be dishonest here:** quoting the within-subject or "
            "leaky number as the result; reporting raw accuracy on imbalanced "
            "classes; or fitting CSP/scaler/ICA on the full dataset before CV. "
            "The headline is the **LOSO mean ± std** above."
        ),
    ],
)

print("done.")
