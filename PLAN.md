# EEG Learning Log — Motor-Imagery BCI from Scratch

## Context

`eeg-learning-log` is currently an empty repo (README + `.gitignore` + a planned `.venv`). The goal is a **learning project**: work end-to-end through a real EEG/BCI pipeline — load public data by code, preprocess it correctly, engineer the classic feature families, train both classical and deep models, and — most importantly — **evaluate honestly** (no leakage, subject-independent headline metric, mean ± std, imbalance-aware metrics).

Two decisions from you shape this plan:
- **Learn memorably/effectively** → a **hybrid layout**: a small reusable `src/eeglog/` package (the mechanics, written once, well) + **numbered notebooks** that are the narrative "log." Every stage ends with a *visible artifact* (a plot or a number) so each concept is anchored to something you saw.
- **Hand-build the core, use MOABB for the headline** → you implement filtering / ICA / epoching / CSP / features by hand with MNE + sklearn to learn the plumbing, then use **MOABB's `CrossSubject` evaluation** as the leakage-proof, comparable-to-published benchmark.

Everything is sized to run on a **laptop CPU in minutes** (small datasets, compact nets, cropped windows).

---

## What you'll be able to do at the end
1. Fetch real EEG by code — `MOABB` / `mne.datasets.eegbci`, nothing downloaded by hand.
2. Filter, notch, re-reference, run **ICA** artifact removal, and **epoch** trials correctly (right order, no edge/causality mistakes).
3. Build **time**-, **frequency**-, **connectivity**-, **CSP**-, and **Riemannian** feature families.
4. Train classical models (LDA / SVM / LogReg / RF) and deep nets (**EEGNet, ShallowConvNet, DeepConvNet, LSTM, tiny Transformer**) on CPU in minutes.
5. Evaluate without fooling yourself: leakage-safe CV, **Leave-One-Subject-Out** headline, mean ± std, balanced-accuracy / kappa / confusion matrices.

---

## Tech stack (all pip-installable, CPU-only)

| Layer | Packages |
|---|---|
| Core / IO / preprocessing | `mne`, `moabb` |
| Artifact handling | `mne-icalabel`, `autoreject` |
| Features | `mne-features`, `antropy`, `mne-connectivity`, `pyriemann` |
| Classical ML | `scikit-learn`, `scipy`, `numpy` |
| Deep learning | `torch` (CPU), `skorch`, `braindecode` |
| Notebooks / viz | `jupyterlab`, `matplotlib`, `seaborn`, `pandas` |

Pin into `requirements.txt`. Recommended floors from research: `mne>=1.12`, `moabb>=1.5`, `pyriemann>=0.11`, `mne-connectivity>=0.8`, `braindecode>=1.6`.

---

## Dataset choice

**Primary: `moabb.datasets.BNCI2014_001`** (BCI Competition IV-2a) — 9 subjects, 22 EEG ch, 4 classes (left/right hand, feet, tongue), ~20–30 MB/subject. The standard MI benchmark; small enough for a laptop. Start with 2-class **left-vs-right** (`LeftRightImagery`) for speed, then go 4-class.

**Secondary for variety:**
- `mne.datasets.eegbci` (PhysioNet MMI) — used for the *hand-built* preprocessing notebook (raw EDF → you do every step yourself). Load a few subjects only.
- `BNCI2014_004` (2b, 3 ch) as a quick "does it generalize to another dataset" check.

Data caches automatically under `~/mne_data` / `~/.mne/data`. **Never commit data** (already covered by `.gitignore`; verify).

---

## Repo structure

```
eeg-learning-log/
├── README.md
├── requirements.txt
├── src/eeglog/
│   ├── data.py          # dataset loaders (MOABB + eegbci wrappers)
│   ├── preprocess.py    # filter, notch, resample, reref, ICA, epoching
│   ├── features.py      # time / freq / connectivity / CSP / Riemannian extractors
│   ├── models_classic.py# sklearn pipeline factories
│   ├── models_deep.py   # braindecode/skorch model factories + LSTM/Transformer
│   └── evaluation.py    # LOSO splitters, metrics, mean±std reporting, MOABB glue
├── notebooks/
│   ├── 01_load_and_explore.ipynb
│   ├── 02_preprocess_ica_epoch.ipynb
│   ├── 03_features.ipynb
│   ├── 04_classical_models.ipynb
│   ├── 05_deep_models.ipynb
│   └── 06_honest_evaluation.ipynb
├── results/             # saved figures + a results.csv leaderboard (gitignored data, kept figs)
└── tests/               # small sanity tests for src/eeglog
```

Pattern: **each notebook imports from `src/eeglog`** and is mostly narration + plots + a checkpoint. The reusable logic lives in the package so it's testable and reused across notebooks.

---

## Build plan (phased — each phase = one module + one notebook + one artifact)

### Phase 0 — Environment
- Update `README.md` setup; write `requirements.txt`; create the `src/eeglog/` skeleton and `notebooks/`.
- **Checkpoint:** `import mne, moabb, braindecode, pyriemann` succeeds; `python -c "import torch; print(torch.__version__)"` is CPU build.

### Phase 1 — Load real data by code → `data.py` + `01_load_and_explore.ipynb`
- `load_moabb(dataset, subjects, paradigm)` wrapping `BNCI2014_001()` + `LeftRightImagery`/`MotorImagery` → `(X, y, metadata)` via `paradigm.get_data(...)`. Keep the `metadata` (it carries the **subject** column needed for LOSO groups).
- `load_eegbci_raw(subject, runs)` using `mne.datasets.eegbci.load_data` + `mne.io.read_raw_edf` for the hand-built track.
- **Learn:** Raw vs Epochs, montages, channels, sampling rate, the events/annotations model.
- **Artifact:** raw trace plot, montage plot, class-count bar chart (note any imbalance).

### Phase 2 — Preprocess by hand → `preprocess.py` + `02_preprocess_ica_epoch.ipynb`
Correct order (this is a core "do it right" lesson):
1. set montage → 2. resample (early, e.g. to 128 Hz) → 3. band-pass (e.g. 8–30 Hz for MI, or 0.5–40 for general) → 4. notch (50/60 Hz) → 5. **ICA** fit on a 1–40 Hz copy, auto-flag artifacts via `find_bads_eog` and/or `mne_icalabel.label_components`, then `ica.apply` → 6. re-reference (average) → 7. **epoch** (`mne.Epochs`, `events_from_annotations`, `tmin/tmax`, baseline) → 8. optional `autoreject`.
- **Learn (the gotchas):** filter on continuous Raw *before* epoching; fit ICA on filtered data; not every dataset has EOG channels; baseline windows; why these are choices, not magic numbers.
- **Artifact:** before/after PSD, ICA component topomaps with flagged artifacts, epochs image / ERD-ERS plot showing the mu/beta desynchronization.
- **Leakage note seeded here:** ICA/autoreject parameters that *learn* from data must later be fit inside CV folds — flag this explicitly in the notebook so Phase 6 isn't a surprise.

### Phase 3 — Feature families → `features.py` + `03_features.ipynb`
One extractor per family, each returning a `(n_epochs, n_features)` matrix or an sklearn-compatible transformer:
- **Time:** Hjorth (`mne_features.univariate.compute_hjorth_*`), statistical moments; entropy via `antropy` (`spectral_entropy`, `perm_entropy`).
- **Frequency:** band power (delta/theta/alpha/beta/gamma) via `mne_features.univariate.compute_pow_freq_bands` (or `mne.time_frequency.psd_array_welch`), relative power, log-transform.
- **Connectivity:** `mne_connectivity.spectral_connectivity_epochs` (`coh`, `plv`, `pli`, `wpli`); flatten the upper triangle per band into a vector.
- **CSP:** `mne.decoding.CSP(n_components=4, log=True, reg='ledoit_wolf')`; plus a small **FBCSP** helper (CSP per filter band, concatenate).
- **Riemannian:** `pyriemann` `Covariances(estimator='lwf')` → `TangentSpace` / `MDM`.
- **Learn:** what each family *captures* and why MI lives in mu/beta ERD/ERS; why CSP uses log-variance.
- **Artifact:** CSP spatial-pattern topomaps (left vs right hand), a band-power-by-class plot, a connectivity matrix heatmap.

### Phase 4 — Classical models → `models_classic.py` + `04_classical_models.ipynb`
Pipeline factories (fit-safe, all preprocessing/feature steps *inside* the pipeline):
- `CSP + LDA` (the classic baseline)
- `FBCSP + SVM`
- `Covariances + TangentSpace + LogisticRegression` (Riemannian, typically strongest)
- `Covariances + MDM` (parameter-free Riemannian)
- a band-power/Hjorth feature-union + RandomForest for contrast
- **Learn:** sklearn `Pipeline`/`make_pipeline` so transforms are fit on train only; quick within-subject CV first.
- **Artifact:** a `results.csv` row per model and a bar chart of within-subject accuracy.

### Phase 5 — Deep models → `models_deep.py` + `05_deep_models.ipynb`
Via `braindecode` + `skorch` `EEGClassifier` (input `(batch, n_chans, n_times)`):
- **EEGNet** (`braindecode.models.EEGNet`) — compact, ~few-K params, fastest.
- **ShallowConvNet** (`ShallowFBCSPNet`) and **DeepConvNet** (`Deep4Net`) — Schirrmeister 2017.
- **Tiny LSTM** and **tiny Transformer** (small custom `nn.Module`s wrapped in `EEGClassifier`, or `EEGConformer` scaled down) — keep hidden sizes/layers tiny.
- **CPU speed tactics:** crop to 2–4 s windows, batch 16–32, ~30–80 epochs, early stopping; pre-cache the feature/epoch arrays to disk. Expect minutes per subject, not hours.
- **Learn:** why EEGNet's depthwise-separable convs are cheap; standardize per-channel **fit on train only**.
- **Artifact:** training/val loss curves; deep-vs-classical accuracy bar chart.

### Phase 6 — Honest evaluation → `evaluation.py` + `06_honest_evaluation.ipynb`
This is the headline phase — the whole point.
- **Leakage-safe CV:** every learned step (scaler, ICA, CSP, covariance, net) fit **inside** the fold. Demonstrate the trap: show inflated score from fitting CSP on all data vs the correct in-fold score.
- **Splitters:** `sklearn.model_selection.LeaveOneGroupOut` / `GroupKFold` with `groups=subject_id`. Contrast **within-session** (inflated) vs **cross-session** vs **cross-subject (LOSO)**.
- **Headline metric = subject-independent (LOSO):** report **mean ± std across subjects**.
- **Imbalance-aware metrics:** balanced accuracy, Cohen's kappa, macro-F1, ROC-AUC (binary), confusion matrices — pick the right one and say why.
- **MOABB headline:** run `moabb.evaluations.CrossSubject` with `MotorImagery`/`LeftRightImagery` over the same pipelines to get a published-comparable number, and reconcile it with your hand-built LOSO result.
- **Artifact:** final leaderboard table (model × metric, mean ± std), per-subject breakdown, confusion matrices, and a short "what would be dishonest here" write-up.

---

## The honesty rules (pin these in the README)
1. All trials from one subject live in exactly one of {train, val, test}.
2. Fit every data-dependent transform (scaler, ICA, CSP, covariances, net) **inside** the CV fold.
3. Headline number is **cross-subject LOSO**, reported as **mean ± std**.
4. Report a chance-corrected / imbalance-aware metric (balanced acc + kappa), not raw accuracy alone, and always show a confusion matrix.
5. State the protocol next to every number (within-session vs LOSO) — never quote a within-session score as if it were the headline.

---

## Verification
- **Env:** `pip install -r requirements.txt` then a one-liner importing every package; confirm CPU torch.
- **Data by code:** run `01` end-to-end on subjects [1,2,3]; confirm download-and-cache with zero manual steps.
- **Pipeline smoke test:** `tests/` runs CSP+LDA on one subject and asserts accuracy > chance (e.g. >0.55 for 2-class) — proves the plumbing works without overfitting claims.
- **Leakage demo:** assert the "CSP fit on all data" score is suspiciously higher than the in-fold score (makes the lesson concrete and regression-proof).
- **Runtime budget:** each notebook documents its wall-clock; the heaviest (deep models, full LOSO) should finish in minutes on CPU with the cropping/subject-subset defaults.
- **Reproducibility:** fixed `random_state`/seeds; `results.csv` regenerates the leaderboard.

---

## Stretch goals (optional, after the core)
- Transfer learning / fine-tune a cross-subject net on a held-out subject's calibration trials.
- Riemannian transfer (`pyriemann` recentering) for cross-session robustness.
- Add a second paradigm (P300 or SSVEP) reusing the same evaluation harness.
- A tiny `make`/CLI to regenerate all figures + leaderboard from scratch.
