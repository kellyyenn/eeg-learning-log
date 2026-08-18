# eeg-learning-log

A hands-on learning log for EEG / brain–computer-interface machine learning, built
around **motor imagery** decoding. Everything runs on a **laptop CPU in minutes**
and loads real public data **by code** — nothing downloaded by hand.

## Results

Ten decoders on **BCI Competition IV-2a**, 22 EEG channels, 4-class motor imagery
(left hand / right hand / feet / tongue), 8–30 Hz, seed 42. Scored on the
**official competition protocol** — train on the `T` session, test on the `E`
session recorded on a different day — which is the protocol every published kappa
for this dataset refers to. Chance kappa = 0.

| model | family | kappa (mean ± std) | S1 | S2 |
|---|---|---|---|---|
| Riemann+TS+LR   | classical | **0.579 ± 0.203** | 0.722 | 0.435 |
| FBCSP+SVM       | classical | 0.546 ± 0.170 | 0.667 | 0.426 |
| CSP+LDA         | classical | 0.507 ± 0.213 | 0.657 | 0.356 |
| Riemann+MDM     | classical | 0.481 ± 0.164 | 0.597 | 0.366 |
| ShallowConvNet  | deep      | 0.352 ± 0.216 | 0.505 | 0.199 |
| EEGNet          | deep      | 0.347 ± 0.354 | 0.597 | 0.097 |
| DeepConvNet     | deep      | 0.269 ± 0.399 | 0.551 | −0.014 |
| TinyTransformer | deep      | 0.236 ± 0.262 | 0.421 | 0.051 |
| BandPower+RF    | classical | 0.176 ± 0.216 | 0.329 | 0.023 |
| TinyLSTM        | deep      | −0.007 ± 0.003 | −0.005 | −0.009 |

> **Caveat — 2 of 9 subjects.** The dataset's only host (`lampx.tugraz.at`) accepts
> the TCP connection on :443 and then drops the TLS handshake; `bnci-horizon-2020.eu`
> just 302s to the same machine. Only S1 and S2 were already cached.
> `python scripts/fetch_iv2a.py --watch` polls for the rest, and the full run is one
> command once they land. Every number here is real, but n=2 is too small for the
> means to be trustworthy — read the per-subject columns, not the average.

Full tables (including within-subject and cross-subject protocols), the
significance testing, and the figures: [results/RESULTS_iv2a.md](results/RESULTS_iv2a.md).

### What the numbers say

**Classical beats deep here, and that is the expected result.** With ~288 training
trials per subject, hand-designed spatial filters win. Riemannian tangent-space
features are the strongest — they encode the whole spatial covariance structure
rather than the handful of directions CSP keeps, and they need no training beyond
a logistic regression. This is the standard finding on IV-2a-sized data; deep nets
overtake classical pipelines when you have many subjects to pretrain across, not
288 trials.

**Subject variance dwarfs model variance.** Every model does worse on S2 than on
S1, and the deep models collapse far harder (EEGNet 0.597 → 0.097) than the
classical ones (Riemann 0.722 → 0.435). Averaging across subjects hides the thing
that actually matters for a real BCI — that some users modulate their sensorimotor
rhythm strongly and some barely do at all.

**TinyLSTM sits at chance, and that is a finding about the architecture.** Reading
out the last hidden state of an LSTM run over raw time samples throws away the
band-power structure that motor imagery lives in. The nets that work here
(ShallowConvNet, EEGNet) are the ones whose architecture encodes the physiology:
a temporal convolution that learns band-pass filters, then a depthwise spatial
convolution that is effectively a learned CSP.

**Protocol choice moves the number as much as model choice.** Shuffled
within-subject 5-fold CV gives Riemann+TS+LR 0.642; the honest held-out session
gives 0.579; and cross-subject collapses to near zero. Same model, same data.

## What it does

1. **Load real data by code** — MOABB (`BNCI2014_001` / BCI Competition IV-2a) and
   `mne.datasets.eegbci` (PhysioNet). Auto-downloads + caches to `~/mne_data`.
2. **Preprocess by hand** — filter, notch, resample, re-reference, **ICA** artifact
   removal, and epoching, done step-by-step so the order and gotchas stick.
3. **Five feature families** — time (Hjorth/entropy), frequency (band power),
   connectivity (PLV/coherence), **CSP/FBCSP**, and **Riemannian** covariances.
4. **Classical + deep models** — LDA / SVM / LogReg / RF and EEGNet, ShallowConvNet,
   DeepConvNet, a tiny LSTM, and a tiny Transformer (braindecode + skorch).
5. **Honest evaluation** — leakage-safe CV, subject-independent (LOSO) headline
   metric, mean ± std, imbalance-aware metrics, confusion matrices.

## The honesty rules

1. All trials from one subject live in exactly one of {train, val, test}.
2. Fit every data-dependent transform (scaler, ICA, CSP, covariances, net)
   **inside** the CV fold — done for free by passing whole pipelines to sklearn CV.
3. The headline number is **cross-subject LOSO**, reported as **mean ± std** —
   falling back to the held-out-session protocol when too few subjects are
   available for LOSO to mean anything (see the caveat under Results).
4. Report a chance-corrected / imbalance-aware metric (balanced accuracy + kappa),
   not raw accuracy alone, and always show a confusion matrix.
5. State the protocol next to every number — never quote a within-session score as
   if it were the headline.

`eeglog.evaluation.leakage_demo` makes the classic CSP-fit-on-all-data mistake on
purpose and shows the inflated score next to the honest one.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

```
src/eeglog/        reusable mechanics (importable, tested)
  data.py          dataset loaders (MOABB + eegbci)
  preprocess.py    filter / notch / ICA / epoch
  features.py      time / freq / connectivity / CSP / Riemannian transformers
  models_classic.py sklearn pipeline factories
  models_deep.py   EEGNet / Shallow / Deep / LSTM / Transformer (skorch)
  evaluation.py    LOSO splits, metrics, leaderboard, leakage demo
notebooks/         the narrative learning log (01..06), regenerate with
                   `python notebooks/_build_notebooks.py`
tests/             smoke + leakage-demo regression tests
```

## Run it

```bash
# Tests (downloads 2 subjects on first run, ~30 s after that)
PYTHONPATH=src .venv/bin/pytest -q

# Notebooks
source .venv/bin/activate && jupyter lab   # open notebooks/01..06 in order
```

## Reproduce the results

```bash
.venv/bin/python scripts/fetch_iv2a.py --watch          # get all 9 subjects
.venv/bin/python scripts/run_benchmark.py --phase all --tag iv2a_full
.venv/bin/python scripts/make_report.py iv2a_full
```

`run_benchmark.py` takes `--dataset {iv2a,physionet}`, `--subjects N`,
`--deep-protocol {loso,crosssession,within}`, `--decimate`, `--max-epochs` and
`--patience`; it checkpoints after every model, so an interrupted run keeps
whatever finished. `make_report.py <tag>` rebuilds the leaderboard, the
significance table and all three figures from the CSVs.

Reproducing exactly what is in this README (the 2 cached subjects):

```bash
.venv/bin/python scripts/run_benchmark.py --phase all --subjects 2 --tag iv2a --deep-protocol crosssession
```

Notebooks add `src/` to the path automatically, so no install step is needed.
