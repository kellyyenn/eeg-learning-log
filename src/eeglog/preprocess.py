"""Hand-built EEG preprocessing — filter, denoise (ICA), epoch, *correctly*.

The point of this module is to do every step yourself (rather than letting MOABB
hide it) so the order and the gotchas become muscle memory:

Correct order, on continuous ``Raw`` *before* epoching:

1. set montage              (channel geometry, needed for ICA topomaps / average ref)
2. resample early           (cheaper everything downstream)
3. band-pass filter         (8-30 Hz for MI; 0.5-40 Hz general)
4. notch filter             (50/60 Hz line noise) -- skip if already band-limited
5. ICA artifact removal     (fit on a 1-40 Hz copy, flag EOG/muscle, then apply)
6. re-reference             (common average)
7. epoch                    (events -> fixed windows, with baseline)
8. (optional) autoreject    (drop/repair bad epochs)

Gotchas baked in below:
* Filter the continuous signal, not epochs (avoids edge artifacts at every trial).
* Fit ICA on a 1 Hz high-passed copy even if you keep lower frequencies in the
  data — ICA decomposition is much cleaner above ~1 Hz.
* Not every dataset has EOG channels; we fall back gracefully.

LEAKAGE WARNING: ICA and autoreject *learn* from data. Used as a fixed
exploratory step here that's fine, but inside a benchmark they must be fit on the
training fold only. The leakage-safe wrappers live in :mod:`eeglog.evaluation`.
"""

from __future__ import annotations

import numpy as np


def preprocess_raw(
    raw,
    *,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    resample_to: float | None = 128.0,
    notch: float | None = None,
    do_ica: bool = True,
    n_ica_components: float | int = 0.99,
    average_ref: bool = True,
    random_state: int = 42,
):
    """Run the full hand-built pipeline on a copy of ``raw``.

    Returns the cleaned ``Raw`` (and leaves the input untouched). Set
    ``do_ica=False`` for a fast pass without artifact removal.
    """
    import mne

    raw = raw.copy().load_data()
    raw.pick("eeg")  # work on EEG only

    if resample_to is not None and raw.info["sfreq"] != resample_to:
        raw.resample(resample_to, npad="auto")

    raw.filter(l_freq, h_freq, method="fir", phase="zero", verbose="ERROR")
    if notch is not None:
        raw.notch_filter(freqs=notch, verbose="ERROR")

    if do_ica:
        raw = apply_ica(raw, n_components=n_ica_components, random_state=random_state)

    if average_ref:
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    return raw


def apply_ica(raw, *, n_components: float | int = 0.99, random_state: int = 42):
    """Fit ICA on a 1 Hz high-passed copy, flag artifact components, apply to ``raw``.

    Detection strategy, best-available first:
    1. EOG channels via :meth:`ICA.find_bads_eog` if present.
    2. ``mne_icalabel`` ML classification (flags eye/muscle/heart/line/noise).
    If neither is available, nothing is excluded (ICA becomes a no-op) and a
    note is printed so the user knows.
    """
    from mne.preprocessing import ICA

    ica = ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),
        max_iter="auto",
        random_state=random_state,
    )
    # Fit on a 1 Hz high-passed copy for a cleaner decomposition.
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="ERROR")
    ica.fit(raw_for_ica, verbose="ERROR")

    exclude: list[int] = []

    # 1) EOG-based detection if EOG channels exist.
    has_eog = any(ch == "eog" for ch in raw.get_channel_types())
    if has_eog:
        eog_idx, _ = ica.find_bads_eog(raw, verbose="ERROR")
        exclude.extend(eog_idx)

    # 2) ML labels via mne-icalabel (works best on average-referenced, 1-100 Hz
    #    data, but is informative here too).
    if not exclude:
        try:
            from mne_icalabel import label_components

            labels = label_components(raw_for_ica, ica, method="iclabel")
            for i, (lab, prob) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
                if lab not in ("brain", "other") and prob > 0.8:
                    exclude.append(i)
        except Exception as exc:  # pragma: no cover - optional dependency / version skew
            print(f"[apply_ica] mne-icalabel unavailable or failed ({exc}); excluding nothing.")

    ica.exclude = sorted(set(exclude))
    if not ica.exclude:
        print("[apply_ica] No artifact components flagged (no EOG channel / no ICLabel hit).")

    out = raw.copy()
    ica.apply(out, verbose="ERROR")
    # Stash the fitted ICA so notebooks can plot components.
    out.info["temp"] = {"ica": ica} if out.info.get("temp") is None else out.info["temp"]
    return out


def make_epochs(
    raw,
    *,
    tmin: float = 0.5,
    tmax: float = 3.5,
    baseline=None,
    event_id: dict | None = None,
    reject=None,
):
    """Epoch a (preprocessed) ``Raw`` from its annotations.

    Defaults to a 0.5-3.5 s motor-imagery window. ``baseline=None`` because the
    data is already band-passed (8-30 Hz removes the slow drift baseline
    correction would target); pass e.g. ``(None, 0)`` for broadband data.
    """
    import mne

    events, auto_event_id = mne.events_from_annotations(raw, verbose="ERROR")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id or auto_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=True,
        verbose="ERROR",
    )
    return epochs


def autoreject_epochs(epochs, *, random_state: int = 42):
    """Drop/repair bad epochs with AutoReject. LEAKAGE: fit per training fold only."""
    from autoreject import AutoReject

    ar = AutoReject(random_state=random_state, n_jobs=1, verbose=False)
    epochs_clean = ar.fit_transform(epochs)
    return epochs_clean, ar


def epochs_to_xy(epochs):
    """Return ``(X, y)`` arrays from an MNE ``Epochs`` for sklearn pipelines."""
    X = epochs.get_data(copy=False)
    # Map event integer codes back to their string names.
    inv = {v: k for k, v in epochs.event_id.items()}
    y = np.array([inv[code] for code in epochs.events[:, -1]])
    return X, y
