"""Load real public EEG datasets *by code* — nothing downloaded by hand.

Two tracks:

* :func:`load_moabb` — the high-level track. MOABB fetches a benchmark dataset
  (default BCI Competition IV-2a) and a paradigm turns it into ready-to-model
  ``(X, y, metadata)`` arrays. The ``metadata`` frame carries the ``subject``
  and ``session`` columns we need for leakage-safe, subject-independent CV.

* :func:`load_eegbci_raw` — the hand-built track. We pull raw EDF from PhysioNet
  and return an MNE ``Raw`` object so the preprocessing notebook can do every
  step itself (filter -> ICA -> epoch).

All downloads cache under ``~/mne_data`` / ``~/.mne``; data is never committed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LoadedData:
    """Container for paradigm-extracted epochs.

    Attributes
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Epoched signal, already band-passed and epoched by the paradigm.
    y : ndarray of str, shape (n_epochs,)
        Class label per epoch (e.g. ``"left_hand"`` / ``"right_hand"``).
    groups : ndarray, shape (n_epochs,)
        Subject id per epoch — pass as ``groups=`` to grouped CV splitters.
    metadata : pandas.DataFrame
        Full MOABB metadata (subject, session, run) for richer splits.
    sfreq : float
        Sampling rate of ``X`` in Hz.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    metadata: "object"  # pandas.DataFrame (kept untyped to avoid import cost here)
    sfreq: float

    @property
    def n_classes(self) -> int:
        return len(np.unique(self.y))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LoadedData(X={self.X.shape}, n_classes={self.n_classes}, "
            f"subjects={sorted(np.unique(self.groups).tolist())}, sfreq={self.sfreq})"
        )


def get_dataset(name: str = "BNCI2014_001"):
    """Return an instantiated MOABB dataset by name.

    Defaults to BCI Competition IV-2a (``BNCI2014_001``): 9 subjects, 22 EEG
    channels, 4 motor-imagery classes — the standard, laptop-sized MI benchmark.
    """
    import moabb.datasets as mds

    try:
        cls = getattr(mds, name)
    except AttributeError as exc:  # pragma: no cover - user error path
        raise ValueError(
            f"Unknown MOABB dataset {name!r}. Examples: 'BNCI2014_001' (IV-2a), "
            f"'BNCI2014_004' (IV-2b), 'PhysionetMI'."
        ) from exc
    return cls()


def get_paradigm(kind: str = "left_right", fmin: float = 8.0, fmax: float = 30.0):
    """Return a MOABB paradigm.

    Parameters
    ----------
    kind : {"left_right", "motor_imagery"}
        ``"left_right"`` -> 2-class ``LeftRightImagery`` (fast; good first pass).
        ``"motor_imagery"`` -> all-class ``MotorImagery``.
    fmin, fmax : float
        Band-pass applied by the paradigm. 8-30 Hz captures mu+beta, where
        motor-imagery ERD/ERS lives.
    """
    from moabb.paradigms import LeftRightImagery, MotorImagery

    if kind == "left_right":
        return LeftRightImagery(fmin=fmin, fmax=fmax)
    if kind == "motor_imagery":
        return MotorImagery(fmin=fmin, fmax=fmax)
    raise ValueError(f"kind must be 'left_right' or 'motor_imagery', got {kind!r}")


def load_moabb(
    dataset: str = "BNCI2014_001",
    subjects: list[int] | None = None,
    paradigm: str = "left_right",
    fmin: float = 8.0,
    fmax: float = 30.0,
) -> LoadedData:
    """Fetch and epoch a MOABB dataset in one call.

    Parameters
    ----------
    dataset : str
        MOABB dataset class name (see :func:`get_dataset`).
    subjects : list of int, optional
        Subjects to load. Default ``[1, 2, 3]`` — enough for a real
        cross-subject split while staying fast on a laptop.
    paradigm : {"left_right", "motor_imagery"}
        See :func:`get_paradigm`.

    Returns
    -------
    LoadedData
    """
    if subjects is None:
        subjects = [1, 2, 3]

    ds = get_dataset(dataset)
    par = get_paradigm(paradigm, fmin=fmin, fmax=fmax)

    X, y, metadata = par.get_data(dataset=ds, subjects=subjects)
    groups = metadata["subject"].to_numpy()
    sfreq = _infer_sfreq(par, ds)

    return LoadedData(
        X=np.asarray(X),
        y=np.asarray(y),
        groups=groups,
        metadata=metadata,
        sfreq=sfreq,
    )


def _infer_sfreq(paradigm, dataset) -> float:
    """Best-effort sampling-rate lookup across MOABB versions."""
    for attr in ("resample", "sfreq"):
        val = getattr(paradigm, attr, None)
        if val:
            return float(val)
    # Fall back to the dataset's native rate via a single-subject probe.
    raw_dict = dataset.get_data(subjects=[dataset.subject_list[0]])
    # Walk the nested {subject: {session: {run: Raw}}} structure to the first Raw.
    node = raw_dict
    while isinstance(node, dict):
        node = next(iter(node.values()))
    return float(node.info["sfreq"])


def load_eegbci_raw(subject: int = 1, runs: list[int] | None = None):
    """Load raw PhysioNet Motor-Movement/Imagery EDF as one concatenated MNE Raw.

    This is the hand-built track: you get an unprocessed ``Raw`` and do filtering,
    ICA and epoching yourself in :mod:`eeglog.preprocess`.

    Parameters
    ----------
    subject : int
        PhysioNet subject id (1-109). Note subject 88 is sampled differently.
    runs : list of int, optional
        Run numbers. Default ``[4, 8, 12]`` = the imagined left/right fist runs.
    """
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    if runs is None:
        runs = [4, 8, 12]  # imagined left vs right fist

    paths = eegbci.load_data(subject, runs, update_path=True)
    raw = concatenate_raws([read_raw_edf(p, preload=True, verbose="ERROR") for p in paths])

    # PhysioNet channel names carry trailing dots ("Fc5.") — standardize so a
    # standard montage maps cleanly.
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="warn")
    return raw
