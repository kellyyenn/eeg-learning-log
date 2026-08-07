"""Feature families for EEG motor imagery, all as sklearn-compatible transformers.

Every extractor takes ``X`` of shape ``(n_epochs, n_channels, n_times)`` and
returns ``(n_epochs, n_features)`` (or a covariance stack for the Riemannian
path), so they slot straight into ``sklearn.pipeline.Pipeline``. That matters:
keeping feature extraction *inside* the pipeline is what makes leakage-safe CV
possible — ``CSP``/``Covariances`` learn from data and must see train-only.

Families
--------
* Time domain    : Hjorth parameters + statistical moments + spectral entropy.
* Frequency      : log band power (delta..gamma) via Welch.
* Connectivity   : pairwise phase/coherence measures, upper-triangle vectorized.
* CSP / FBCSP    : the classic spatial-filter features for MI.
* Riemannian     : trial covariance matrices (for pyriemann classifiers).

The CSP and Riemannian helpers are thin wrappers over ``mne.decoding.CSP`` and
``pyriemann`` so you use the battle-tested implementations; the time/freq/
connectivity transformers are hand-written so the mechanics are visible.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Standard EEG bands (Hz). Gamma capped at 40 to respect typical MI band-passing.
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


# --------------------------------------------------------------------------- #
# Time domain
# --------------------------------------------------------------------------- #
class TimeDomainFeatures(BaseEstimator, TransformerMixin):
    """Per-channel Hjorth parameters, statistical moments, and spectral entropy.

    Produces, per channel: mean, std, skew, kurtosis, Hjorth activity/mobility/
    complexity, and spectral entropy -> concatenated across channels.
    """

    def __init__(self, sfreq: float):
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scipy.stats import kurtosis, skew

        try:
            import antropy as ant
        except Exception:  # pragma: no cover
            ant = None

        feats = []
        for epoch in X:  # epoch: (n_channels, n_times)
            row = []
            # Hjorth on each channel.
            d1 = np.diff(epoch, axis=1)
            d2 = np.diff(d1, axis=1)
            var0 = epoch.var(axis=1) + 1e-12
            var1 = d1.var(axis=1) + 1e-12
            var2 = d2.var(axis=1) + 1e-12
            activity = var0
            mobility = np.sqrt(var1 / var0)
            complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)

            row.extend(epoch.mean(axis=1))
            row.extend(epoch.std(axis=1))
            row.extend(skew(epoch, axis=1))
            row.extend(kurtosis(epoch, axis=1))
            row.extend(activity)
            row.extend(mobility)
            row.extend(complexity)
            if ant is not None:
                row.extend(
                    ant.spectral_entropy(ch, self.sfreq, method="welch", normalize=True)
                    for ch in epoch
                )
            feats.append(row)
        return np.nan_to_num(np.asarray(feats, dtype=float))


# --------------------------------------------------------------------------- #
# Frequency domain
# --------------------------------------------------------------------------- #
class BandPowerFeatures(BaseEstimator, TransformerMixin):
    """Log band power per channel and band via Welch PSD.

    Output columns = n_channels * n_bands. Log-transform makes the (roughly
    log-normal) power values friendlier to linear classifiers.
    """

    def __init__(self, sfreq: float, bands: dict | None = None, relative: bool = False):
        self.sfreq = sfreq
        self.bands = bands or BANDS
        self.relative = relative

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scipy.signal import welch

        n_per_seg = min(256, X.shape[-1])
        freqs, psd = welch(X, fs=self.sfreq, nperseg=n_per_seg, axis=-1)
        # psd: (n_epochs, n_channels, n_freqs)
        total = psd.sum(axis=-1, keepdims=True) + 1e-12

        cols = []
        for fmin, fmax in self.bands.values():
            mask = (freqs >= fmin) & (freqs < fmax)
            band = psd[:, :, mask].sum(axis=-1)  # (n_epochs, n_channels)
            if self.relative:
                band = band / total[:, :, 0]
            cols.append(np.log(band + 1e-12))
        # Stack -> (n_epochs, n_channels * n_bands)
        return np.concatenate(cols, axis=1)


# --------------------------------------------------------------------------- #
# Connectivity
# --------------------------------------------------------------------------- #
class ConnectivityFeatures(BaseEstimator, TransformerMixin):
    """Pairwise spectral connectivity, upper-triangle vectorized.

    Uses ``mne_connectivity.spectral_connectivity_epochs``. Method defaults to
    ``"plv"`` (phase-locking value); ``"wpli"`` is more noise-robust for MI.
    """

    def __init__(self, sfreq: float, method: str = "plv", fmin: float = 8.0, fmax: float = 30.0):
        self.sfreq = sfreq
        self.method = method
        self.fmin = fmin
        self.fmax = fmax

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from mne_connectivity import spectral_connectivity_epochs

        con = spectral_connectivity_epochs(
            X,
            method=self.method,
            sfreq=self.sfreq,
            fmin=self.fmin,
            fmax=self.fmax,
            faverage=True,
            verbose="ERROR",
        )
        # get_data('dense'): (n_epochs?, n_ch, n_ch, n_bands) — but epoch-level
        # connectivity here is computed across epochs, giving one matrix. For a
        # per-epoch feature we compute connectivity per single-epoch instead.
        raise NotImplementedError(
            "spectral_connectivity_epochs aggregates across epochs; use "
            "connectivity_per_epoch() for per-trial feature vectors."
        )


def connectivity_per_epoch(
    X, sfreq: float, method: str = "plv", fmin: float = 8.0, fmax: float = 30.0
):
    """Compute one connectivity vector per epoch (upper triangle of the matrix).

    Returns ``(n_epochs, n_pairs)`` where ``n_pairs = n_ch*(n_ch-1)/2``.

    Note: connectivity needs multiple time segments per estimate; we treat each
    epoch independently by reshaping it as a single "epoch" with cwt_morlet over
    its samples. This is slower than band power — use on small channel counts.
    """
    from mne_connectivity import spectral_connectivity_epochs

    n_epochs, n_ch, _ = X.shape
    iu = np.triu_indices(n_ch, k=1)
    out = np.empty((n_epochs, len(iu[0])))
    for i, epoch in enumerate(X):
        con = spectral_connectivity_epochs(
            epoch[np.newaxis],  # (1, n_ch, n_times)
            method=method,
            mode="multitaper",
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            verbose="ERROR",
        )
        mat = con.get_data(output="dense")[:, :, 0]  # (n_ch, n_ch)
        out[i] = mat[iu]
    return out


# --------------------------------------------------------------------------- #
# CSP / FBCSP
# --------------------------------------------------------------------------- #
def make_csp(n_components: int = 4):
    """Return an ``mne.decoding.CSP`` configured for MI (log-variance features)."""
    from mne.decoding import CSP

    return CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)


class FBCSP(BaseEstimator, TransformerMixin):
    """Filter-Bank CSP: band-pass into sub-bands, CSP each, concatenate features.

    Motor imagery lives across mu and beta; running CSP per sub-band and stacking
    captures multi-band structure that single-band CSP misses. CSP filters are
    learned in :meth:`fit` (train-only when inside a pipeline).
    """

    def __init__(self, sfreq: float, n_components: int = 4, bands=None):
        self.sfreq = sfreq
        self.n_components = n_components
        # Overlapping sub-bands across mu+beta work well for MI.
        self.bands = bands or [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]

    def _bandpass(self, X, fmin, fmax):
        from mne.filter import filter_data

        return filter_data(
            X.astype(np.float64), self.sfreq, fmin, fmax, verbose="ERROR"
        )

    def fit(self, X, y):
        from mne.decoding import CSP

        self.csps_ = []
        for fmin, fmax in self.bands:
            Xb = self._bandpass(X, fmin, fmax)
            csp = CSP(n_components=self.n_components, reg="ledoit_wolf", log=True)
            csp.fit(Xb, y)
            self.csps_.append(csp)
        return self

    def transform(self, X):
        parts = []
        for (fmin, fmax), csp in zip(self.bands, self.csps_):
            Xb = self._bandpass(X, fmin, fmax)
            parts.append(csp.transform(Xb))
        return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------- #
# Riemannian
# --------------------------------------------------------------------------- #
def make_covariances(estimator: str = "lwf"):
    """Return a ``pyriemann.estimation.Covariances`` transformer.

    ``"lwf"`` = Ledoit-Wolf shrinkage; robust when n_times is not >> n_channels.
    Feeds ``TangentSpace`` or ``MDM`` (see :mod:`eeglog.models_classic`).
    """
    from pyriemann.estimation import Covariances

    return Covariances(estimator=estimator)
