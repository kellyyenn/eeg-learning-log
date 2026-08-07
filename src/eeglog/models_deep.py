"""Deep models for EEG, all sized to train on a laptop CPU in minutes.

We use ``braindecode`` (PyTorch + skorch) for the published architectures and add
two compact custom nets (LSTM, Transformer) so the whole zoo speaks the same
``skorch`` interface and drops into the same sklearn-style pipelines/CV.

Models
------
* EEGNet         : ``EEGNetv4`` — compact depthwise-separable CNN, fewest params.
* ShallowConvNet : ``ShallowFBCSPNet`` — band-power-style shallow CNN.
* DeepConvNet    : ``Deep4Net`` — deeper hierarchical CNN.
* TinyLSTM       : 1-layer LSTM over time, channels as features.
* TinyTransformer: 2-layer Transformer encoder over time tokens.

All expose ``build_*`` returning a skorch ``EEGClassifier`` whose ``fit``/
``predict`` accept ``X`` of shape ``(n_epochs, n_channels, n_times)``.

CPU speed comes from: small inputs (crop time), few epochs, modest batch size,
and tiny hidden sizes. Per-channel standardization is fit on train only — pass
``X`` already cropped/scaled, or wrap in a pipeline with a scaler step.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _classifier(
    module,
    *,
    max_epochs=250,
    batch_size=32,
    lr=1e-3,
    weight_decay=0.0,
    patience=40,
    random_state=42,
    **kwargs,
):
    """Wrap a torch module in a skorch ``EEGClassifier`` with CPU-friendly defaults.

    Defaults matter a lot here. IV-2a gives you only ~288 trials per session, and
    the published EEGNet / ConvNet results train for *hundreds* of epochs. An
    earlier version of this file used ``max_epochs=40, patience=8``, which stops
    while the nets are still near chance (EEGNet scored kappa 0.05 instead of
    ~0.5) -- the models looked broken when they were merely undertrained. Keep
    ``patience`` large relative to the noise in a ~58-trial validation split.
    """
    from braindecode import EEGClassifier
    from skorch.callbacks import EarlyStopping
    from skorch.dataset import ValidSplit

    torch.manual_seed(random_state)
    return EEGClassifier(
        module,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_split=ValidSplit(0.2, random_state=random_state),
        callbacks=[("early", EarlyStopping(patience=patience, load_best=True))],
        device="cpu",
        verbose=0,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Braindecode published architectures
# --------------------------------------------------------------------------- #
def build_eegnet(n_chans, n_times, n_classes, **kw):
    from braindecode.models import EEGNetv4

    net = EEGNetv4(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
    return _classifier(net, **kw)


def build_shallow(n_chans, n_times, n_classes, **kw):
    from braindecode.models import ShallowFBCSPNet

    net = ShallowFBCSPNet(
        n_chans=n_chans, n_outputs=n_classes, n_times=n_times, final_conv_length="auto"
    )
    return _classifier(net, **kw)


def build_deep(n_chans, n_times, n_classes, **kw):
    from braindecode.models import Deep4Net

    net = Deep4Net(
        n_chans=n_chans, n_outputs=n_classes, n_times=n_times, final_conv_length="auto"
    )
    return _classifier(net, **kw)


# --------------------------------------------------------------------------- #
# Tiny custom recurrent / attention nets
# --------------------------------------------------------------------------- #
class TinyLSTM(nn.Module):
    """1-layer LSTM over the time axis; channels are the per-step feature vector."""

    def __init__(self, n_chans, n_times, n_classes, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_chans, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):  # x: (batch, n_chans, n_times)
        x = x.permute(0, 2, 1)  # -> (batch, n_times, n_chans)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])  # last time step


class TinyTransformer(nn.Module):
    """Compact Transformer encoder over time tokens (mean-pooled, then linear)."""

    def __init__(self, n_chans, n_times, n_classes, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(n_chans, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):  # x: (batch, n_chans, n_times)
        x = x.permute(0, 2, 1)  # -> (batch, n_times, n_chans)
        x = self.proj(x)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))  # global average pool over time


def build_lstm(n_chans, n_times, n_classes, hidden=32, **kw):
    return _classifier(TinyLSTM(n_chans, n_times, n_classes, hidden=hidden), **kw)


def build_transformer(n_chans, n_times, n_classes, **kw):
    return _classifier(TinyTransformer(n_chans, n_times, n_classes), **kw)


def all_deep(n_chans, n_times, n_classes, **kw) -> dict:
    """Return every deep model keyed by name (uninstantiated builders applied)."""
    return {
        "EEGNet": build_eegnet(n_chans, n_times, n_classes, **kw),
        "ShallowConvNet": build_shallow(n_chans, n_times, n_classes, **kw),
        "DeepConvNet": build_deep(n_chans, n_times, n_classes, **kw),
        "TinyLSTM": build_lstm(n_chans, n_times, n_classes, **kw),
        "TinyTransformer": build_transformer(n_chans, n_times, n_classes, **kw),
    }


def as_float32(X) -> np.ndarray:
    """braindecode/skorch want float32 inputs of shape (n, n_chans, n_times)."""
    return np.asarray(X, dtype=np.float32)


def decimate_time(X, factor: int = 4) -> np.ndarray:
    """Downsample the time axis by ``factor`` (simple stride).

    The CNNs (EEGNet/Shallow/Deep) handle long inputs fine, but the recurrent
    and attention nets are O(n_times) / O(n_times^2) on CPU — feeding them a
    1000-sample window is slow. Decimating to ~250 samples keeps them in the
    "minutes on CPU" budget with negligible accuracy cost for MI. Use a proper
    anti-aliased ``scipy.signal.decimate`` if you care about the high band.
    """
    return as_float32(X)[:, :, ::factor]
