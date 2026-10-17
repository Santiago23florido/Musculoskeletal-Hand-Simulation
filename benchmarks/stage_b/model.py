"""Shared FiLM MUAP morphing and a strictly EMG-only causal state observer."""

import torch
from torch import nn
from torch.nn import functional as F


class FilmBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1))
        self.temporal = nn.Conv3d(channels, channels, (5, 1, 1), padding=(2, 0, 0))
        self.condition = nn.Sequential(
            nn.Linear(channels + 1, 16), nn.SiLU(), nn.Linear(16, 2 * channels)
        )

    def forward(self, x, q, embedding):
        affine = self.condition(torch.cat([q[:, None], embedding], -1))
        gain, bias = affine.chunk(2, -1)
        y = F.silu(self.spatial(x))
        y = self.temporal(y)
        y = y * (1 + 0.2 * torch.tanh(gain[:, :, None, None, None]))
        y = y + 0.2 * torch.tanh(bias[:, :, None, None, None])
        return x + F.silu(y)


class MorphNet(nn.Module):
    """h(q)=h_cal+q*Gamma(h_cal,q); identical learned weights for every unit.

    A fixed L2 bound limits the residual relative to the unit's calibration
    energy. This is a size prior, not an identifiability or generalization proof.
    No subject, unit or electrode identifiers are trainable inputs.
    """

    def __init__(self, channels=8, bound=2.0):
        super().__init__()
        self.spec = dict(channels=channels, bound=bound)
        self.input = nn.Conv3d(1, channels, (5, 3, 3), padding=(2, 1, 1))
        self.embedding = nn.Sequential(nn.Linear(channels, channels), nn.Tanh())
        self.blocks = nn.ModuleList([FilmBlock(channels), FilmBlock(channels)])
        self.output = nn.Conv3d(channels, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, template, q):
        # template: units x 1 x time x 10 x 10, normalized by own calibration energy.
        x = F.silu(self.input(template))
        embedding = self.embedding(x.mean((2, 3, 4)))
        for block in self.blocks:
            x = block(x, q, embedding)
        delta = self.output(x) * q[:, None, None, None, None]
        norm = delta.flatten(1).norm(dim=1).clamp_min(1e-8)
        scale = (self.spec["bound"] / norm).clamp_max(1)
        return template + delta * scale[:, None, None, None, None]


class BlindState(nn.Module):
    """Angle is a training target only; forward accepts exactly 120 EMG features."""

    def __init__(self, features=120, hidden=24):
        super().__init__()
        self.spec = dict(features=features, hidden=hidden)
        self.gru = nn.GRU(features, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, features, hidden=None):
        if features.shape[-1] != self.spec["features"]:
            raise ValueError("Blind state accepts EMG features only")
        output, hidden = self.gru(features.clamp(-6, 6), hidden)
        return torch.sigmoid(self.output(output)).squeeze(-1), hidden


def template_objective(prediction, target, q, calibration=None):
    error = (prediction - target).flatten(1).square().sum(1)
    energy = target.flatten(1).square().sum(1).clamp_min(1e-7)
    # Temporal slope regularization supplements the supervised waveform fit.
    slope = (
        ((prediction[:, :, 1:] - prediction[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1]))
        .flatten(1)
        .square()
        .sum(1)
    )
    size = (
        (prediction - calibration).flatten(1).square().sum(1).mean()
        if calibration is not None
        else prediction.sum() * 0
    )
    return (error / energy + 0.1 * slope / energy).mean() + 0.001 * size
