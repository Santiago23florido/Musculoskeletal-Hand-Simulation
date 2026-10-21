"""C1 uncertainty, C3 bounded control and C4 residual event likelihood."""

import math

import torch
from torch import nn


class BlindState(nn.Module):
    """Stage B GRU (120 -> 24), with a Gaussian mean/log-variance head."""

    def __init__(self, features=120, hidden=24):
        super().__init__()
        self.spec = dict(features=features, hidden=hidden)
        self.gru = nn.GRU(features, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 2)

    def forward(self, features, hidden=None):
        if features.shape[-1] != 120:
            raise ValueError("Exactly 120 EMG features are required; angular input is forbidden")
        x, hidden = self.gru(features.clamp(-6, 6), hidden)
        output = self.output(x)
        return output[..., 0].sigmoid(), output[..., 1].clamp(-8, 2), hidden


def gaussian_nll(mean, log_variance, target):
    return 0.5 * (log_variance + (target - mean).square() * (-log_variance).exp()).mean()


class Controller(nn.Module):
    """159 shared parameters; exact identity at initialization (black-box training)."""

    def __init__(self, beta_max=0.5):
        super().__init__()
        self.beta_max = float(beta_max)
        self.layers = nn.Sequential(nn.Linear(9, 12), nn.Tanh(), nn.Linear(12, 3))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
        assert sum(p.numel() for p in self.parameters()) <= 300

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 9:
            raise ValueError("Nine declared EMG/state/memory features per unit are required")
        raw = self.layers(features.clamp(-6, 6)).tanh()
        wh = (math.log(4) * raw[:, 0].mean()).exp()
        sv = (math.log(4) * raw[:, 1]).exp()
        beta = self.beta_max * raw[:, 2].square()
        return wh, sv, beta


class SpikeLikelihood(nn.Module):
    """Original 19->8->1 likelihood plus source, memory confidence, blind speed."""

    def __init__(self, features=22, hidden=8, bound=4.0):
        super().__init__()
        self.spec = dict(features=features, hidden=hidden, bound=bound)
        self.layers = nn.Sequential(nn.Linear(features, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, primary):
        if features.shape[-1] != 22:
            raise ValueError("Only the 22 declared features are allowed")
        prior = (primary.float() * 2 - 1) * math.log(9)
        return prior + self.spec["bound"] * self.layers(features.clamp(-6, 6)).squeeze(-1).tanh()

    def is_identity(self):
        return not bool(torch.count_nonzero(self.layers[-1].weight)) and not bool(
            torch.count_nonzero(self.layers[-1].bias)
        )
