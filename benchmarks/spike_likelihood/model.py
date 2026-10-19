import torch
from torch import nn


class SpikeLikelihood(nn.Module):
    """Bounded shared correction of baseline candidate log odds; zero-init identity."""

    def __init__(self, features=19, hidden=8, bound=4.0):
        super().__init__()
        self.spec = dict(features=features, hidden=hidden, bound=bound)
        self.layers = nn.Sequential(nn.Linear(features, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, primary):
        if features.shape[-1] != self.spec["features"]:
            raise ValueError("Only declared EMG/calibration features are allowed")
        prior = (primary.float() * 2 - 1) * 2.1972245773362196
        return prior + self.spec["bound"] * torch.tanh(self.layers(features.clamp(-6, 6))).squeeze(
            -1
        )
