"""A shared, unit-equivariant correction to calibrated linear filters."""

import torch
from torch import nn


class ResidualGate(nn.Module):
    """y = y_pareto + a * (y_calibration - y_pareto), |a| <= 0.15.

    On augmented sources (y_pareto,y_calibration), the residual operator has
    spectral norm at most sqrt(2)*0.15 < 1. Thus the augmented transform is
    invertible, with minimum singular value >= 1-sqrt(2)*0.15. This does not
    imply a RoA guarantee. Weights never index subject, electrode or unit.
    """

    def __init__(self, features=10, hidden=8, bound=0.15):
        super().__init__()
        if not 0 < bound < 1 / 2**0.5:
            raise ValueError("Invalid residual bound")
        self.spec = dict(features=features, hidden=hidden, bound=bound)
        self.layer = nn.Linear(features, hidden)
        self.output = nn.Linear(hidden, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        return self.spec["bound"] * torch.tanh(
            self.output(torch.tanh(self.layer(features.clamp(-6, 6))))
        ).squeeze(-1)


def correct(base, calibration, coefficient):
    return base + coefficient * (calibration - base)


def waveform_loss(sources, trusted, template, *, radius=10):
    """Calibration source-response shape consistency; not multichannel MUAP truth."""
    windows = sources.unfold(0, 2 * radius + 1, 1)
    selected = trusted[radius:-radius]
    energy = windows.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
    normalized = windows / energy
    # ICA sources have arbitrary sign; a sign flip is not an identity error.
    errors = torch.minimum(
        (normalized - template[None]).square().sum(-1),
        (normalized + template[None]).square().sum(-1),
    )
    counts = selected.sum(0)
    per_unit = (errors * selected).sum(0) / counts.clamp_min(1)
    # An absent spike is handled by the reference contrast penalty, not a made-up label.
    return per_unit[counts > 0].mean() if torch.any(counts > 0) else sources.sum() * 0
