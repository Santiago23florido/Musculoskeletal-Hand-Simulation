"""Five-parameter C3 baseline; inactive until the L3 validation gate passes."""

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ManualController:
    """Two drift gains, memory slope/centre and maximum mixing weight.

    The declared controller input uses speed (1/s); multiply by dt to recover
    the per-batch |delta state| in the requested manual baseline equation.
    Zero drift gains and beta_max=0 are the exact Pareto identity.
    """

    a_wh: float = 0.0
    a_sv: float = 0.0
    b: float = 8.0
    c: float = 0.75
    beta_max: float = 0.0

    def __post_init__(self):
        if min(self.a_wh, self.a_sv, self.b) < 0:
            raise ValueError("Drift gains and memory slope must be nonnegative")
        if not 0 <= self.c <= 1 or not 0 <= self.beta_max <= 0.8:
            raise ValueError("Invalid memory centre or beta_max")

    @property
    def spec(self):
        return asdict(self)

    def __call__(self, features, *, dt=204 / 2048):
        if features.ndim != 2 or features.shape[1] != 9 or dt <= 0:
            raise ValueError("Nine causal inputs and a positive batch interval are required")
        displacement = features[:, 4].abs() * dt
        mass = features[:, 6].clamp(0, 1)
        wh = (1 + self.a_wh * displacement.mean()).clamp(0.25, 4)
        sv = (1 + self.a_sv * displacement).clamp(0.25, 4)
        beta = self.beta_max * torch.sigmoid(self.b * (mass - self.c))
        return wh, sv, beta
