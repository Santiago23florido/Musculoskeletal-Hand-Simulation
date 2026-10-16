"""Causal state encoder and the Stage A transform from report section 5.4."""

import math

import torch
from torch import nn


class MotionCorrection(nn.Module):
    """A(z)=I+U(z)V(z)^T; six bases, rank three, one shared model across records.

    The Frobenius-product bound is a certified upper bound on the spectral norm.
    Hence ||UV^T||_2 <= bound < 1 and sigma_min(A) >= 1-bound.
    """

    def __init__(
        self,
        dimension=1000,
        features=122,
        state_dim=1,
        hidden=16,
        bases=6,
        rank=3,
        bound=0.8,
        encoder="gru",
    ):
        super().__init__()
        if not 0 < bound < 1 or state_dim not in (1, 2, 3):
            raise ValueError("Invalid invertibility bound or state dimension")
        self.spec = dict(
            dimension=dimension,
            features=features,
            state_dim=state_dim,
            hidden=hidden,
            bases=bases,
            rank=rank,
            bound=bound,
            encoder=encoder,
        )
        self.encoder = nn.GRU(features, hidden, batch_first=True) if encoder == "gru" else None
        self.state = nn.Linear(hidden if self.encoder is not None else features, state_dim)
        self.gate = nn.Sequential(nn.Linear(state_dim, 16), nn.Tanh(), nn.Linear(16, bases))
        self.u_basis = nn.Parameter(torch.zeros(bases, dimension, rank))
        self.v_basis = nn.Parameter(torch.randn(bases, dimension, rank) / math.sqrt(dimension))
        self.register_buffer("feature_mean", torch.zeros(features))
        self.register_buffer("feature_scale", torch.ones(features))

    def forward(self, features, hidden=None):
        x = ((features - self.feature_mean) / self.feature_scale).clamp(-12, 12)
        if self.encoder is not None:
            x, hidden = self.encoder(x, hidden)
        state = torch.tanh(self.state(x))
        weights = self.gate(state).softmax(-1)
        u = torch.einsum("btk,kdr->btdr", weights, self.u_basis)
        v = torch.einsum("btk,kdr->btdr", weights, self.v_basis)
        # Known reference posture: the initial five-second calibration remains fixed.
        u = u * features[..., -2].abs().clamp(0, 1)[..., None, None]
        norm_bound = u.norm(dim=(-2, -1)) * v.norm(dim=(-2, -1))
        scale = (self.spec["bound"] / norm_bound.clamp_min(1e-8)).clamp(max=1.0)
        u = u * scale[..., None, None]
        return u, v, state, hidden


def apply_correction(x, u, v):
    return x + (x @ v) @ u.transpose(-1, -2)


def kl_change(x, u, v, shrinkage=0.001):
    """Exact KL(A C A^T||I)-KL(C||I), C=(1-rho)cov(x)+rho I.

    No D-by-D covariance or parameter-independent logdet(C) is needed. The
    ridge is applied BEFORE A; upstream applies it AFTER A. This small, explicit
    estimator difference is recorded rather than claimed numerically identical.
    Supports leading batch dimensions. Covariance uses ddof=0, as the authors.
    """
    x = x - x.mean(-2, keepdim=True)
    xv, xu = x @ v, x @ u
    n = x.shape[-2]
    cvv = (1 - shrinkage) * (xv.transpose(-1, -2) @ xv) / n
    cvv = cvv + shrinkage * (v.transpose(-1, -2) @ v)
    cross = (1 - shrinkage) * (xv * xu).sum((-2, -1)) / n
    cross = cross + shrinkage * (u * v).sum((-2, -1))
    utu = u.transpose(-1, -2) @ u
    trace_delta = 2 * cross + (cvv * utu.transpose(-1, -2)).sum((-2, -1))
    identity = torch.eye(u.shape[-1], dtype=x.dtype, device=x.device)
    sign, logdet = torch.linalg.slogdet(identity + v.transpose(-1, -2) @ u)
    if torch.any(sign <= 0):
        raise FloatingPointError("Non-invertible low-rank correction")
    return 0.5 * trace_delta - logdet


def contrast_loss(sources, mask, target, scale):
    """Authors' spike-based squared, z-scored log-cosh consistency loss.

    Detection/gating is detached; absent trusted spikes receive their fixed
    squared penalty 9 and cannot fabricate gradients through labels.
    """
    log_cosh = sources + torch.nn.functional.softplus(-2 * sources) - math.log(2)
    count = mask.sum(-2)
    value = (log_cosh * mask).sum(-2) / count.clamp_min(1)
    error = (value - target) / scale.clamp_min(1e-7)
    return torch.where(count > 0, error.square(), torch.full_like(error, 9)).mean(-1)
