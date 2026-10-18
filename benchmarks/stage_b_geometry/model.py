"""Shared calibrated warp with no additive unit-independent waveform."""

import torch
from torch import nn


class GeometricMorph(nn.Module):
    def __init__(self, bound=2.0):
        super().__init__()
        self.spec = dict(bound=bound)
        self.coefficients = nn.Parameter(torch.zeros(30, 8))

    def features(self, template):
        energy = template[:, 0].square()
        mass = energy.sum((1, 2, 3)).clamp_min(1e-12)
        t = torch.linspace(-1, 1, template.shape[2], device=template.device)[None, :, None, None]
        y = torch.linspace(-1, 1, template.shape[3], device=template.device)[None, None, :, None]
        x = torch.linspace(-1, 1, template.shape[4], device=template.device)[None, None, None, :]

        def mean(value):
            return (energy * value).sum((1, 2, 3)) / mass

        mt, my, mx = mean(t), mean(y), mean(x)
        vt, vy, vx = mean(t * t) - mt.square(), mean(y * y) - my.square(), mean(x * x) - mx.square()
        return torch.stack(
            [
                torch.ones_like(mt),
                mt,
                my,
                mx,
                vt.clamp_min(0).sqrt(),
                vy.clamp_min(0).sqrt(),
                vx.clamp_min(0).sqrt(),
                mean(t * y) - mt * my,
                mean(t * x) - mt * mx,
                mean(x * y) - mx * my,
            ],
            1,
        )

    def forward(self, template, q):
        m, _, nt, ny, nx = template.shape
        q = q.clamp(0, 1)
        basis = torch.stack([q, q.square(), q.pow(3)], 1)
        conditioning = (basis[:, :, None] * self.features(template)[:, None, :]).flatten(1)
        limits = template.new_tensor([16.0, 0.4, 2.0, 2.0, 0.4, 0.4, 0.4, 0.7])
        parameters = torch.tanh(conditioning @ self.coefficients) * limits
        delay, logtime, dx, dy, logx, logy, angle, loggain = parameters.unbind(1)
        t = torch.arange(nt, dtype=template.dtype, device=template.device)
        y, x = torch.meshgrid(
            torch.arange(ny, dtype=template.dtype, device=template.device),
            torch.arange(nx, dtype=template.dtype, device=template.device),
            indexing="ij",
        )
        ts = (t[None] - nt // 2 - delay[:, None]) / logtime.exp()[:, None] + nt // 2
        xx = (x[None] - (nx - 1) / 2 - dx[:, None, None]) / logx.exp()[:, None, None]
        yy = (y[None] - (ny - 1) / 2 - dy[:, None, None]) / logy.exp()[:, None, None]
        cosine, sine = angle.cos()[:, None, None], angle.sin()[:, None, None]
        xs = cosine * xx + sine * yy + (nx - 1) / 2
        ys = -sine * xx + cosine * yy + (ny - 1) / 2
        ti, yi, xi = ts.floor().long(), ys.floor().long(), xs.floor().long()
        ft, fy, fx = ts - ti, ys - yi, xs - xi
        flat = template[:, 0].flatten(1)
        values = []
        for dt in (0, 1):
            for iy in (0, 1):
                for ix in (0, 1):
                    a, b, c = ti + dt, yi + iy, xi + ix
                    valid = ((a >= 0) & (a < nt))[:, :, None, None] & (
                        (b >= 0) & (b < ny) & (c >= 0) & (c < nx)
                    )[:, None]
                    index = (
                        a.clamp(0, nt - 1)[:, :, None, None] * ny * nx
                        + b.clamp(0, ny - 1)[:, None] * nx
                        + c.clamp(0, nx - 1)[:, None]
                    )
                    sample = flat.gather(1, index.reshape(m, -1)).reshape(m, nt, ny, nx)
                    wt, wy, wx = (
                        (ft if dt else 1 - ft),
                        (fy if iy else 1 - fy),
                        (fx if ix else 1 - fx),
                    )
                    values.append(sample * wt[:, :, None, None] * wy[:, None] * wx[:, None] * valid)
        warped = torch.stack(values).sum(0)[:, None] * loggain.exp()[:, None, None, None, None]
        delta = warped - template
        radius = self.spec["bound"] * template.flatten(1).norm(dim=1)
        scale = (radius / delta.flatten(1).norm(dim=1).clamp_min(1e-12)).clamp_max(1)
        result = template + delta * scale[:, None, None, None, None]
        return torch.where((q == 0)[:, None, None, None, None], template, result)
