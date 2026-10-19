"""Lag-complete templates and self-cancellation-robust MVDR filters.

The Stage B separator ``w = R^-1 h`` is a Capon/MVDR beamformer. When ``R`` contains the
target unit and ``h`` is mismatched, the filter partially cancels the true unit
(Cox, 1973; Li, Stoica and Wang, 2003): its response falls below the calibrated
threshold while other units stay suppressed, which is the observed high-precision,
low-recall failure. Two remedies are implemented: removing a fraction ``rho`` of the
unit's own modelled contribution from ``R`` and diagonal loading.
"""

import torch


def lag_matrix(template, projection, amplitude, extension=10):
    """Every projected extended vector one discharge produces.

    ``template`` is ``units x 1 x time x 10 x 10`` as in Stage B. Returns
    ``(units, time + extension - 1, D)`` and the row of lag zero, which equals
    ``benchmarks.stage_b.inference.extended_template``.
    """
    units, _, time = template.shape[:3]
    center = time // 2
    flat = template[:, 0].reshape(units, time, -1)
    taus = torch.arange(-center, time - center + extension - 1)
    positions = center + taus[:, None] - torch.arange(extension)[None, :]
    valid = ((positions >= 0) & (positions < time)).to(template.dtype)
    gathered = flat[:, positions.clamp(0, time - 1)] * valid[None, :, :, None]
    extended = gathered.flatten(2) * amplitude[:, None, None]
    return extended @ projection, center


def stage_b_covariance(covariance, h, rate):
    """The original Stage B model term, kept only to reproduce its alpha mixing."""
    dimension = covariance.shape[0]
    eye = torch.eye(dimension, dtype=covariance.dtype)
    model = h.T @ (h * rate[:, None])
    residual = ((covariance.trace() - model.trace()) / dimension).clamp_min(0.01)
    return model + residual * eye


def robust_filters(covariance, lags, lag0, rate, *, rho, loading, alpha=0.0, floor=0.05):
    """MVDR filters ``w_j`` with ``w_j^T h_j = 1``, robust to self-cancellation.

    ``A = (1-alpha) R + alpha M + loading * mean_eig(R) I`` and the effective covariance
    for unit ``j`` is ``A - rho * rate_j * C_j^T C_j`` (its own lag-complete template).
    The inverse uses Woodbury on a shared Cholesky factor. The inner matrix
    ``I - U^T A^-1 U`` is eigen-clipped at ``floor``, so the subtraction can never
    make the effective covariance indefinite, whatever the template error.
    """
    dimension = covariance.shape[0]
    eye = torch.eye(dimension, dtype=covariance.dtype)
    h = lags[:, lag0]
    if alpha > 0:
        covariance = (1 - alpha) * covariance + alpha * stage_b_covariance(covariance, h, rate)
    mean_eig = (covariance.trace() / dimension).clamp_min(1e-12)
    factor = torch.linalg.cholesky(covariance + max(loading, 1e-6) * mean_eig * eye)
    base = torch.cholesky_solve(h.T, factor)
    if rho > 0:
        columns = []
        for unit in range(h.shape[0]):
            u = lags[unit].T * (rho * rate[unit]).clamp_min(0).sqrt()
            a_inv_u = torch.cholesky_solve(u, factor)
            inner = torch.eye(u.shape[1], dtype=u.dtype) - u.T @ a_inv_u
            values, vectors = torch.linalg.eigh(0.5 * (inner + inner.T))
            projected = vectors.T @ (u.T @ base[:, unit])
            columns.append(
                base[:, unit] + a_inv_u @ (vectors @ (projected / values.clamp_min(floor)))
            )
        base = torch.stack(columns, 1)
    response = (h * base.T).sum(1)
    return base / response.clamp_min(1e-12)[None]
