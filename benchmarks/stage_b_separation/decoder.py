"""Differentiable MVDR and a loss in the geometry of the actual separator."""

import torch

from benchmarks.stage_b.inference import extended_template


def covariance_model(covariance, h, rate, alpha=0.5, ridge=0.001):
    dimension = covariance.shape[-1]
    eye = torch.eye(dimension, dtype=h.dtype, device=h.device)
    model = h.T @ (h * rate[:, None])
    residual = ((covariance.trace() - model.trace()) / dimension).clamp_min(0.01)
    matrix = (1 - alpha) * covariance + alpha * (model + residual * eye)
    return matrix + ridge * matrix.trace().clamp_min(1e-6) / dimension * eye


def mvdr(covariance, h, rate):
    matrix = covariance_model(covariance, h, rate)
    factor = torch.linalg.cholesky(matrix)
    inverse_h = torch.cholesky_solve(h.T, factor)
    response = (h * inverse_h.T).sum(1).clamp_min(1e-8)
    return inverse_h / response[None]


def teacher(case, state):
    target = case["target"][state]
    h = extended_template(target, case["projection"], case["amplitude"])
    covariance = case["covariance"][state]
    weights = mvdr(covariance, h, case["rate"])
    factor = torch.linalg.cholesky(covariance_model(covariance, h, case["rate"]))
    whitened = torch.linalg.solve_triangular(factor, h.T, upper=False)
    energy = (weights * (covariance @ weights)).sum(0).clamp_min(1e-8)
    return dict(weights=weights, factor=factor, whitened=whitened, energy=energy)


def objective(model, case, state, fixed_teacher, *, noise=0.0):
    cal = case["cal"]
    observed = cal + noise * torch.randn_like(cal) / cal[0].numel() ** 0.5 if noise else cal
    q = case["q"][state].expand(len(cal))
    prediction = model(observed, q)
    h = extended_template(prediction, case["projection"], case["amplitude"])
    covariance = case["covariance"][state]
    weights = mvdr(covariance, h, case["rate"])
    delta = weights - fixed_teacher["weights"]
    # E[(x^T(w-w_teacher))^2] for centered x with covariance C.
    source_error = ((delta * (covariance @ delta)).sum(0) / fixed_teacher["energy"]).mean()
    whitened = torch.linalg.solve_triangular(fixed_teacher["factor"], h.T, upper=False)
    projected_error = (
        (whitened - fixed_teacher["whitened"]).square().sum(0)
        / fixed_teacher["whitened"].square().sum(0).clamp_min(1e-8)
    ).mean()
    angular_error = (
        1 - torch.nn.functional.cosine_similarity(whitened, fixed_teacher["whitened"], dim=0).mean()
    )
    target = case["target"][state]
    waveform_error = (
        (prediction - target).flatten(1).square().sum(1)
        / target.flatten(1).square().sum(1).clamp_min(1e-8)
    ).mean()
    size = (prediction - cal).flatten(1).square().sum(1).mean()
    # The coercive projected term prevents a near-zero-output local solution of
    # source distillation and controls the amplitude as well as the direction.
    loss = (
        projected_error
        + 0.25 * source_error
        + 0.1 * angular_error
        + 0.01 * waveform_error
        + 0.001 * size
    )
    return loss, dict(
        source=source_error,
        projected=projected_error,
        angular=angular_error,
        waveform=waveform_error,
    )
