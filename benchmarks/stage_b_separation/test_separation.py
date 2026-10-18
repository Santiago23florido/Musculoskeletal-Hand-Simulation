import pytest
import torch

from .decoder import covariance_model, mvdr, objective, teacher
from .states import guarded


def test_mvdr_is_distortionless_and_solves_constrained_minimum_variance():
    torch.manual_seed(8)
    x = torch.randn(40, 9, dtype=torch.float64)
    covariance = x.T @ x / len(x)
    h = torch.randn(3, 9, dtype=torch.float64)
    rate = torch.full((3,), 0.02, dtype=torch.float64)
    w = mvdr(covariance, h, rate)
    torch.testing.assert_close((h * w.T).sum(1), torch.ones(3, dtype=h.dtype))
    matrix = covariance_model(covariance, h, rate)
    for index in range(3):
        perturbation = torch.randn(9, dtype=h.dtype)
        perturbation -= h[index] * (perturbation @ h[index]) / h[index].square().sum()
        alternative = w[:, index] + perturbation
        assert alternative @ matrix @ alternative >= w[:, index] @ matrix @ w[:, index]


def test_covariance_source_loss_equals_explicit_source_error():
    torch.manual_seed(11)
    x = torch.randn(80, 9, dtype=torch.float64)
    c = x.T @ x / len(x)
    a, b = torch.randn(9, 3, dtype=x.dtype), torch.randn(9, 3, dtype=x.dtype)
    d = a - b
    implicit = (d * (c @ d)).sum(0)
    explicit = (x @ a - x @ b).square().mean(0)
    torch.testing.assert_close(implicit, explicit)


def test_exact_teacher_template_has_zero_projected_and_source_risk():
    torch.manual_seed(15)
    cal = torch.randn(2, 1, 64, 10, 10) * 0.01
    target = cal + 0.001
    case = dict(
        cal=cal,
        target=target[None],
        q=torch.tensor([0.5]),
        projection=torch.randn(1000, 12) * 0.1,
        amplitude=torch.ones(2),
        covariance=torch.eye(12)[None],
        rate=torch.full((2,), 0.01),
    )
    fixed = teacher(case, 0)
    _, metrics = objective(lambda observed, q: target, case, 0, fixed)
    assert abs(float(metrics["source"])) < 1e-6
    assert abs(float(metrics["projected"])) < 1e-6
    assert abs(float(metrics["angular"])) < 1e-6


def test_decoder_objective_has_finite_gradient_through_covariance_and_normalization():
    from benchmarks.stage_b.model import MorphNet

    torch.manual_seed(12)
    model = MorphNet(channels=4)
    cal = torch.randn(2, 1, 64, 10, 10) * 0.01
    projection = torch.randn(1000, 12) * 0.1
    target = cal.clone()
    target[:, :, 30:34] += 0.001
    case = dict(
        cal=cal,
        target=target[None],
        q=torch.tensor([0.5]),
        projection=projection,
        amplitude=torch.ones(2),
        covariance=torch.eye(12)[None],
        rate=torch.full((2,), 0.01),
    )
    fixed = teacher(case, 0)
    loss, _ = objective(model, case, 0, fixed)
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize(
    "subject,condition,snr,phase",
    [
        ("sub-02", "staircase", 30, "training"),
        ("sub-03", "staircase", 15, "validation"),
        ("sub-01", "triangular-ramp5s", 30, "training"),
    ],
)
def test_separation_fitting_rejects_outer_labels_and_reserved_conditions(
    subject, condition, snr, phase
):
    with pytest.raises(ValueError):
        guarded(subject, condition, snr, "sub-02", phase)
