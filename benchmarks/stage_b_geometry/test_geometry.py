import pytest
import torch

from .model import GeometricMorph

torch.set_num_threads(2)


def test_identity_reference_and_zero_input_are_preserved():
    torch.manual_seed(31)
    model = GeometricMorph()
    cal = torch.randn(3, 1, 64, 10, 10) * 0.01
    q = torch.tensor([0.0, 0.4, 1.0])
    assert torch.equal(model(cal, q), cal)
    with torch.no_grad():
        model.coefficients.normal_(std=0.02)
    assert torch.equal(model(cal, q)[0], cal[0])
    assert torch.equal(model(torch.zeros_like(cal), q), torch.zeros_like(cal))
    assert torch.all(
        (model(cal, q) - cal).flatten(1).norm(dim=1) <= 2 * cal.flatten(1).norm(dim=1) + 1e-6
    )


def test_warp_is_amplitude_homogeneous_and_unit_permutation_equivariant():
    torch.manual_seed(32)
    model = GeometricMorph()
    with torch.no_grad():
        model.coefficients.normal_(std=0.03)
    cal = torch.randn(3, 1, 64, 10, 10) * 0.01
    q = torch.tensor([0.2, 0.4, 0.8])
    torch.testing.assert_close(model(2 * cal, q), 2 * model(cal, q))
    order = torch.tensor([2, 0, 1])
    torch.testing.assert_close(model(cal[order], q[order]), model(cal, q)[order])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_geometric_risk_backpropagates_deterministically_in_cuda():
    from benchmarks.stage_b.experiment import training_device

    from .decoder import objective, teacher

    device = training_device(33)
    model = GeometricMorph().to(device)
    cal = torch.randn(2, 1, 64, 10, 10, device=device) * 0.01
    target = cal.roll(1, 2)
    case = dict(
        cal=cal,
        target=target[None],
        q=torch.tensor([0.5], device=device),
        projection=torch.randn(1000, 12, device=device) * 0.1,
        amplitude=torch.ones(2, device=device),
        covariance=torch.eye(12, device=device)[None],
        rate=torch.full((2,), 0.01, device=device),
    )
    loss, _ = objective(model, case, 0, teacher(case, 0))
    loss.backward()
    assert torch.isfinite(model.coefficients.grad).all()
    assert model.coefficients.grad.abs().sum() > 0
