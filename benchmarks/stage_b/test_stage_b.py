"""Biophysical mapping, numerical inverse, causal observer and leakage guards."""

import pytest
import torch

from .data import supervision
from .inference import analytic_filters, extended_template, interpolate
from .model import BlindState, MorphNet, template_objective

torch.set_num_threads(2)


def test_zero_initialization_and_reference_posture_are_exact_identity():
    torch.manual_seed(3)
    model = MorphNet(channels=4)
    template = torch.randn(3, 1, 16, 4, 4)
    q = torch.tensor([0.0, 0.3, 1.0])
    assert torch.equal(model(template, q), template)
    with torch.no_grad():
        model.output.weight.normal_()
    predicted = model(template, q)
    assert torch.equal(predicted[0], template[0])
    assert torch.all((predicted - template).flatten(1).norm(dim=1) <= model.spec["bound"] + 1e-5)


def test_shared_morph_is_unit_equivariant_and_has_finite_learning_signal():
    torch.manual_seed(4)
    model = MorphNet(channels=4)
    template = torch.randn(3, 1, 16, 4, 4) * 0.1
    q = torch.tensor([0.2, 0.5, 0.8])
    loss = template_objective(model(template, q), template * 1.2, q, calibration=template)
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        model.output.weight.normal_(std=0.02)
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(
        model(template[permutation], q[permutation]), model(template, q)[permutation]
    )


def test_analytic_inverse_matches_dense_shrunk_solution_and_is_finite_when_singular():
    torch.manual_seed(5)
    x = torch.randn(30, 9, dtype=torch.float64)
    covariance = x.T @ x / len(x)
    h = torch.randn(3, 9, dtype=torch.float64)
    rate = torch.full((3,), 0.01, dtype=torch.float64)
    model = h.T @ (h * rate[:, None])
    residual = ((covariance.trace() - model.trace()) / 9).clamp_min(0.01)
    matrix = 0.5 * covariance + 0.5 * (model + residual * torch.eye(9))
    matrix += 0.001 * matrix.trace() / 9 * torch.eye(9)
    expected = torch.linalg.solve(matrix, h.T)
    expected /= (h * expected.T).sum(1).sqrt()[None]
    torch.testing.assert_close(analytic_filters(covariance, h, rate), expected)
    assert torch.isfinite(analytic_filters(torch.ones(9, 9), h.float(), rate.float())).all()


def test_extended_template_uses_original_delayed_block_channel_order():
    template = torch.arange(2 * 1 * 64 * 10 * 10).float().reshape(2, 1, 64, 10, 10)
    projection = torch.eye(1000)
    amplitude = torch.tensor([1.0, 2.0])
    actual = extended_template(template, projection, amplitude)
    expected = torch.cat([template[:, 0, 32 - lag].flatten(1) for lag in range(10)], -1)
    torch.testing.assert_close(actual, expected * amplitude[:, None])


def test_blind_observer_is_causal_and_rejects_added_kinematic_features():
    torch.manual_seed(7)
    model = BlindState()
    x = torch.randn(1, 80, 120)
    changed = x.clone()
    changed[:, 40:] *= -10
    a, _ = model(x)
    b, _ = model(changed)
    torch.testing.assert_close(a[:, :40], b[:, :40], rtol=0, atol=0)
    first, hidden = model(x[:, :40])
    last, _ = model(x[:, 40:], hidden)
    torch.testing.assert_close(torch.cat([first, last], 1), a)
    with pytest.raises(ValueError, match="EMG features only"):
        model(torch.randn(1, 10, 122))


@pytest.mark.parametrize(
    "subject,condition,snr,phase",
    [
        ("sub-02", "staircase", 30, "training"),
        ("sub-03", "staircase", 15, "validation"),
        ("sub-01", "triangular-ramp5s", 30, "training"),
    ],
)
def test_supervision_cannot_access_outer_subject_or_reserved_conditions(
    subject, condition, snr, phase
):
    with pytest.raises(ValueError):
        supervision(subject, condition, snr, outer="sub-02", phase=phase)


def test_template_table_interpolates_endpoints_and_linear_functions_exactly():
    grid = torch.linspace(0, 1, 129)
    table = (2 * grid + 3)[:, None, None].expand(-1, 2, 4)
    for q in (0.0, 0.2345, 1.0):
        torch.testing.assert_close(
            interpolate(table, torch.tensor(q)), torch.full((2, 4), 2 * q + 3)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA training environment")
def test_cuda_training_and_cpu_deployment_checkpoint_are_compatible(tmp_path):
    from .experiment import training_device

    device = training_device(13)
    model = MorphNet(channels=4).to(device)
    template = torch.randn(2, 1, 16, 4, 4, device=device) * 0.1
    q = torch.tensor([0.3, 0.8], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    objective = template_objective(model(template, q), template * 1.2, q, template)
    objective.backward()
    optimizer.step()
    observer = BlindState().to(device)
    state, _ = observer(torch.randn(1, 30, 120, device=device))
    state.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in observer.parameters())
    path = tmp_path / "cuda.pt"
    torch.save(model.state_dict(), path)
    deployed = MorphNet(channels=4)
    deployed.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    assert next(deployed.parameters()).device.type == "cpu"
    torch.testing.assert_close(
        model(template, q).cpu(), deployed(template.cpu(), q.cpu()), rtol=1e-4, atol=1e-6
    )
