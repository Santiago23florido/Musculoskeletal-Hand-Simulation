"""Nontrivial safety contracts for the untrained residual components."""

import copy

import pytest
import torch

from .memory import PosturalMemory, effective_filters, limited_blend, project_filters
from .models import BlindState, Controller, SpikeLikelihood, gaussian_nll


def test_zero_control_and_likelihood_preserve_pareto_decisions_exactly():
    controller = Controller()
    wh, sv, beta = controller(torch.randn(7, 9))
    assert wh == 1 and torch.all(sv == 1) and torch.all(beta == 0)
    model = SpikeLikelihood()
    primary = torch.rand(19) > 0.5
    assert torch.equal(model(torch.randn(19, 22), primary) > 0, primary)
    assert sum(p.numel() for p in controller.parameters()) <= 300


def test_control_bounds_hold_under_extreme_trained_weights():
    controller = Controller(beta_max=0.7)
    with torch.no_grad():
        for p in controller.parameters():
            p.normal_(0, 100)
    wh, sv, beta = controller(torch.randn(100, 9) * 50)
    assert 0.25 <= wh <= 4 and torch.all((sv >= 0.25) & (sv <= 4))
    assert torch.all((beta >= 0) & (beta <= 0.7))


def test_uncertain_observer_prefix_is_causal_and_nll_has_gradients():
    model = BlindState()
    features = torch.randn(1, 13, 120)
    changed = features.clone()
    changed[:, 7:] += 100
    mean, logvar, _ = model(features)
    other_mean, other_logvar, _ = model(changed)
    torch.testing.assert_close(mean[:, :7], other_mean[:, :7], rtol=0, atol=0)
    torch.testing.assert_close(logvar[:, :7], other_logvar[:, :7], rtol=0, atol=0)
    gaussian_nll(mean, logvar, torch.rand_like(mean)).backward()
    assert torch.isfinite(model.output.weight.grad).all()
    with pytest.raises(ValueError):
        model(torch.randn(1, 13, 121))


def test_effective_filters_round_trip_through_a_changed_whitening_frame():
    w = torch.randn(13, 13, dtype=torch.float64)
    w = w @ w.T + torch.eye(13)
    separators = torch.randn(4, 13, dtype=torch.float64)
    effective = effective_filters(w, separators)
    changed = w + 0.1 * torch.eye(13)
    proposed = project_filters(changed, effective)
    torch.testing.assert_close(proposed @ changed, effective, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kappa", [0.02, 0.1, 0.3])
def test_limiter_bounds_every_row_after_qr(kappa):
    p = torch.linalg.qr(torch.randn(20, 5))[0].T
    result = limited_blend(p, torch.randn_like(p) * 100, torch.ones(5), kappa)
    assert torch.all((result - p).norm(dim=1) / p.norm(dim=1) <= kappa)
    torch.testing.assert_close(result @ result.T, torch.eye(5), atol=1e-6, rtol=1e-6)
    assert torch.equal(limited_blend(p, torch.randn_like(p), torch.zeros(5), kappa), p)


def test_memory_uses_only_past_inserts_and_preserves_calibration():
    p, centroid = torch.randn(3, 10), torch.randn(2, 3)
    memory = PosturalMemory(p, centroid)
    same_past = copy.deepcopy(memory)
    before = memory.retrieve(0.5, batch=0)
    memory.insert(0.5, p * 2, centroid * 3, torch.ones(3), batch=0)
    for a, b in zip(before, same_past.retrieve(0.5, batch=0), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="future"):
        memory.retrieve(0.5, batch=0)
    after = memory.retrieve(0.5, batch=1)
    assert torch.norm(after[0] - before[0]) > 0
    memory.insert(0, p * 100, centroid * 100, torch.ones(3), batch=1)
    assert torch.equal(memory.filters[0], p) and torch.equal(memory.centroids[0], centroid)


def test_low_confidence_events_cannot_pollute_memory():
    memory = PosturalMemory(torch.randn(2, 9), torch.randn(2, 2), threshold=0.8)
    old = memory.filters.clone()
    memory.insert(0.5, torch.randn(2, 9), torch.randn(2, 2), torch.full((2,), 0.79), batch=0)
    assert torch.equal(memory.filters, old)
