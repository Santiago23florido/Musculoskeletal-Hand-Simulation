"""Numerical inverse, physical constraints, identity and dormant recovery."""

import torch

from .physics import (
    TemplateState,
    merge_recovery,
    reference_envelope,
    spatial_whitener,
    synthesize,
)

torch.set_num_threads(2)


def calibration_example():
    generator = torch.Generator().manual_seed(42)
    z = torch.randn(80, 8, generator=generator) * 0.02
    spikes = torch.zeros(80, 2, dtype=torch.bool)
    spikes[[10, 30, 50, 70], 0] = True
    spikes[[20, 40, 60], 1] = True
    z[spikes[:, 0], 0] += 2
    z[spikes[:, 1], 1] += 2
    return z, spikes


def test_small_inverse_matches_dense_regularized_covariance_inverse():
    generator = torch.Generator().manual_seed(4)
    h = torch.randn(15, 3, generator=generator, dtype=torch.float64)
    normalized = h / h.norm(dim=0)
    ridge = 0.1
    # Woodbury-equivalent synthesis in the full observation space.
    dense = torch.linalg.solve(normalized @ normalized.T + ridge * torch.eye(15), normalized)
    torch.testing.assert_close(synthesize(h, ridge), dense)


def test_collinear_templates_and_constant_channels_remain_finite():
    h = torch.ones(12, 3, dtype=torch.float64)
    assert torch.isfinite(synthesize(h)).all()
    x = torch.zeros(40, 5)
    x[:, 0] = torch.linspace(-1, 1, 40)
    assert torch.isfinite(spatial_whitener(x)).all()


def test_morph_is_anchored_bounded_and_has_new_spatial_directions():
    z, spikes = calibration_example()
    state = TemplateState(z, spikes, prior=1)
    state.threshold.fill_(0.05)
    moved = z.clone()
    moved[spikes[:, 0], 3] += 0.7
    moved[spikes[:, 1], 4] += 0.8
    before = state.anchor.clone()
    state.observe(moved, spikes, q=0.6)
    assert torch.equal(state.templates(0), before)
    change = state.templates(100) - before
    assert torch.all(change.norm(dim=0) <= 0.5 * before.norm(dim=0) + 1e-6)
    assert (state.templates(0.6)[[3, 4]] - before[[3, 4]]).abs().sum() > 0.1


def test_template_filters_and_evidence_follow_unit_permutation():
    z, spikes = calibration_example()
    permutation = torch.tensor([1, 0])
    a, b = TemplateState(z, spikes), TemplateState(z, spikes[:, permutation])
    torch.testing.assert_close(b.initial_filter, a.initial_filter[:, permutation])
    assert torch.equal(b.evidence(z, b.anchor), a.evidence(z, a.anchor)[:, permutation])


def test_identical_templates_reject_ambiguous_unit_identity():
    z, spikes = calibration_example()
    state = TemplateState(z, spikes)
    h = state.anchor[:, :1].repeat(1, 2)
    assert not state.evidence(z, h).any()


def test_recovery_needs_evidence_and_preserves_cross_batch_refractory():
    primary = torch.zeros(30, 1, dtype=torch.bool)
    candidates = primary.clone()
    candidates[[10, 15]] = True
    last_output, last_primary = torch.tensor([-1000]), torch.tensor([0])
    result, count = merge_recovery(
        primary,
        candidates,
        torch.zeros_like(primary),
        start=100,
        last_output=last_output,
        last_primary=last_primary,
        refractory=20,
        silence=50,
    )
    assert count == 0 and not result.any()
    result, count = merge_recovery(
        primary,
        candidates,
        torch.ones_like(primary),
        start=100,
        last_output=last_output,
        last_primary=last_primary,
        refractory=20,
        silence=50,
    )
    assert count == 1 and result[10, 0] and not result[15, 0]
    candidates[:] = False
    candidates[0] = True
    result, count = merge_recovery(
        primary,
        candidates,
        torch.ones_like(primary),
        start=120,
        last_output=last_output,
        last_primary=last_primary,
        refractory=20,
        silence=50,
    )
    assert count == 0
    assert last_primary.item() == 0  # Rescue events do not create primary evidence.


def test_active_primary_units_are_not_artificially_recruited():
    primary = torch.zeros(30, 1, dtype=torch.bool)
    primary[25] = True
    candidates = torch.ones_like(primary)
    result, count = merge_recovery(
        primary,
        candidates,
        candidates,
        start=100,
        last_output=torch.tensor([90]),
        last_primary=torch.tensor([90]),
        refractory=20,
        silence=50,
    )
    assert count == 0 and torch.equal(result, primary)


def test_reference_protection_uses_kinematics_and_not_a_recording_clock():
    assert reference_envelope(0, 0) == (0, True)
    assert reference_envelope(0.5, 0) == (1, False)
    assert reference_envelope(0, 2) == (0, False)
    assert 0 < reference_envelope(0.07, 0)[0] < 1
