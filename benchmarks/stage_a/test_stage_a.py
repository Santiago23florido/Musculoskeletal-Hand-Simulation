"""Numerical guarantees for the learned correction and the reference integration."""

import numpy as np
import pytest
import torch

from benchmarks.stage_a.model import MotionCorrection, apply_correction, contrast_loss, kl_change

torch.set_num_threads(2)


def test_factorized_transform_and_kl_match_dense_matrices_and_gradients():
    torch.manual_seed(4)
    x = torch.randn(40, 7, dtype=torch.float64)
    u = (torch.randn(7, 3, dtype=torch.float64) * 0.03).requires_grad_()
    v = (torch.randn(7, 3, dtype=torch.float64) * 0.03).requires_grad_()
    a = torch.eye(7) + u @ v.T
    centered = x - x.mean(0)
    c = 0.999 * centered.T @ centered / len(x) + 0.001 * torch.eye(7)
    new_c = a @ c @ a.T
    dense = 0.5 * (
        new_c.trace() - c.trace() - torch.linalg.slogdet(new_c)[1] + torch.linalg.slogdet(c)[1]
    )
    factored = kl_change(x, u, v)
    torch.testing.assert_close(apply_correction(x, u, v), x @ a.T)
    torch.testing.assert_close(factored, dense, atol=1e-9, rtol=1e-7)
    gradient = torch.autograd.grad(factored, (u, v), retain_graph=True)
    expected = torch.autograd.grad(dense, (u, v))
    for actual, target in zip(gradient, expected, strict=True):
        torch.testing.assert_close(actual, target, atol=1e-8, rtol=1e-6)


def test_invertibility_bound_even_with_extreme_weights():
    model = MotionCorrection(dimension=12, features=8, bound=0.8)
    with torch.no_grad():
        model.u_basis.fill_(100)
        model.v_basis.normal_(0, 100)
    features = torch.randn(1, 5, 8)
    features[..., -2] = 1
    u, v, _, _ = model(features)
    for left, right in zip(u[0], v[0], strict=True):
        a = torch.eye(12) + left @ right.T
        assert torch.linalg.matrix_norm(left @ right.T, ord=2) <= 0.80001
        assert torch.linalg.svdvals(a).min() >= 0.19999


def test_identity_initialization_has_finite_nonzero_learning_gradient():
    model = MotionCorrection(dimension=7, features=8)
    f = torch.randn(1, 4, 8)
    f[..., -2] = 1
    u, v, _, _ = model(f)
    assert torch.count_nonzero(u) == 0
    loss = kl_change(torch.randn(4, 30, 7), u[0], v[0]).mean()
    loss.backward()
    assert torch.isfinite(model.u_basis.grad).all()
    assert model.u_basis.grad.abs().sum() > 0


def test_causal_encoder_prefix_is_unchanged_by_future_inputs():
    torch.manual_seed(9)
    model = MotionCorrection(dimension=7, features=8)
    f = torch.randn(1, 12, 8)
    changed = f.clone()
    changed[:, 7:] += 50
    _, _, before, _ = model(f)
    _, _, after, _ = model(changed)
    torch.testing.assert_close(before[:, :7], after[:, :7])


def test_reference_posture_remains_identity_after_training_weights_change():
    model = MotionCorrection(dimension=7, features=8)
    with torch.no_grad():
        model.u_basis.normal_()
    features = torch.randn(1, 4, 8)
    features[..., -2] = 0
    u, _, _, _ = model(features)
    assert torch.count_nonzero(u) == 0


@pytest.mark.parametrize("bound", [0, 1, 1.1])
def test_invalid_spectral_bounds_are_rejected(bound):
    with pytest.raises(ValueError):
        MotionCorrection(bound=bound)


def test_woodbury_coupling_matches_dense_reference_change():
    from benchmarks.stage_a.inference import frame_delta

    torch.manual_seed(0)
    old_u, old_v, u, v = [torch.randn(11, 3, dtype=torch.float64) * 0.02 for _ in range(4)]
    b = torch.randn(4, 11, dtype=torch.float64)
    old_a, a = torch.eye(11) + old_u @ old_v.T, torch.eye(11) + u @ v.T
    expected = -b @ (a - old_a) @ torch.linalg.inv(old_a)
    torch.testing.assert_close(frame_delta(b, u, v, old_u, old_v), expected)


def test_contrast_loss_matches_authors_value_with_inactive_unit():
    from benchmarks.stage_a.common import reference

    reference()
    from adapt_decomp.adaptation.ops import compute_contrast_error

    sources = torch.randn(60, 3)
    mask = torch.rand(60, 3) > 0.9
    mask[:, 2] = False
    target, scale = torch.ones(3), torch.full((3,), 0.2)
    _, error, active, _ = compute_contrast_error(sources, mask, target, "spike_based", scale)
    expected = torch.where(active, error.square(), torch.full_like(error, 9)).mean()
    actual = contrast_loss(sources, mask.float(), target, scale)
    torch.testing.assert_close(actual, expected)


def test_acceptance_requires_both_ramps_and_no_staircase_regression():
    from benchmarks.stage_a.evaluate import gates

    baseline = {"staircase": 91.64, "triangular-ramp10s": 72.06, "triangular-ramp5s": 64.17}
    measured = dict(baseline)
    measured["triangular-ramp10s"] += 5
    measured["triangular-ramp5s"] += 5
    summary = {"pareto": {"by_condition": baseline}, "stage_a": {"by_condition": measured}}
    assert all(c["passed"] for c in gates(summary).values())
    measured["staircase"] -= 0.01
    assert not all(c["passed"] for c in gates(summary).values())
    assert np.isfinite(gates(summary)["staircase"]["delta_pp"])


def test_cholesky_kl_matches_dense_gaussian_formula_in_batches():
    from benchmarks.stage_a.objectives import gaussian_kl

    torch.manual_seed(5)
    windows = torch.randn(3, 40, 9, dtype=torch.float64)
    actual = gaussian_kl(windows)
    expected = []
    for x in windows:
        x = x - x.mean(0)
        c = 0.999 * x.T @ x / len(x) + 0.001 * torch.eye(9, dtype=x.dtype)
        expected.append(0.5 * (c.trace() - torch.linalg.slogdet(c)[1] - 9))
    torch.testing.assert_close(actual, torch.stack(expected))


def test_partial_evaluation_cannot_claim_go_even_if_all_available_deltas_pass(tmp_path):
    from benchmarks.stage_a.evaluate import finalize_results

    conditions = ["staircase", "triangular-ramp10s", "triangular-ramp5s"]
    methods = ["stage_a", "stage_a_no_coupling", "identity_sv_sd"]
    rows = []
    for condition in conditions:
        for method in ["pareto", *methods]:
            rows.append(
                dict(
                    subject="sub-01",
                    condition=condition,
                    snr_db=30,
                    unit=0,
                    method=method,
                    roa=0.5 if method == "pareto" else 0.9,
                    precision=0.9,
                    recall=0.9,
                    f1=0.9,
                    sil=0.9,
                    first_hold_roa=0.9,
                    ramp_roa=0.9,
                    last_hold_roa=0.9,
                )
            )
    plan = dict(
        subjects=["sub-01"],
        conditions=conditions,
        snrs=[30],
        methods={method: {} for method in methods},
    )
    result = finalize_results(
        rows, [dict(method="stage_a", median_batch_ms=1.0)], plan, tmp_path, 1.0
    )
    assert all(c["passed"] for c in result["go_no_go"].values())
    assert result["complete_p1"] is False
    assert result["status"] == "partial diagnostic; not a P1 gate"


def test_subject_partition_without_training_subject_has_null_training_mean():
    from benchmarks.stage_a.evaluate import summarize

    result = summarize(
        [dict(subject="sub-02", condition="staircase", snr_db=30, method="stage_a", roa=0.5)]
    )
    assert result["stage_a"]["trained_subject_mean_roa_pct"] is None
    assert result["stage_a"]["unseen_subject_mean_roa_pct"] == 50


def test_partition_merger_permits_only_summary_changes(tmp_path):
    from benchmarks.stage_a.merge import evaluation_core_sha256

    paths = [tmp_path / f"v{i}.py" for i in range(3)]
    for path, source in zip(
        paths,
        [
            "def evaluate(): return 1\ndef summarize(): return 0\n",
            "def evaluate(): return 1\ndef summarize(): return None\n",
            "def evaluate(): return 2\ndef summarize(): return None\n",
        ],
        strict=True,
    ):
        path.write_text(source)
    assert evaluation_core_sha256(paths[0]) == evaluation_core_sha256(paths[1])
    assert evaluation_core_sha256(paths[0]) != evaluation_core_sha256(paths[2])
