"""Numerical identities, self-cancellation, angle phases and an end-to-end synthetic run."""

import numpy as np
import pytest
import torch

from benchmarks.stage_b.inference import analytic_filters, extended_template

from .common import METHODS
from .filters import lag_matrix, robust_filters
from .phases import angle_phases, describe, masked

torch.set_num_threads(2)


def test_lag_zero_row_is_the_stage_b_extended_template_and_lags_are_shifts():
    torch.manual_seed(1)
    template = torch.randn(3, 1, 64, 10, 10, dtype=torch.float64)
    projection = torch.randn(1000, 40, dtype=torch.float64)
    amplitude = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    lags, lag0 = lag_matrix(template, projection, amplitude)
    assert lags.shape == (3, 73, 40)
    torch.testing.assert_close(lags[:, lag0], extended_template(template, projection, amplitude))
    shifted = torch.roll(template, shifts=1, dims=2)
    shifted[:, :, 0] = 0
    expected = extended_template(shifted, projection, amplitude)
    torch.testing.assert_close(lags[:, lag0 - 1], expected)


def _problem(seed=2, dimension=24, units=3, lags=5):
    torch.manual_seed(seed)
    x = torch.randn(400, dimension, dtype=torch.float64)
    covariance = x.T @ x / len(x) + torch.eye(dimension, dtype=torch.float64)
    c = torch.randn(units, lags, dimension, dtype=torch.float64) * 0.3
    rate = torch.full((units,), 0.05, dtype=torch.float64)
    return covariance, c, rate


def test_woodbury_matches_dense_solve_and_mvdr_has_unit_response():
    covariance, c, rate = _problem()
    w = robust_filters(covariance, c, 2, rate, rho=0.7, loading=0.05, floor=1e-9)
    mean = covariance.trace() / covariance.shape[0]
    for j in range(c.shape[0]):
        matrix = covariance + 0.05 * mean * torch.eye(covariance.shape[0], dtype=covariance.dtype)
        matrix = matrix - 0.7 * rate[j] * c[j].T @ c[j]
        expected = torch.linalg.solve(matrix, c[j, 2])
        expected = expected / (c[j, 2] @ expected)
        torch.testing.assert_close(w[:, j], expected)
        torch.testing.assert_close(c[j, 2] @ w[:, j], torch.tensor(1.0, dtype=torch.float64))


def test_without_remedies_the_direction_matches_stage_b_with_alpha_zero():
    covariance, c, rate = _problem()
    h = c[:, 2]
    ours = robust_filters(covariance, c, 2, rate, rho=0.0, loading=0.001)
    theirs = analytic_filters(covariance, h, rate, alpha=0.0, ridge=0.001)
    cosine = (ours * theirs).sum(0) / (ours.norm(dim=0) * theirs.norm(dim=0))
    torch.testing.assert_close(cosine, torch.ones(3, dtype=torch.float64))


def test_over_subtraction_stays_finite_and_positive():
    covariance, c, rate = _problem()
    w = robust_filters(covariance, c * 30, 2, rate * 50, rho=1.0, loading=0.0)
    assert torch.isfinite(w).all()
    assert torch.all((c[:, 2] * 30 * w.T).sum(1) > 0)


def _shifted_lags(wave, basis, n_lags):
    rows = []
    for tau in range(n_lags):
        v = torch.zeros(basis.shape[0], dtype=torch.float64)
        for k in range(len(wave)):
            if 0 <= tau - k < len(wave):
                v += wave[tau - k] * basis[:, k]
        rows.append(v)
    return torch.stack(rows)


def test_target_in_covariance_cancels_a_mismatched_template_and_subtraction_restores_it():
    """Minimal reproduction of the Stage B failure mode and its remedy.

    The template is the true waveform shifted by one sample: a mild timing error
    such as conduction-velocity changes produce. With the target in R, MVDR keeps unit
    response to the template but suppresses the true unit; removing the template's own
    lag subspace from R restores much of that response.
    """
    torch.manual_seed(5)
    basis = torch.linalg.qr(torch.randn(40, 40, dtype=torch.float64))[0]
    wave = torch.tensor([0.1, 0.5, 1.0, 0.4, -0.6, -0.3, -0.1], dtype=torch.float64)
    true = _shifted_lags(wave, basis, 13)
    template = _shifted_lags(0.9 * torch.roll(wave, 1) + 0.1 * wave, basis, 13)
    covariance = 0.01 * torch.eye(40, dtype=torch.float64) + 4.0 * true.T @ true / 13
    rate = torch.tensor([4.0 / 13], dtype=torch.float64)
    plain = robust_filters(covariance, template[None], 6, rate, rho=0.0, loading=0.0)
    robust = robust_filters(covariance, template[None], 6, rate, rho=1.0, loading=0.0, floor=1e-3)
    for w in (plain, robust):
        torch.testing.assert_close(template[6] @ w[:, 0], torch.tensor(1.0, dtype=torch.float64))
    assert float(true[6] @ plain[:, 0]) < 0.3
    assert float(true[6] @ robust[:, 0]) > 1.8 * float(true[6] @ plain[:, 0])


def test_angle_phases_separate_rest_motion_and_holds():
    fs = 2048
    staircase = np.concatenate(
        [np.zeros(10 * fs), np.linspace(0, 10, 10 * fs), np.full(5 * fs, 10.0)]
    )
    phases = angle_phases(staircase, fs)
    assert phases["reference_rest"][6 * fs : 9 * fs].all()
    assert not phases["reference_rest"][: 5 * fs].any()  # calibration excluded
    assert phases["moving"][12 * fs : 19 * fs].all()
    assert phases["hold_away"][21 * fs :].all()
    ramp = np.concatenate(
        [np.zeros(5 * fs), np.linspace(0, 40, 5 * fs), np.linspace(40, 0, 5 * fs)]
    )
    summary = describe(ramp, fs)
    assert summary["max_abs_angle_first_10s_deg"] > 39
    assert summary["moving_fraction_first_10s"] > 0.45
    binary = np.ones((ramp.size, 2), dtype=np.uint8)
    out = masked(binary, angle_phases(ramp, fs)["moving"])
    assert out.sum() < binary.sum() and out.dtype == binary.dtype


def _synthetic_record(seed=0, units=3, dimension=48, seconds=12):
    """A small record with the Stage B structure, rendered from known templates."""
    from adapt_decomp import CBSSResult
    from adapt_decomp.adaptation import AdaptConfig

    rng = np.random.default_rng(seed)
    fs, n_cal = 2048, 5 * 2048
    n = seconds * fs
    template = torch.zeros(units, 1, 64, 10, 10)
    t = torch.arange(64).float()
    for u in range(units):
        wave = torch.exp(-0.5 * ((t - 32) / 2.5) ** 2) - 0.6 * torch.exp(-0.5 * ((t - 36) / 3) ** 2)
        spatial = torch.from_numpy(rng.normal(size=(10, 10))).float()
        template[u, 0] = wave[:, None, None] * spatial[None]
    amplitude = template.flatten(1).norm(dim=1)
    template = template / amplitude[:, None, None, None, None]
    emg = rng.normal(scale=0.05, size=(n + 64, 100)).astype(np.float32)
    spikes = np.zeros((n, units), dtype=np.int32)
    for u in range(units):
        times = np.arange(200 + 37 * u, n - 64, 140)
        spikes[times, u] = 1
        shape = (template[u, 0] * amplitude[u]).reshape(64, 100).numpy()
        for s in times:
            emg[s - 32 : s + 32] += shape
    emg = emg[:n]
    extended = np.zeros((n, 1000), dtype=np.float32)
    for i in range(10):
        extended[i:, 100 * i : 100 * (i + 1)] = emg[: n - i]
    extended -= extended[:n_cal].mean(0)
    basis = np.linalg.qr(rng.normal(size=(1000, dimension)))[0].astype(np.float32)
    scale = np.sqrt(((extended[:n_cal] @ basis) ** 2).mean(0)).clip(1e-6)
    projection = torch.from_numpy(basis / scale)
    x = torch.from_numpy(extended) @ projection
    sources = (x[:n_cal] @ torch.eye(dimension)[:, :units]).numpy()
    calibration = CBSSResult(
        sources=sources,
        spikes=spikes[:n_cal],
        spikes_dict={},
        sep_vectors=np.eye(dimension, units, dtype=np.float32),
        whitening=np.eye(dimension, dtype=np.float32),
        extension_mean=np.zeros((1, dimension), dtype=np.float32),
        spikes_centr=np.ones(units, dtype=np.float32),
        base_centr=np.full(units, 0.01, dtype=np.float32),
        sil=np.full(units, 0.9, dtype=np.float32),
        cov_isi=np.full(units, 0.1, dtype=np.float32),
        ext_fact=10,
        emg=emg[:n_cal],
    )
    config = AdaptConfig(fs=fs, device="cpu", compute_loss=False, debug=False)
    batches = (n + config.batch_size - 1) // config.batch_size
    return dict(
        x=x,
        x_cal=x[:n_cal],
        features=torch.zeros(batches, 120),
        q=torch.zeros(batches),
        template=template,
        amplitude=amplitude,
        projection=projection,
        calibration=calibration,
        config=config,
        test_emg=emg,
    ), spikes


@pytest.mark.parametrize("method", ["static_subtract", "static_loaded"])
def test_end_to_end_static_decoding_recovers_synthetic_discharges(method):
    from .inference import predict

    record, truth = _synthetic_record()
    result = predict(record, [], METHODS[method])
    assert result["spikes"].shape == truth.shape
    after = slice(5 * 2048, None)
    for unit in range(truth.shape[1]):
        found = np.flatnonzero(result["spikes"][after, unit])
        expected = np.flatnonzero(truth[after, unit])
        matched = sum(np.min(np.abs(found - e)) <= 4 for e in expected) if len(found) else 0
        assert matched / len(expected) > 0.9


def test_models_are_required_for_kinematic_and_blind_modes():
    from .inference import predict

    record, _ = _synthetic_record(units=2, seconds=6)
    with pytest.raises(ValueError, match="frozen Stage B models"):
        predict(record, [], METHODS["moca_k_subtract"])
