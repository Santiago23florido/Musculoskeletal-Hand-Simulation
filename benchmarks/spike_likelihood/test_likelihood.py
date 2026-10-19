import numpy as np
import pytest
import torch

from .data import batch_peaks, decode, labels
from .model import SpikeLikelihood


def test_peak_candidates_ignore_future_blocks_even_at_batch_boundaries():
    rng = np.random.default_rng(9)
    original = rng.random(128, dtype=np.float32)
    changed = original.copy()
    changed[64:] *= 20
    a = batch_peaks(original, 32, 4, 0.1)
    b = batch_peaks(changed, 32, 4, 0.1)
    np.testing.assert_array_equal(a[a < 64], b[b < 64])


def test_zero_correction_preserves_baseline_events_and_learned_scores_can_replace_a_peak():
    model = SpikeLikelihood()
    times = np.asarray([5, 8, 23, 31, 40, 63, 80, 100, 120], dtype=np.int32)
    primary = np.asarray([True, False, True, True, True, True, True, True, True])
    baseline = np.zeros((128, 1), dtype=np.uint8)
    baseline[times[primary], 0] = 1
    record = dict(
        n=128,
        m=1,
        batch=32,
        distance=3,
        time=times,
        unit=np.zeros(len(times), dtype=int),
        initial=baseline[:32],
    )
    with torch.no_grad():
        probability = (
            model(torch.zeros(len(times), 19), torch.from_numpy(primary)).sigmoid().numpy()
        )
    np.testing.assert_array_equal(decode(record, probability, 0.5), baseline)
    record = dict(
        n=64,
        m=1,
        batch=32,
        distance=10,
        time=np.asarray([10, 14]),
        unit=np.asarray([0, 0]),
        initial=np.zeros((0, 1), dtype=np.uint8),
    )
    changed = decode(record, np.asarray([0.7, 0.95], dtype=np.float32), 0.5)
    assert np.flatnonzero(changed[:, 0]).tolist() == [14]


def test_likelihood_rejects_extra_kinematic_columns_and_bounds_residual_odds():
    model = SpikeLikelihood()
    with pytest.raises(ValueError):
        model(torch.zeros(3, 21), torch.zeros(3, dtype=torch.bool))
    with torch.no_grad():
        model.layers[-1].weight.fill_(3)
    primary = torch.tensor([True, False, True])
    prior = (primary.float() * 2 - 1) * 2.1972245773362196
    assert torch.all((model(torch.randn(3, 19), primary) - prior).abs() <= 4.00001)


@pytest.mark.parametrize(
    "subject,condition,snr,phase",
    [
        ("sub-02", "staircase", 30, "training"),
        ("sub-03", "staircase", 15, "validation"),
        ("sub-01", "triangular-ramp5s", 30, "training"),
    ],
)
def test_label_guards_execute_before_loading_supervision(
    monkeypatch, subject, condition, snr, phase
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Supervision accessed before guard")

    monkeypatch.setattr("benchmarks.spike_likelihood.data.supervision", forbidden)
    with pytest.raises(ValueError):
        labels(subject, condition, snr, outer="sub-02", phase=phase)
