"""Posture inputs are causal and batch-aligned; the posture head starts at the prior."""

import math

import numpy as np
import pytest
import torch

from .posture_c4 import (
    LAG_BATCHES,
    PostureLikelihood,
    hold_calibration,
    posture_columns,
    speed_mismatch,
)


def test_posture_columns_use_the_candidate_batch_and_only_past_batches():
    q = np.linspace(0, 0.9, 50)
    time = np.array([0, 203, 204, 204 * LAG_BATCHES + 7, 204 * 30])
    columns = posture_columns(q, time, 204)
    np.testing.assert_allclose(columns[:, 0], q[[0, 0, 1, LAG_BATCHES, 30]], rtol=1e-6)
    expected = q[[0, 0, 1, LAG_BATCHES, 30]] - q[[0, 0, 0, 0, 30 - LAG_BATCHES]]
    np.testing.assert_allclose(columns[:, 1], expected, rtol=1e-6, atol=1e-7)
    future = q.copy()
    future[31:] = 5.0
    np.testing.assert_array_equal(posture_columns(future, time, 204), columns)


def test_speed_check_detects_a_one_batch_misalignment():
    rng = np.random.default_rng(1)
    speed = rng.uniform(0, 2, 40)
    time = np.sort(rng.integers(0, 204 * 40, 500))
    features = np.zeros((len(time), 22), dtype=np.float32)
    features[:, 21] = speed[time // 204].astype(np.float32)
    assert speed_mismatch(features, time, speed, 204) == 0
    shifted = np.minimum(time + 204, 204 * 40 - 1)
    assert speed_mismatch(features, shifted, speed, 204) > 0.01


def test_posture_head_starts_at_the_prior_and_rejects_undeclared_inputs():
    head = PostureLikelihood()
    primary = torch.tensor([True, False, True])
    logits = head(torch.randn(3, 24), primary)
    torch.testing.assert_close(logits, (primary.float() * 2 - 1) * math.log(9))
    with pytest.raises(ValueError):
        head(torch.randn(3, 22), primary)
    with pytest.raises(ValueError):
        PostureLikelihood(features=22)


def test_last_calibration_batch_keeps_pareto_detections():
    fs, batch = 2048, 204
    stop = math.ceil(5 * fs / batch) * batch
    events = dict(
        initial=np.zeros((5 * fs, 2), dtype=np.uint8),
        time=np.array([10300, 10350, stop + 50]),
        unit=np.array([0, 1, 0]),
        primary=np.array([False, True, False]),
    )
    binary = np.zeros((stop + 200, 2), dtype=np.uint8)
    binary[10300, 0] = binary[stop + 50, 0] = 1  # accepted non-Pareto candidates
    binary = hold_calibration(binary, events, batch, fs)
    assert binary[10300, 0] == 0 and binary[10350, 1] == 1 and binary[stop + 50, 0] == 1
