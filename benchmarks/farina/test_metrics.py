"""Protect against duplicate spike credit and evaluation-window alignment errors."""

import numpy as np
import pytest
from metrics import score_spikes


def test_a_prediction_cannot_explain_two_reference_spikes():
    score = score_spikes([10, 12], [11], tolerance=1)
    assert (score["tp"], score["fn"], score["fp"]) == (1, 1, 0)
    assert score["f1"] == pytest.approx(2 / 3)
    assert score["roa"] == 0.5


def test_lag_is_applied_before_cropping_and_does_not_mutate_input():
    predicted = np.array([8, 18, 28])
    score = score_spikes([10, 20, 30], predicted, tolerance=0, lag=2, start=10, stop=30)
    assert score["roa"] == 1
    assert score["tp"] == 2
    assert predicted.tolist() == [8, 18, 28]


def test_deletion_and_false_detection_are_both_penalized():
    score = score_spikes([10, 20, 30], [10, 20, 50], tolerance=0)
    assert (score["tp"], score["fp"], score["fn"]) == (2, 1, 1)
    assert score["precision"] == score["recall"] == pytest.approx(2 / 3)
    assert score["roa"] == 0.5


def test_empty_and_invalid_trains():
    assert score_spikes([10], [], tolerance=1)["fn"] == 1
    assert score_spikes([], [10], tolerance=1)["fp"] == 1
    assert score_spikes([], [], tolerance=1)["roa"] == 0
    with pytest.raises(ValueError):
        score_spikes([10, 10], [10], tolerance=1)
