"""Replay parity and the realistic-dimensional stationary amplitude regression."""

import pytest
import torch

from benchmarks.robust_mvdr.common import METHODS
from benchmarks.robust_mvdr.inference import predict as robust_predict
from benchmarks.robust_mvdr.test_robust import _synthetic_record
from benchmarks.stage_b.inference import predict as stage_b_predict

from . import fdsi_check
from .harness import VARIANTS, amplitude_metrics, compare_predictions, decode, synthetic_record


@pytest.mark.parametrize("method", ["stage_b_static", "static_fifo_mvdr"])
def test_replay_matches_frozen_predict(method):
    torch.set_num_threads(2)
    record, _ = _synthetic_record(seconds=6)
    original = (
        stage_b_predict(record, [], mode="static")
        if method == "stage_b_static"
        else robust_predict(record, [], METHODS["static_fifo_mvdr"])
    )
    compare_predictions(decode(record, VARIANTS[method]), original)


def test_stationary_512_dimensional_fifo_loses_amplitude_in_sample():
    torch.set_num_threads(2)
    record, truth = synthetic_record(seconds=8)
    ratios = {}
    for name in ("static_fifo_mvdr", "fifo_mvdr_out_of_sample"):
        result = decode(record, VARIANTS[name])
        amplitude = amplitude_metrics(
            result["sources"], truth, 2048, dict(calibration=(1, 5), after=(5, 8))
        )
        ratios[name] = amplitude["after"]["median_ratio_vs_calibration"]
    assert ratios["static_fifo_mvdr"] < 0.75, ratios
    assert ratios["fifo_mvdr_out_of_sample"] > 0.8, ratios


@pytest.mark.parametrize("subject", ["sub-02", "sub-03", "sub-04", "sub-05", "all"])
def test_real_data_audit_rejects_any_subject_except_sub01(subject, tmp_path):
    with pytest.raises(ValueError, match="permits only sub-01"):
        fdsi_check.run_check(tmp_path, subject=subject)
    assert not list(tmp_path.iterdir())


def test_missing_cache_is_skipped_without_creating_it(monkeypatch, tmp_path):
    monkeypatch.setattr(fdsi_check, "ROOT", tmp_path)
    assert fdsi_check.cached_record("sub-01", "triangular-ramp5s") is None
    assert not list(tmp_path.iterdir())
