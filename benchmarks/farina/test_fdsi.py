"""Guard the scientific pooling, complete coverage and report acceptance gate."""

import copy

import numpy as np
import pytest
from reproduce_fdsi import (
    CONFIGS,
    TABLE_2,
    aggregate,
    array_header,
    checked_roa,
    compare_table,
    identity_source,
)


def sample_rows():
    rows = []
    # Unequal unit counts expose accidental averaging over recording means.
    for method in CONFIGS:
        for snr, scores in [(15, [0.0]), (20, [1.0] * 3), (25, [0.5]), (30, [0.5])]:
            for unit, score in enumerate(scores):
                rows.append(
                    {
                        "method": method,
                        "subject": "sub-01",
                        "condition": "staircase",
                        "snr_db": snr,
                        "unit": unit,
                        "roa": score,
                    }
                )
    return rows


def test_pool_unit_instances_instead_of_recording_means():
    result = aggregate(sample_rows(), expected_recordings=4, expected_units=6)
    for method in CONFIGS:
        assert result[method]["mean_roa_pct"] == pytest.approx(100 * 4 / 6)
        assert result[method]["pct_units_roa_ge_90"] == 50
        assert result[method]["mean_roa_by_snr_pct"]["15"] == 0
        assert result[method]["unit_instances_by_snr"] == {"15": 1, "20": 3, "25": 1, "30": 1}


def test_missing_or_duplicate_rows_cannot_pass_the_benchmark():
    rows = sample_rows()
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        aggregate(rows[:-1], expected_recordings=4, expected_units=6)
    rows[-1] = rows[-2].copy()
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        aggregate(rows, expected_recordings=4, expected_units=6)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-0.01], [1.01], [[0.5]], []])
def test_invalid_cached_scores_are_rejected(values):
    with pytest.raises(ValueError):
        checked_roa(values, 1)


def test_percentage_point_gate_is_inclusive_and_checks_every_table_column():
    overall = {}
    for method, targets in TABLE_2.items():
        overall[method] = dict(
            zip(["mean_roa_pct", "median_roa_pct", "pct_units_roa_ge_90"], targets[:3], strict=True)
        )
        overall[method]["mean_roa_by_snr_pct"] = dict(
            zip(["15", "20", "25", "30"], targets[3:], strict=True)
        )
    overall["pareto"]["mean_roa_pct"] += 1.0
    assert len(compare_table(overall)) == 28
    assert all(check["passed"] for check in compare_table(overall))
    bad = copy.deepcopy(overall)
    bad["oracle"]["mean_roa_by_snr_pct"]["15"] += 1.01
    assert sum(not check["passed"] for check in compare_table(bad)) == 1


def test_npz_header_inspection_preserves_shape_and_dtype(tmp_path):
    path = tmp_path / "example.npz"
    np.savez_compressed(path, emg=np.zeros((123, 100), dtype=np.float32))
    shape, dtype = array_header(path, "emg")
    assert shape == (123, 100)
    assert dtype == np.dtype("float32")


def test_optional_cache_identities_use_calibration_but_conflicting_order_is_rejected():
    assert identity_source(None, [28, 32]).startswith("calibration;")
    assert identity_source([28, 32], [28, 32]).startswith("cached identities")
    with pytest.raises(ValueError, match="disagree"):
        identity_source([32, 28], [28, 32])
