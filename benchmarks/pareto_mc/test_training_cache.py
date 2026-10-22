from types import SimpleNamespace

import numpy as np
import pytest

from . import training_cache


def test_lossless_cache_reuses_labels_without_loading_truth_again(tmp_path, monkeypatch):
    monkeypatch.setattr(training_cache, "OUTPUT", tmp_path)
    archive = tmp_path / "source.npz"
    archive.write_bytes(b"test source")
    rng = np.random.default_rng(31)
    original = dict(
        features=rng.normal(size=(31, 22)).astype(np.float32),
        target=rng.integers(0, 2, size=31).astype(np.float32),
        unit=np.zeros(31, dtype=np.int16),
        primary=np.zeros(31, dtype=bool),
        time=np.arange(31, dtype=np.int32),
        initial=np.zeros((20, 1), dtype=np.uint8),
        keep=np.ones(31, dtype=bool),
        lags=np.array([2]),
        truth=[np.array([22, 27])],
        record=dict(
            raw=np.empty((100, 0)),
            config=SimpleNamespace(fs=2048, batch_size=204, spike_min_dist=20),
            calibration=SimpleNamespace(gt_matched_indices=np.array([3])),
        ),
    )
    calls = []

    def loader(*args):
        calls.append(args)
        return original

    case = ("sub-01", "staircase", 30)
    first = training_cache.cached_labelled(archive, case, "training", loader)
    second = training_cache.cached_labelled(archive, case, "training", loader)
    assert len(calls) == 1
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(first[key], value)
            np.testing.assert_array_equal(second[key], value)
    np.testing.assert_array_equal(second["truth"][0], original["truth"][0])
    with pytest.raises(ValueError):
        training_cache.cached_labelled(archive, ("sub-05", *case[1:]), "training", loader)
    archive.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="changed"):
        training_cache.cached_labelled(archive, case, "training", loader)
