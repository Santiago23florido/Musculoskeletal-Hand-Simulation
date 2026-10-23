"""Shared input computation must preserve every independent adaptive trajectory."""

import copy

import numpy as np
import torch

from .backbone import replay
from .memory_loop import MemoryConfig
from .shared_rollout import shared_replay
from .test_backbone import fixture
from .test_memory_loop import setup


def test_shared_whitening_matches_independent_spikes_sources_and_detector_states():
    _, adapter, signal = setup(MemoryConfig(bins=8, h=0.03, beta_max=0.8))
    # Extend the synthetic input beyond the calibration hold to exercise memory.
    signal = signal.repeat(8, 1)
    independent = [copy.deepcopy(adapter), copy.deepcopy(adapter)]
    shared = [copy.deepcopy(adapter), copy.deepcopy(adapter)]
    independent[1].memory_config = MemoryConfig(bins=8, h=0.03, beta_max=0.2)
    shared[1].memory_config = independent[1].memory_config
    master, _ = fixture()
    expected = [replay(signal, value) for value in independent]
    actual = shared_replay(signal, master, shared, collect_sources=True)
    for left, right, reference, adapter in zip(expected, actual, independent, shared, strict=True):
        for key in ("spikes", "sources", "centroid", "trusted_count"):
            np.testing.assert_array_equal(left[key], right[key])
        for key in ("sep_vectors", "whitening", "spikes_centr", "base_centr", "source_fifo"):
            torch.testing.assert_close(
                getattr(reference.decomp, key), getattr(adapter.decomp, key), rtol=0, atol=0
            )
    assert np.asarray(shared[0].memory_trace)[:, 1].max() > 0
    assert not torch.equal(shared[0].decomp.sep_vectors, shared[1].decomp.sep_vectors)
