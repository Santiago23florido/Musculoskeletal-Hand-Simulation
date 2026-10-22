"""Integration contracts: C2 must affect feedback, remain causal, and start at L1."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from .memory_loop import MemoryConfig, MemoryLoop
from .models import SpikeLikelihood
from .test_backbone import fixture
from .test_feedback import configured


def setup(config):
    original, signal = fixture()
    baseline = configured(original, SpikeLikelihood())
    adapter = copy.deepcopy(baseline)
    adapter.__class__ = MemoryLoop
    calibration = adapter.builder.calibration
    calibration.base_centr = adapter.decomp.base_centr_cal.numpy()
    adapter.configure_memory(calibration, config)
    return baseline, adapter, signal


def test_zero_memory_preserves_l1_spikes_and_all_adaptive_states():
    baseline, adapter, signal = setup(MemoryConfig())
    for index, start in enumerate(range(0, len(signal), adapter.config.batch_size)):
        block = signal[start : start + adapter.config.batch_size]
        for a, b in zip(
            adapter.process_batch(block, index), baseline.process_batch(block, index), strict=True
        ):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key in ("sep_vectors", "whitening", "spikes_centr", "base_centr", "source_fifo"):
            torch.testing.assert_close(
                getattr(adapter.decomp, key), getattr(baseline.decomp, key), rtol=0, atol=0
            )


def test_active_memory_is_bounded_and_calibration_snapshot_is_immutable():
    _, adapter, signal = setup(MemoryConfig(bins=8, h=0.03, beta_max=0.8, kappa=0.02))
    initial = adapter.memory.filters.clone()
    # No observer in the small fixture: q=0, logvar=0. Use a deliberately
    # certain state to exercise the active branch rather than uncertainty veto.
    adapter.query_reliability = 1.0
    block = signal[: adapter.config.batch_size]
    adapter.current_index = 60
    adapter.current_x = block
    adapter.retrieved = adapter.memory.filters[0]
    adapter.mass = torch.ones(2)
    adapter.state_trace = [[0.0, -8.0, 0.0]]
    adapter.feedback_mask = torch.ones(len(block), 2, dtype=torch.bool)
    adapter.last_candidate_count = torch.ones(2)
    trusted = torch.ones(len(block), 2, dtype=torch.bool)
    z = torch.randn(len(block), adapter.decomp.sep_vectors.shape[1])
    adapter._update_sep_vectors(z, torch.randn(len(block), 2), trusted, 60)
    trace = adapter.memory_trace[-1]
    assert (trace[1] > 0).all() and (trace[2] > 0).any()
    assert trace[2].max() <= 0.02 + 1e-6
    assert torch.equal(adapter.memory.filters, initial)
    assert torch.count_nonzero(adapter.memory.quality[1:]) == 0


def test_active_memory_cannot_observe_future_emg():
    _, adapter, signal = setup(MemoryConfig(bins=8, h=0.03, beta_max=0.3))
    other = copy.deepcopy(adapter)
    changed = signal.clone()
    cut = adapter.config.batch_size * 4
    changed[cut:] *= 100
    for index, start in enumerate(range(0, cut, adapter.config.batch_size)):
        a = adapter.process_batch(signal[start : start + adapter.config.batch_size], index + 60)
        b = other.process_batch(changed[start : start + adapter.config.batch_size], index + 60)
        for x, y in zip(a, b, strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    np.testing.assert_array_equal(adapter.memory_trace, other.memory_trace)
    assert np.asarray(adapter.memory_trace)[:, 1].max() > 0
    assert torch.equal(adapter.memory.filters, other.memory.filters)


def test_undeclared_memory_settings_are_rejected():
    for kwargs in ({"bins": 9}, {"h": 0.5}, {"kappa": 0.01}, {"beta_max": 0.9}):
        with pytest.raises(ValueError):
            replace(MemoryConfig(), **kwargs)
