"""Identity and raw-input causality, including the filter and detector FIFO state."""

import copy

import numpy as np
import pytest
import torch

from .backbone import AdaptDecomp, ParetoMC, build
from .common import guard, write_new


def fixture():
    from adapt_decomp import CBSSResult
    from adapt_decomp.adaptation import AdaptConfig

    torch.manual_seed(210)
    n, channels, ext, units = 2048, 3, 2, 2
    signal = torch.randn(n, channels)
    sources = torch.randn(n, units)
    spikes = torch.zeros(n, units, dtype=torch.int32)
    spikes[50::101] = 1
    sources[spikes.bool()] = 4
    config = AdaptConfig(
        device="cpu",
        fs=2048,
        ext_fact=ext,
        compute_loss=False,
        lr_mode="fixed",
        contrast_scope="spike_based",
        wh_sv_coupling=True,
        wh_learning_rate=0.04038249056090869,
        sv_learning_rate=0.0024222252021885837,
    )
    calibration = CBSSResult(
        sources=sources.numpy(),
        spikes=spikes.numpy(),
        spikes_dict={},
        sep_vectors=np.eye(channels * ext, units, dtype=np.float32),
        whitening=np.eye(channels * ext, dtype=np.float32),
        extension_mean=np.zeros((1, channels * ext), dtype=np.float32),
        spikes_centr=np.full(units, 16, dtype=np.float32),
        base_centr=np.full(units, 0.1, dtype=np.float32),
        sil=np.full(units, 0.9),
        cov_isi=np.full(units, 0.1),
        ext_fact=ext,
        emg=signal.numpy(),
    )
    return build(calibration, config), signal


def test_c0_matches_original_spikes_sources_and_all_adaptive_states():
    adapter, signal = fixture()
    original = copy.deepcopy(adapter)
    original.__class__ = AdaptDecomp
    assert isinstance(adapter, ParetoMC)
    for index, start in enumerate(range(0, len(signal), adapter.config.batch_size)):
        x = signal[start : start + adapter.config.batch_size]
        for actual, expected in zip(
            adapter.process_batch(x, index), original.process_batch(x, index), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name in ("whitening", "sep_vectors", "spikes_centr", "base_centr", "source_fifo"):
            torch.testing.assert_close(
                getattr(adapter.decomp, name), getattr(original.decomp, name), rtol=0, atol=0
            )


def test_raw_future_cannot_change_prefix_predictions_or_state():
    adapter, signal = fixture()
    other = copy.deepcopy(adapter)
    changed = signal.clone()
    cut = 4 * adapter.config.batch_size
    changed[cut:] += 100
    for index, start in enumerate(range(0, cut, adapter.config.batch_size)):
        x = signal[start : start + adapter.config.batch_size]
        y = changed[start : start + adapter.config.batch_size]
        for actual, expected in zip(
            adapter.process_batch(x, index), other.process_batch(y, index), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(adapter.decomp.whitening, other.decomp.whitening, rtol=0, atol=0)


def test_guard_rejects_sub05_before_any_input_is_opened():
    for phase in ("training", "validation", "identity", "test"):
        with pytest.raises(ValueError, match="sub-05"):
            guard("sub-05", phase)
    with pytest.raises(ValueError):
        guard("sub-04", "training")


def test_registered_artifacts_are_never_overwritten(tmp_path):
    path = tmp_path / "protocol.json"
    write_new(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_new(path, {"a": 2})
