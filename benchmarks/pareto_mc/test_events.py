import copy

import numpy as np
import torch

from .closed_loop import ClosedLoop, calibration_template
from .events import EventFeatures, decode_block
from .models import SpikeLikelihood
from .test_backbone import fixture


def test_identity_head_preserves_the_full_feedback_trajectory():
    original, signal = fixture()
    signal = signal.repeat(7, 1)
    head = copy.deepcopy(original)
    head.__class__ = ClosedLoop
    # Use the synthetic calibration without requiring 100-channel C1 features.
    from types import SimpleNamespace

    c = SimpleNamespace(
        sources=original.decomp.sources_calib.numpy(),
        spikes=original.decomp.spikes_calib.numpy(),
        spikes_centr=original.decomp.spikes_centr_cal.numpy(),
        whitening=original.decomp.whitening.numpy(),
        sep_vectors=original.decomp.sep_vectors.T.numpy(),
        sil=np.ones(2),
    )
    head.configure(c, SpikeLikelihood(), observer=None)
    for index, start in enumerate(range(0, len(signal), head.config.batch_size)):
        x = signal[start : start + head.config.batch_size]
        a, b = head.process_batch(x, index), original.process_batch(x, index)
        for actual, expected in zip(a, b, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            head.decomp.sep_vectors, original.decomp.sep_vectors, rtol=0, atol=0
        )
        torch.testing.assert_close(
            head.decomp.spikes_centr, original.decomp.spikes_centr, rtol=0, atol=0
        )


def test_candidate_shape_never_uses_a_future_batch():
    adapter, _ = fixture()
    from types import SimpleNamespace

    c = SimpleNamespace(
        sources=adapter.decomp.sources_calib.numpy(),
        spikes=adapter.decomp.spikes_calib.numpy(),
        spikes_centr=adapter.decomp.spikes_centr_cal.numpy(),
        sil=np.ones(2),
    )
    builder = EventFeatures(c, calibration_template(c))
    p, f = np.random.default_rng(1).normal(size=(2, 205, 2))
    primary = np.zeros_like(p, dtype=bool)
    primary[[20, 100], :] = True
    before = builder.build(p, f, primary, speed=0.1)
    after = builder.build(p.copy(), f.copy(), primary.copy(), speed=0.1)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    restored = decode_block(before, np.where(before["primary"], 0.9, 0.1), 0.5, 205, 2, 20)
    np.testing.assert_array_equal(restored, primary)
