"""Ablation modes isolate the blend and the confidence input; screening is exact."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from .l2_ablation import (
    HiddenMass,
    NoMixing,
    apply_mode,
    calibration_hits,
    calibration_terms,
    combine,
    contrast_weights,
    mode_settings,
)
from .memory_loop import MemoryConfig
from .test_memory_loop import setup


class FixedObserver:
    """Blind states that sit at, approach and leave the reference posture."""

    def __init__(self):
        self.calls = 0

    def __call__(self, features, hidden):
        q = (0.0, 0.004, 0.012, 0.03, 0.2)[self.calls % 5]
        logvar = (-8.0, -6.0, -8.0, 2.0, -8.0)[self.calls % 5]
        self.calls += 1
        return torch.tensor([[q]]), torch.tensor([[logvar]]), hidden


class NoFeatures:
    def step(self, x):
        return torch.zeros(120)


def observed(config, mode=None):
    _, adapter, signal = setup(config)
    adapter.observer, adapter.extractor = FixedObserver(), NoFeatures()
    if mode is not None:
        apply_mode(adapter, mode)
    # Extend beyond the five-second calibration hold so memory can act.
    return adapter, signal.repeat(8, 1)


def replay_all(adapter, signal):
    spikes = []
    for index, start in enumerate(range(0, len(signal), adapter.config.batch_size)):
        block = signal[start : start + adapter.config.batch_size]
        spikes.append(adapter.process_batch(block, index)[0])
    return torch.cat(spikes), np.asarray(adapter.memory_trace)


def assert_same_trajectory(left, right):
    for key in ("sep_vectors", "whitening", "spikes_centr", "base_centr", "source_fifo"):
        torch.testing.assert_close(
            getattr(left.decomp, key), getattr(right.decomp, key), rtol=0, atol=0
        )


def test_rollout_free_hit_rate_matches_the_calibration_only_loop():
    config = MemoryConfig(bins=8, h=0.03, threshold=0.6, beta_max=0.5)
    adapter, signal = observed(config)
    _, trace = replay_all(adapter, signal)
    mass, positive = calibration_terms(
        np.asarray(adapter.state_trace),
        config.bins,
        config.h,
        config.beta_max,
        batch_size=adapter.config.batch_size,
        fs=adapter.config.fs,
    )
    expected = calibration_hits(mass, positive, config.threshold)
    np.testing.assert_array_equal(trace[:, 3, :].astype(bool), np.repeat(expected[:, None], 2, 1))
    post = np.arange(len(trace)) * adapter.config.batch_size >= 5 * adapter.config.fs
    np.testing.assert_array_equal(trace[post, 0, 0], mass.numpy()[post])
    # The fixture must exercise hits, threshold misses and the variance veto.
    assert expected.any() and not expected[post].all()


def test_feature_mode_keeps_confidence_but_never_blends():
    active = MemoryConfig(bins=8, h=0.03, threshold=0.6, beta_max=0.8)
    feature, signal = observed(active, "feature")
    off, _ = observed(MemoryConfig(**mode_settings(asdict(active), "off")))
    assert isinstance(feature, NoMixing)
    spikes_feature, trace = replay_all(feature, signal)
    spikes_off, _ = replay_all(off, signal)
    # The fixture head is the identity, so only a blend could separate the two.
    torch.testing.assert_close(spikes_feature, spikes_off, rtol=0, atol=0)
    assert_same_trajectory(feature, off)
    assert trace[:, 0].max() >= 0.6 and trace[:, 1].max() == 0 and trace[:, 3].max() == 0


def test_mixing_mode_blends_exactly_as_the_accepted_loop():
    active = MemoryConfig(bins=8, h=0.03, threshold=0.6, beta_max=0.8)
    mixing, signal = observed(active, "mixing")
    full, _ = observed(active)
    off, _ = observed(MemoryConfig(**mode_settings(asdict(active), "off")))
    assert isinstance(mixing.builder.builder, HiddenMass)
    _, trace = replay_all(mixing, signal)
    replay_all(full, signal)
    replay_all(off, signal)
    assert_same_trajectory(mixing, full)
    assert trace[:, 3].max() == 1
    assert not torch.equal(mixing.decomp.sep_vectors, off.decomp.sep_vectors)


def test_hidden_mass_zeroes_only_the_confidence_feature():
    _, adapter, _ = setup(MemoryConfig())
    builder = adapter.builder.builder
    rng = np.random.default_rng(0)
    pareto, memory = rng.normal(size=(204, 2)), rng.normal(size=(204, 2))
    primary = np.zeros((204, 2), dtype=bool)
    primary[[30, 120], [0, 1]] = True
    shown = builder.build(pareto, memory, primary, speed=0.3, mass=np.array([0.7, 0.9]))
    hidden = HiddenMass(builder).build(
        pareto, memory, primary, speed=0.3, mass=np.array([0.7, 0.9])
    )
    np.testing.assert_array_equal(
        np.delete(shown["features"], 20, 1), np.delete(hidden["features"], 20, 1)
    )
    assert set(np.unique(shown["features"][:, 20])) == {np.float32(0.7), np.float32(0.9)}
    assert not hidden["features"][:, 20].any()


def test_contrasts_recover_factorial_effects_and_propagate_undefined_metrics():
    values = {}
    for seed in range(3):
        for mode, value in dict(full=4.0, feature=3.0, mixing=2.0, off=1.0).items():
            values[f"l2_seed{seed}_{mode}"] = dict(roa=value, ramp=None if seed else value)
    values.update(
        l1_off=dict(roa=0.5, ramp=0.5),
        l1_mixing=dict(roa=0.75, ramp=0.75),
        l2_seed2_full=dict(roa=4.0, ramp=None),
    )
    effects = {name: combine(w, values) for name, w in contrast_weights().items()}
    assert effects["memory_total"]["roa"] == pytest.approx(3)
    assert effects["blend_main"]["roa"] == pytest.approx(1)
    assert effects["confidence_input_main"]["roa"] == pytest.approx(2)
    assert effects["interaction"]["roa"] == pytest.approx(0)
    assert effects["c4_retraining"]["roa"] == pytest.approx(0.5)
    assert effects["l1_head_blend"]["roa"] == pytest.approx(0.25)
    assert effects["memory_total"]["ramp"] is None


def test_off_mode_removes_memory_and_unknown_modes_are_rejected():
    settings = asdict(MemoryConfig(beta_max=0.6))
    assert mode_settings(settings, "off")["beta_max"] == 0
    assert mode_settings(settings, "full") == settings
    with pytest.raises(ValueError):
        mode_settings(settings, "partial")
