"""Unit tests for benchmarks.pareto_mc.embc_neuromotion."""

from __future__ import annotations

import numpy as np

from benchmarks.pareto_mc.embc_neuromotion import (
    CONDITIONS,
    FS,
    add_gaussian_noise,
    make_neuromotion_config,
    make_profiles,
    select_optimal_10x10,
    spikes_list_to_binary,
)


def test_make_profiles_shapes_and_calibration_window():
    for cond in CONDITIONS:
        effort, angle = make_profiles(cond)
        assert effort.shape == (90 * FS,)
        assert angle.shape == (90 * FS,)
        assert np.allclose(effort, 0.15)
        # First 5 s must be strictly isometric at 0 degrees for CBSS calibration
        assert np.all(angle[: 5 * FS] == 0.0)
        assert angle.max() >= 39.5
        assert angle.min() >= 0.0


def test_select_optimal_10x10_picks_peak_columns():
    emg_3d = np.zeros((10, 32, 200), dtype=np.float32)
    emg_3d[:, 0, :] = 5.0  # wraps around column 0 (cols 27..31, 0..4)
    selected, cols = select_optimal_10x10(emg_3d)
    assert selected.shape == (200, 100)
    assert len(cols) == 10
    assert 0 in cols


def test_spikes_list_to_binary_and_noise():
    binary = spikes_list_to_binary([np.array([10, 50, 9999]), np.array([])], n_samples=100)
    assert binary.shape == (100, 2)
    assert int(binary[:, 0].sum()) == 2
    assert int(binary[:, 1].sum()) == 0

    clean = np.sin(np.linspace(0, 20, 1000)).astype(np.float32).reshape(100, 10)
    noisy = add_gaussian_noise(clean, snr_db=20, seed=123)
    assert noisy.shape == clean.shape
    ratio = np.std(noisy - clean) / np.std(clean)
    assert 0.07 < ratio < 0.13


def test_make_neuromotion_config_valid():
    cfg = make_neuromotion_config(210601)
    assert cfg["SubjectConfiguration"]["SubjectSeed"] == 210601
    assert cfg["MovementConfiguration"]["TargetMuscle"] == "FDSI"
    assert cfg["RecordingConfiguration"]["SamplingFrequency"] == FS
