"""Conventions of the synthetic generator used to fix the robust filter."""

import numpy as np

from .synthetic.generator import ElectrodeGrid, MuapModel, discharge_trains, sample_motor_unit_pool
from .synthetic.study import R_EXT, all_lags, ext_template, extend


def test_muaps_are_deterministic_finite_and_change_with_length():
    grid = ElectrodeGrid()
    first = sample_motor_unit_pool(8, np.random.default_rng(3))
    second = sample_motor_unit_pool(8, np.random.default_rng(3))
    model = MuapModel(grid)
    rest = model.unit_muap(first, 0, 1.0)
    assert rest.shape == (10, 10, 64) and np.isfinite(rest).all()
    np.testing.assert_array_equal(rest, model.unit_muap(second, 0, 1.0))
    shortened = model.unit_muap(first, 0, 0.94)
    assert np.linalg.norm(shortened - rest) / np.linalg.norm(rest) > 0.05


def test_extended_templates_follow_the_block_order_of_extend():
    rng = np.random.default_rng(0)
    pool = sample_motor_unit_pool(4, rng)
    muap = MuapModel(ElectrodeGrid()).unit_muap(pool, 0, 1.0).reshape(100, 64)
    lags = all_lags(muap)
    assert lags.shape == (64 + R_EXT - 1, 100 * R_EXT)
    np.testing.assert_array_equal(lags[20], ext_template(muap, 20))
    signal = np.zeros((200, 100))
    signal[50:114] = muap.T
    np.testing.assert_allclose(extend(signal)[50 + 20], lags[20], rtol=1e-6, atol=1e-6)


def test_discharge_trains_respect_recruitment():
    pool = sample_motor_unit_pool(64, np.random.default_rng(3))
    trains = discharge_trains(pool, 0.15, 2048 * 5, 2048.0, np.random.default_rng(1))
    active = [len(t) > 0 for t in trains]
    assert all(active[u] == (pool.threshold[u] <= 0.15) for u in range(64))
    assert all(np.all(np.diff(t) >= 0.02 * 2048 - 1) for t in trains if len(t) > 1)
