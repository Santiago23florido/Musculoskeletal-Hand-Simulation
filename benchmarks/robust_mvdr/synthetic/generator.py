"""Independent line-source HD-EMG generator with length-dependent MUAPs.

Each fibre carries two Rosenfalck intracellular action potentials that leave the
innervation zone in opposite directions and fade before the tendons. The surface
potential follows from the transmembrane current through a ``1/r`` homogeneous,
isotropic volume conductor. A normalised muscle length ``s`` (1 at calibration)
rescales every unit at once with the constant-volume rules of NeuroMotion's
``MSKpose.len2params``: depth ~ s^-1/2, conduction velocity ~ s^-1, fibre length ~ s.
It is not BioMime, so conclusions drawn here are not an inverse crime on FDSI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_OVERSAMPLE = 4
_DZ_MM = 0.5
_TAPER_MM = 4.0
_END_PLATE_DELAY_MS = 1.0
_MIN_DEPTH_MM = 1.0
_DEPTH_EXP, _CV_EXP, _LENGTH_EXP = -0.5, -1.0, 1.0


@dataclass(frozen=True)
class ElectrodeGrid:
    """Monopolar grid with rows along and columns across the fibres (FDSI: 10 x 10)."""

    n_along: int = 10
    n_across: int = 10
    ied_mm: float = 8.0
    fs: float = 2048.0
    n_samples: int = 64

    @property
    def n_channels(self) -> int:
        return self.n_along * self.n_across

    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        along = (np.arange(self.n_along) - (self.n_along - 1) / 2) * self.ied_mm
        across = (np.arange(self.n_across) - (self.n_across - 1) / 2) * self.ied_mm
        x, y = np.meshgrid(along, across, indexing="ij")
        return x.ravel(), y.ravel()


@dataclass(frozen=True)
class MotorUnitPool:
    """Per-unit geometry and thresholds; per-fibre offsets are ``(n_units, n_fibres)``."""

    x_iz: np.ndarray
    y0: np.ndarray
    depth: np.ndarray
    cv: np.ndarray
    fibre_length: np.ndarray
    iz_fraction: np.ndarray
    weight: np.ndarray
    threshold: np.ndarray
    fibre_dx: np.ndarray
    fibre_dy: np.ndarray
    fibre_dd: np.ndarray

    @property
    def n_units(self) -> int:
        return int(self.x_iz.size)


def sample_motor_unit_pool(
    n_units: int,
    rng: np.random.Generator,
    *,
    n_fibres: int = 8,
    recruitment_range: float = 30.0,
    max_threshold: float = 0.6,
    size_range: float = 30.0,
) -> MotorUnitPool:
    """One muscle following the size principle (Fuglevand-style thresholds)."""
    rank = np.arange(n_units) / n_units
    threshold = (np.exp(np.log(recruitment_range) * rank) - 1) / (recruitment_range - 1)
    threshold *= max_threshold
    weight = np.exp(np.log(size_range) * rank)
    territory = 0.8 + 2.2 * np.sqrt(weight / size_range)
    return MotorUnitPool(
        x_iz=rng.normal(0.0, 4.0, n_units),
        y0=rng.uniform(-30.0, 30.0, n_units),
        depth=rng.uniform(4.0, 14.0, n_units),
        cv=np.clip(rng.normal(4.0, 0.3, n_units), 3.2, 4.8),
        fibre_length=rng.uniform(55.0, 70.0, n_units),
        iz_fraction=np.clip(rng.normal(0.5, 0.05, n_units), 0.35, 0.65),
        weight=weight,
        threshold=threshold,
        fibre_dx=rng.normal(0.0, 1.5, (n_units, n_fibres)),
        fibre_dy=rng.normal(0.0, 1.0, (n_units, n_fibres)) * territory[:, None],
        fibre_dd=rng.normal(0.0, 0.5, (n_units, n_fibres)) * territory[:, None],
    )


def _rosenfalck_d2(z: np.ndarray) -> np.ndarray:
    """Second derivative of 96 z^3 exp(-z) mV behind the wavefront (z in mm)."""
    zc = np.clip(z, 0.0, None)
    return np.where(z > 0, 96.0 * np.exp(-zc) * (6 * zc - 6 * zc**2 + zc**3), 0.0)


class MuapModel:
    """Surface MUAPs of a pool at any normalised muscle length."""

    def __init__(self, grid: ElectrodeGrid):
        self.grid = grid
        self._xe, self._ye = grid.positions()
        n_hi = grid.n_samples * _OVERSAMPLE
        self._t_hi_ms = np.arange(n_hi) / (grid.fs * _OVERSAMPLE) * 1e3
        lag = np.arange(n_hi)[:, None] - _OVERSAMPLE * np.arange(grid.n_samples)[None, :]
        kernel = np.exp(-0.5 * (lag / (0.6 * _OVERSAMPLE)) ** 2)
        self._decimate = kernel / kernel.sum(axis=0, keepdims=True)

    def unit_muap(self, pool: MotorUnitPool, unit: int, s: float) -> np.ndarray:
        """MUAP of one unit, shape ``(n_along, n_across, n_samples)``."""
        if s <= 0:
            raise ValueError("Normalised muscle length must be positive")
        velocity = pool.cv[unit] * s**_CV_EXP  # m/s == mm/ms
        length = pool.fibre_length[unit] * s**_LENGTH_EXP
        fraction = pool.iz_fraction[unit]
        z = np.arange(-fraction * length, (1 - fraction) * length + _DZ_MM / 2, _DZ_MM)
        to_end = np.minimum(z - z[0], z[-1] - z)
        taper = np.where(to_end < _TAPER_MM, 0.5 - 0.5 * np.cos(np.pi * to_end / _TAPER_MM), 1.0)
        front = velocity * (self._t_hi_ms - _END_PLATE_DELAY_MS)
        current = _rosenfalck_d2(front[None, :] - np.abs(z)[:, None]) * taper[:, None]
        x_iz = pool.x_iz[unit] + pool.fibre_dx[unit]
        y = pool.y0[unit] + pool.fibre_dy[unit]
        depth = np.maximum((pool.depth[unit] + pool.fibre_dd[unit]) * s**_DEPTH_EXP, _MIN_DEPTH_MM)
        dx = x_iz[:, None, None] + z[None, None, :] - self._xe[None, :, None]
        dy = (y[:, None] - self._ye[None, :])[:, :, None]
        inverse_r = (1.0 / np.sqrt(dx**2 + dy**2 + depth[:, None, None] ** 2)).mean(axis=0)
        potential = (inverse_r @ current) * _DZ_MM @ self._decimate
        potential *= pool.weight[unit]
        return potential.reshape(self.grid.n_along, self.grid.n_across, self.grid.n_samples)


def discharge_trains(
    pool: MotorUnitPool,
    excitation: float,
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
    *,
    min_rate_hz: float = 8.0,
    rate_gain_hz: float = 60.0,
    max_rate_hz: float = 35.0,
    isi_cov: float = 0.15,
    refractory_s: float = 0.02,
) -> list[np.ndarray]:
    """Discharge sample indices of every unit under constant excitation (empty if inactive)."""
    trains = []
    for threshold in pool.threshold:
        if excitation < threshold:
            trains.append(np.empty(0, dtype=np.int64))
            continue
        rate = min(min_rate_hz + rate_gain_hz * (excitation - threshold), max_rate_hz)
        mean_isi = 1.0 / rate
        n_max = int(n_samples / fs * rate * 1.5) + 4
        isi = np.maximum(rng.normal(mean_isi, isi_cov * mean_isi, n_max), refractory_s)
        times = rng.uniform(0, mean_isi) + np.cumsum(isi) - isi[0]
        times = times[times * fs < n_samples]
        trains.append(np.unique(np.round(times * fs).astype(np.int64)))
    return trains
