"""Stage B decoding loop with self-cancellation-robust filters.

Inputs, frozen morph/observer weights, FIFO length, gain calibration and the original
SD detector are those of ``benchmarks.stage_b``; only the filter synthesis changes.
"""

import copy
import time
from collections import deque

import numpy as np
import torch

from benchmarks.pareto_residual.detector import Detector

from .common import FIFO_BATCHES, LAG_KNOTS, SUBTRACTION_FLOOR
from .filters import lag_matrix, robust_filters


def _filters(covariance, lags, lag0, rate, spec):
    return robust_filters(
        covariance,
        lags,
        lag0,
        rate,
        rho=spec["rho"],
        loading=spec["loading"],
        alpha=spec["alpha"],
        floor=SUBTRACTION_FLOOR,
    )


def initial_decoder(record, spec, lags, lag0):
    """Calibration filters, gains and centroids, as Stage B but with the robust filter."""
    calibration = record["calibration"]
    rate = torch.from_numpy(calibration.spikes).float().mean(0)
    covariance = record["x_cal"].T @ record["x_cal"] / len(record["x_cal"])
    filters = _filters(covariance, lags, lag0, rate, spec)
    sources = record["x_cal"] @ filters
    spikes = torch.from_numpy(calibration.spikes).bool()
    peak = torch.stack(
        [sources[spikes[:, j], j].square().mean().sqrt() for j in range(spikes.shape[1])]
    ).clamp_min(1e-6)
    gain = torch.from_numpy(calibration.spikes_centr).float().sqrt() / peak
    sources = sources * gain
    calibrated = copy.copy(calibration)
    calibrated.sources = sources.numpy()
    calibrated.spikes_centr = np.asarray(
        [float(sources[spikes[:, j], j].square().mean()) for j in range(spikes.shape[1])],
        dtype=np.float32,
    )
    calibrated.base_centr = np.asarray(
        [float(sources[~spikes[:, j], j].square().mean()) for j in range(spikes.shape[1])],
        dtype=np.float32,
    )
    return calibrated, filters, gain, covariance, rate


@torch.no_grad()
def lag_tables(record, models, knots=LAG_KNOTS):
    """Per-seed lag-complete projected templates of the frozen morph, on a q grid."""
    tables = []
    for morph, _ in models:
        values = []
        for q in torch.linspace(0, 1, knots):
            template = morph(record["template"], q.expand(len(record["template"])))
            values.append(lag_matrix(template, record["projection"], record["amplitude"])[0])
        tables.append(torch.stack(values))
    return tables


def _interpolate(table, q):
    position = float(q.clamp(0, 1)) * (len(table) - 1)
    lower = min(int(position), len(table) - 2)
    fraction = position - lower
    return table[lower] * (1 - fraction) + table[lower + 1] * fraction


@torch.no_grad()
def predict(record, models, spec, *, tables=None, detector_factory=Detector):
    """Static, kinematic (K) or blind (B) state with the robust filter of ``spec``."""
    mode = spec["mode"]
    if mode not in ("static", "k", "b") or (mode != "static" and not models):
        raise ValueError("K and B need the frozen Stage B models; static needs none")
    static_lags, lag0 = lag_matrix(record["template"], record["projection"], record["amplitude"])
    calibrated, filters0, gain, covariance0, rate = initial_decoder(record, spec, static_lags, lag0)
    if mode != "static" and tables is None:
        tables = lag_tables(record, models)
    config = record["config"]
    detector = detector_factory(calibrated, config)
    fifo = deque(maxlen=FIFO_BATCHES)
    for start in range(max(0, len(record["x_cal"]) - 2040), len(record["x_cal"]), 204):
        block = record["x_cal"][start : start + 204]
        fifo.append(block.T @ block / len(block))
    hidden = [None] * len(models)
    units = record["template"].shape[0]
    predictions = np.zeros((len(record["x"]), units), dtype=np.uint8)
    sources_out = np.zeros_like(predictions, dtype=np.float32)
    states, durations = [], []
    for index, start in enumerate(range(0, len(record["x"]), config.batch_size)):
        tick = time.perf_counter()
        stop = min(start + config.batch_size, len(record["x"]))
        trim = config.ext_fact if index == 0 else 0
        block = record["x"][start + trim : stop]
        if mode == "static":
            qs = [torch.tensor(0.0)]
        elif mode == "k":
            qs = [record["q"][index] for _ in models]
        else:
            qs = []
            for seed, (_, observer) in enumerate(models):
                q, hidden[seed] = observer(
                    record["features"][index : index + 1][None], hidden[seed]
                )
                qs.append(q.squeeze())
        states.append(float(torch.stack(qs).mean()))
        if start < 5 * config.fs:
            filters = filters0
        else:
            fifo.append(block.T @ block / len(block))
            covariance = torch.stack(list(fifo)).mean(0) if fifo else covariance0
            if mode == "static":
                lags = static_lags
            else:
                lags = torch.stack(
                    [_interpolate(table, q) for table, q in zip(tables, qs, strict=True)]
                ).mean(0)
            filters = _filters(covariance, lags, lag0, rate, spec)
        sources = block @ filters * gain
        binary, trusted = detector.step(sources, index)
        rate = 0.95 * rate + 0.05 * trusted.float().mean(0)
        predictions[start + trim : stop] = binary.numpy()
        sources_out[start + trim : stop] = sources.numpy()
        durations.append((time.perf_counter() - tick) * 1000)
    predictions[: 5 * config.fs] = record["calibration"].spikes
    return dict(
        spikes=predictions,
        sources=sources_out,
        q=np.asarray(states),
        timing=dict(
            median_core_batch_ms=float(np.median(durations)),
            p95_core_batch_ms=float(np.percentile(durations, 95)),
            scope="projected preprocessed inputs; excludes projection, features and disk I/O",
        ),
    )
