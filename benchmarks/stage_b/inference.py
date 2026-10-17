"""Closed-form filters in a documented calibration PCA subspace."""

import copy
import time
from collections import deque

import numpy as np
import torch

from benchmarks.pareto_residual.detector import Detector


def extended_template(template, projection, amplitude, extension=10):
    center = template.shape[2] // 2
    positions = center - torch.arange(extension)
    # Same delayed-block channel ordering as adapt_decomp.extend_data.
    extended = template[:, 0, positions].flatten(1) * amplitude[:, None]
    return extended @ projection


def analytic_filters(covariance, template, rate, *, alpha=0.5, ridge=0.001):
    """R^-1 H with model covariance shrinkage and a shared Cholesky solve.

    The isotropic residual preserves observed covariance trace not explained by
    calibrated identities. Unknown units remain represented by the FIFO term.
    """
    dimension = covariance.shape[0]
    eye = torch.eye(dimension, dtype=covariance.dtype, device=covariance.device)
    model = template.T @ (template * rate[:, None])
    residual = ((covariance.trace() - model.trace()) / dimension).clamp_min(0.01)
    covariance = (1 - alpha) * covariance + alpha * (model + residual * eye)
    covariance = covariance + ridge * covariance.trace().clamp_min(1e-6) / dimension * eye
    factor = torch.linalg.cholesky(covariance)
    filters = torch.cholesky_solve(template.T, factor)
    normalization = (template * filters.T).sum(1).clamp_min(1e-8).sqrt()
    return filters / normalization[None]


def initial_decoder(record):
    calibration = record["calibration"]
    rate = torch.from_numpy(calibration.spikes).float().mean(0)
    h0 = extended_template(record["template"], record["projection"], record["amplitude"])
    covariance = record["x_cal"].T @ record["x_cal"] / len(record["x_cal"])
    filters = analytic_filters(covariance, h0, rate)
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
def build_tables(record, models, knots=129):
    """Precompute the learned morph function, using no EMG beyond calibration.

    Projection is linear, so interpolation can be done on the compact projected
    templates. Both neural seeds retain their own template function and observer.
    """
    tables = []
    for morph, _ in models:
        values = []
        for q in torch.linspace(0, 1, knots):
            template = morph(record["template"], q.expand(len(record["template"])))
            values.append(extended_template(template, record["projection"], record["amplitude"]))
        tables.append(torch.stack(values))
    return tables


def interpolate(table, q):
    position = float(q.clamp(0, 1)) * (len(table) - 1)
    lower = min(int(position), len(table) - 2)
    fraction = position - lower
    return table[lower] * (1 - fraction) + table[lower + 1] * fraction


@torch.no_grad()
def predict(record, models, *, mode, oracle=None, alpha=0.5, tables=None):
    """K consumes q; B receives only EMG features. Optional oracle is audit-only."""
    if mode not in ("k", "b", "static", "oracle"):
        raise ValueError("Unknown inference mode")
    calibrated, filters0, gain, covariance0, rate = initial_decoder(record)
    if models and tables is None:
        tables = build_tables(record, models)
    config = record["config"]
    detector = Detector(calibrated, config)
    fifo = deque(maxlen=10)
    for start in range(max(0, len(record["x_cal"]) - 2040), len(record["x_cal"]), 204):
        block = record["x_cal"][start : start + 204]
        fifo.append(block.T @ block / len(block))
    hidden = [None] * len(models)
    predictions = np.zeros((len(record["x"]), record["template"].shape[0]), dtype=np.uint8)
    sources_out = np.zeros_like(predictions, dtype=np.float32)
    state_predictions, durations = [], []
    for index, start in enumerate(range(0, len(record["x"]), config.batch_size)):
        tick = time.perf_counter()
        stop = min(start + config.batch_size, len(record["x"]))
        trim = config.ext_fact if index == 0 else 0
        block = record["x"][start + trim : stop]
        qs = []
        for seed, (_, encoder) in enumerate(models):
            if mode == "b":
                q, hidden[seed] = encoder(record["features"][index : index + 1][None], hidden[seed])
                qs.append(q.squeeze())
            else:
                qs.append(record["q"][index] if mode != "static" else torch.tensor(0.0))
        if mode in ("static", "oracle") and not models:
            qs = [record["q"][index] if mode == "oracle" else torch.tensor(0.0)]
        mean_q = float(torch.stack(qs).mean())
        state_predictions.append(mean_q)
        if start < 5 * config.fs:
            filters = filters0
        else:
            fifo.append(block.T @ block / len(block))
            covariance = torch.stack(list(fifo)).mean(0) if fifo else covariance0
            if mode == "oracle":
                distances = (oracle["q"] - record["q"][index]).abs()
                # Diagnostic interpolation uses training truth only, never a deployable method.
                template = oracle["template"][int(distances.argmin())]
                h = extended_template(template, record["projection"], record["amplitude"])
            elif mode == "static":
                h = extended_template(record["template"], record["projection"], record["amplitude"])
            else:
                h = torch.stack(
                    [interpolate(table, q) for table, q in zip(tables, qs, strict=True)]
                ).mean(0)
            filters = analytic_filters(covariance, h, rate, alpha=alpha)
        sources = block @ filters * gain
        binary, trusted = detector.step(sources, index)
        rate = 0.95 * rate + 0.05 * trusted.float().mean(0)
        predictions[start + trim : stop] = binary.numpy()
        sources_out[start + trim : stop] = sources.numpy()
        durations.append((time.perf_counter() - tick) * 1000)
    # Preserve only the provided initial CBSS calibration; no post-calibration reference spikes.
    predictions[: 5 * config.fs] = record["calibration"].spikes
    return dict(
        spikes=predictions,
        sources=sources_out,
        q=np.asarray(state_predictions),
        timing=dict(
            median_core_batch_ms=float(np.median(durations)),
            p95_core_batch_ms=float(np.percentile(durations, 95)),
            scope="projected preprocessed inputs; excludes projection, features and disk I/O",
        ),
    )
