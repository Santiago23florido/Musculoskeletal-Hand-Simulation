import json

import numpy as np
import torch
from scipy.signal import find_peaks

from benchmarks.pareto_residual.data import prepare as source_record
from benchmarks.stage_b.common import OUTPUT as STAGE_B_OUTPUT
from benchmarks.stage_b.data import supervision
from benchmarks.stage_b_separation.states import guarded

from .common import (
    DATA,
    OUTPUT,
    SUBJECTS,
    pareto_path,
    reference,
    sha256,
    stub,
    write_json,
)


def batch_peaks(values, batch, distance, height):
    n = len(values)
    blocks = (n + batch - 1) // batch
    matrix = np.full((blocks, batch), -np.inf, dtype=np.float32)
    matrix.ravel()[:n] = values
    padded = np.pad(matrix, ((0, 0), (distance, distance)), constant_values=-np.inf).ravel()
    peaks, _ = find_peaks(padded, distance=distance, height=height)
    width = batch + 2 * distance
    positions = (peaks // width) * batch + (peaks % width) - distance
    return positions[(positions >= 0) & (positions < n)].astype(np.int32)


def decode(record, probability, threshold):
    n, m = record["n"], record["m"]
    predictions = np.zeros((n, m), dtype=np.uint8)
    for j in range(m):
        mask = record["unit"] == j
        values = np.full(n, -np.inf, dtype=np.float32)
        values[record["time"][mask]] = probability[mask]
        peaks = batch_peaks(values, record["batch"], record["distance"], threshold)
        predictions[peaks, j] = 1
    predictions[: len(record["initial"])] = record["initial"]
    return predictions


@torch.no_grad()
def prepare(subject, condition, snr):
    reference()
    from adapt_decomp import AdaptationResult

    name = stub(subject, condition, snr)
    folder = OUTPUT / "features" / name
    source = source_record(subject, condition, snr)
    c, cfg = source["calibration"], source["config"]
    if (folder / "metadata.json").exists():
        meta = json.loads((folder / "metadata.json").read_text())
        if meta["code_sha256"] != sha256(__file__):
            raise ValueError("Preserve previous event features")
        with np.load(folder / "events.npz") as a:
            arrays = {k: a[k] for k in a.files}
        return dict(
            **arrays,
            **{k: meta[k] for k in ("n", "m", "batch", "distance")},
            calibration=c,
            config=cfg,
            metadata=meta,
        )
    baseline = (
        AdaptationResult.load(pareto_path(subject, condition, snr)).spikes.numpy().astype(np.uint8)
    )
    peak = np.sqrt(np.asarray(c.spikes_centr)).clip(1e-6)
    p = source["base"].numpy() / peak
    f = source["frozen"].numpy() / peak
    n, m = p.shape
    rows, units, times, primary = [], [], [], []
    offsets = np.arange(-10, 11)
    context = source["features"].numpy()[:, :, 2:]  # discard both true kinematic columns
    template = source["template"].numpy()
    competition = np.maximum(np.abs(p).max(1), 1e-6)
    for j in range(m):
        tp = batch_peaks(np.abs(p[:, j]), cfg.batch_size, 4, 0.02)
        tf = batch_peaks(np.abs(f[:, j]), cfg.batch_size, 4, 0.02)
        t = np.unique(np.concatenate([tp, tf, np.flatnonzero(baseline[:, j])])).astype(np.int32)
        index = t[:, None] + offsets
        start = (t // cfg.batch_size) * cfg.batch_size
        valid = (index >= start[:, None]) & (index < np.minimum(start + cfg.batch_size, n)[:, None])
        positions = index.clip(0, n - 1)
        wp = p[positions, j] * valid
        wf = f[positions, j] * valid
        normp = np.linalg.norm(wp, axis=1).clip(1e-6)
        normf = np.linalg.norm(wf, axis=1).clip(1e-6)
        pp, pf = p[t, j] ** 2, f[t, j] ** 2

        def log(v):
            return np.log(np.maximum(v, 1e-6)) * 0.25

        z = (
            np.column_stack(
                [
                    log(pp),
                    log(pf),
                    log((wp * wp).max(1)),
                    log((wf * wf).max(1)),
                    log(pp / (normp**2 / 21 + 1e-6)),
                    log(pf / (normf**2 / 21 + 1e-6)),
                    np.abs(wp @ template[j]) / normp,
                    np.abs(wf @ template[j]) / normf,
                    np.abs((wp * wf).sum(1)) / (normp * normf),
                    baseline[t, j],
                    log(competition[t] ** 2 / (pp + 1e-6)),
                    np.full(
                        len(t), np.log(max(float(c.spikes[:, j].mean() * cfg.fs), 1e-5)) * 0.25
                    ),
                    np.full(len(t), float(np.nan_to_num(c.sil[j]))),
                    context[t // cfg.batch_size, j],
                ]
            )
            .clip(-6, 6)
            .astype(np.float32)
        )
        if z.shape[1] != 19 or not np.isfinite(z).all():
            raise ValueError("Invalid event features")
        rows.append(z)
        times.append(t)
        units.append(np.full(len(t), j, dtype=np.int16))
        primary.append(baseline[t, j].astype(bool))
    arrays = dict(
        features=np.concatenate(rows),
        time=np.concatenate(times),
        unit=np.concatenate(units),
        primary=np.concatenate(primary),
        initial=baseline[: len(c.spikes)],
    )
    record = dict(
        **arrays,
        n=n,
        m=m,
        batch=cfg.batch_size,
        distance=cfg.spike_min_dist,
        calibration=c,
        config=cfg,
    )
    identity = decode(record, np.where(arrays["primary"], 0.9, 0.1).astype(np.float32), 0.5)
    differences = int(np.count_nonzero(identity != baseline))
    if differences:
        raise ValueError(
            f"Identity likelihood changed Pareto at {differences} samples: "
            f"{np.argwhere(identity != baseline)[:8].tolist()}"
        )
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "events.npz", **arrays)
    meta = dict(
        subject=subject,
        condition=condition,
        snr_db=snr,
        n=n,
        m=m,
        batch=cfg.batch_size,
        distance=cfg.spike_min_dist,
        candidates=len(arrays["time"]),
        features=19,
        code_sha256=sha256(__file__),
        events_sha256=sha256(folder / "events.npz"),
        source_metadata=source["metadata"],
        gt_spikes_loaded=False,
        angle_inputs=False,
        baseline_spike_disagreements=differences,
    )
    write_json(folder / "metadata.json", meta)
    return dict(**record, metadata=meta)


def labels(subject, condition, snr, *, outer, phase):
    guarded(subject, condition, snr, outer, phase)
    # Reuse the validated initial-calibration lags, not full-record scoring lags.
    supervision(subject, condition, snr, outer=outer, phase=phase)
    record = prepare(subject, condition, snr)
    name = stub(subject, condition, snr)
    target_meta = json.loads((STAGE_B_OUTPUT / "targets" / name / "metadata.json").read_text())
    path = DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_spikes.npz"
    with np.load(path, allow_pickle=True) as archive:
        timings = archive["spikes"][record["calibration"].gt_matched_indices]
    y = np.zeros(len(record["time"]), dtype=np.float32)
    truth = []
    for j, t in enumerate(timings):
        t = np.asarray(t, dtype=int) - int(target_meta["lag_from_initial_calibration_only"][j])
        t = t[(t >= 0) & (t < record["n"])]
        truth.append(t)
        mask = record["unit"] == j
        positions = record["time"][mask]
        insertion = np.searchsorted(t, positions)
        left = t[(insertion - 1).clip(0, len(t) - 1)]
        right = t[insertion.clip(0, len(t) - 1)]
        y[mask] = np.minimum(np.abs(positions - left), np.abs(positions - right)) <= 4
    return dict(**record, target=y, truth=truth)


def prepare_subject(subject):
    if subject not in SUBJECTS:
        raise ValueError("Unknown subject")
    from .common import CONDITIONS

    for condition in CONDITIONS:
        for snr in (30, 25, 20, 15):
            r = prepare(subject, condition, snr)
            print("events", subject, condition, snr, len(r["time"]), flush=True)
