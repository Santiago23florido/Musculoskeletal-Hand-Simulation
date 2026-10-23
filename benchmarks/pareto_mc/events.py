"""C4 event features shared by open preparation and the real closed loop."""

import json

import numpy as np
import torch
from scipy.signal import find_peaks

from benchmarks.stage_a.data import load_calibration

from .common import OUTPUT, ROOT, guard, now, sha256, stub, write_new
from .data import load_truth
from .observer import load_observer


class EventFeatures:
    def __init__(self, calibration, template):
        self.calibration = calibration
        self.template = np.asarray(template)
        self.peak = np.sqrt(np.asarray(calibration.spikes_centr)).clip(1e-6)
        self.scale = np.sqrt(np.mean(calibration.sources**2, axis=0)).clip(1e-5)

    def build(self, pareto, memory, primary, *, speed, mass=None):
        """Only current-batch waveforms; the second view starts at calibration."""
        p = np.asarray(pareto) / self.peak
        f = np.asarray(memory) / self.peak
        primary = np.asarray(primary, dtype=bool)
        m = p.shape[1]
        mass = np.zeros(m) if mass is None else np.asarray(mass)
        context = np.column_stack(
            [
                np.log(np.sqrt(np.mean(pareto**2, axis=0)).clip(1e-6) / self.scale),
                np.log(np.sqrt(np.mean(memory**2, axis=0)).clip(1e-6) / self.scale),
                np.log(np.abs(p).max(0).clip(1e-6)),
                np.log(np.abs(f).max(0).clip(1e-6)),
                np.mean(pareto * memory, axis=0)
                / np.sqrt(np.mean(pareto**2, axis=0) * np.mean(memory**2, axis=0)).clip(1e-6),
                np.full(
                    m, np.log(np.sqrt(np.mean(pareto**2, axis=0)).clip(1e-6) / self.scale).mean()
                ),
            ]
        )
        competition = np.abs(p).max(1).clip(1e-6)
        features, units, times, primaries = [], [], [], []
        offsets = np.arange(-10, 11)
        for j in range(m):
            pp = find_peaks(np.abs(p[:, j]), distance=4, height=0.02)[0]
            ff = find_peaks(np.abs(f[:, j]), distance=4, height=0.02)[0]
            t = np.unique(np.r_[pp, ff, np.flatnonzero(primary[:, j])])
            if not len(t):
                continue
            positions = t[:, None] + offsets
            valid = (positions >= 0) & (positions < len(p))
            index = positions.clip(0, len(p) - 1)
            wp, wf = p[index, j] * valid, f[index, j] * valid
            normp = np.linalg.norm(wp, axis=1).clip(1e-6)
            normf = np.linalg.norm(wf, axis=1).clip(1e-6)
            powerp, powerf = p[t, j] ** 2, f[t, j] ** 2

            def log(value):
                return np.log(np.maximum(value, 1e-6)) * 0.25

            source = (~np.isin(t, pp) & ~primary[t, j]).astype(float)
            z = (
                np.column_stack(
                    [
                        log(powerp),
                        log(powerf),
                        log((wp * wp).max(1)),
                        log((wf * wf).max(1)),
                        log(powerp / (normp**2 / 21 + 1e-6)),
                        log(powerf / (normf**2 / 21 + 1e-6)),
                        np.abs(wp @ self.template[j]) / normp,
                        np.abs(wf @ self.template[j]) / normf,
                        np.abs((wp * wf).sum(1)) / (normp * normf),
                        primary[t, j],
                        log(competition[t] ** 2 / (powerp + 1e-6)),
                        np.full(
                            len(t),
                            np.log(max(float(self.calibration.spikes[:, j].mean() * 2048), 1e-5))
                            * 0.25,
                        ),
                        np.full(len(t), float(np.nan_to_num(self.calibration.sil[j]))),
                        np.broadcast_to(context[j], (len(t), 6)),
                        source,
                        np.full(len(t), mass[j]),
                        np.full(len(t), speed),
                    ]
                )
                .clip(-6, 6)
                .astype(np.float32)
            )
            if z.shape[1] != 22 or not np.isfinite(z).all():
                raise ValueError("Non-finite or undeclared candidate features")
            features.append(z)
            times.append(t.astype(np.int32))
            units.append(np.full(len(t), j, dtype=np.int16))
            primaries.append(primary[t, j])
        if not features:
            return dict(
                features=np.empty((0, 22), dtype=np.float32),
                time=np.empty(0, dtype=np.int32),
                unit=np.empty(0, dtype=np.int16),
                primary=np.empty(0, dtype=bool),
            )
        return dict(
            features=np.concatenate(features),
            time=np.concatenate(times),
            unit=np.concatenate(units),
            primary=np.concatenate(primaries),
        )


def decode_block(events, probability, threshold, samples, units, distance):
    binary = np.zeros((samples, units), dtype=bool)
    for j in range(units):
        mask = events["unit"] == j
        values = np.full(samples + 2, -np.inf)
        values[events["time"][mask] + 1] = probability[mask]
        indices = find_peaks(values, distance=distance, height=threshold)[0] - 1
        binary[indices, j] = True
    return binary


def descriptor(subject, condition, snr, *, phase):
    guard(subject, phase)
    c, cfg = load_calibration(subject, condition, snr)
    metadata = json.loads(
        (
            ROOT
            / "outputs/pareto_residual/features"
            / stub(subject, condition, snr)
            / "metadata.json"
        ).read_text()
    )
    return dict(
        subject=subject,
        condition=condition,
        snr_db=snr,
        calibration=c,
        config=cfg,
        raw=np.empty((metadata["samples"], 0)),
    )


@torch.no_grad()
def prepare_open(subject, condition, snr, *, phase):
    guard(subject, phase)
    name = stub(subject, condition, snr)
    folder = OUTPUT / "L1/open/features"
    path = folder / f"{name}.npz"
    if path.exists():
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if metadata["events_sha256"] != sha256(path) or metadata["code_sha256"] != sha256(__file__):
            raise ValueError("Open candidate cache changed")
        with np.load(path) as archive:
            return {k: archive[k] for k in archive.files}
    record = descriptor(subject, condition, snr, phase=phase)
    c, cfg = record["calibration"], record["config"]
    old = ROOT / "outputs/pareto_residual/features" / name
    p = np.load(old / "base.npy", mmap_mode="r")
    f = np.load(old / "frozen.npy", mmap_mode="r")
    template = np.load(old / "template.npy")
    from adapt_decomp import AdaptationResult

    from .common import pareto_path

    baseline = AdaptationResult.load(pareto_path(subject, condition, snr)).spikes.numpy()
    # Never read the old feature cache: it contains measured angular columns.
    with np.load(OUTPUT / "C1/features" / f"{name}.npz") as archive:
        state_features = torch.from_numpy(archive["features"])
    observer = load_observer()
    mean, _, _ = observer(state_features[None])
    q = mean[0].numpy()
    speed = np.r_[0, np.abs(np.diff(q)) / (cfg.batch_size / cfg.fs)]
    builder = EventFeatures(c, template)
    arrays = {key: [] for key in ("features", "time", "unit", "primary")}
    for index, start in enumerate(range(0, len(p), cfg.batch_size)):
        trim = cfg.ext_fact if index == 0 else 0
        stop = min(start + cfg.batch_size, len(p))
        event = builder.build(
            p[start + trim : stop],
            f[start + trim : stop],
            baseline[start + trim : stop],
            speed=speed[index],
        )
        event["time"] += start + trim
        for key in arrays:
            arrays[key].append(event[key])
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    arrays["initial"] = baseline[: 5 * cfg.fs].astype(np.uint8)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    write_new(
        folder / f"{name}.json",
        dict(
            prepared_at_utc=now(),
            events_sha256=sha256(path),
            true_spikes_loaded=False,
            angular_features=False,
            code_sha256=sha256(__file__),
            source_sha256={k: sha256(old / f"{k}.npy") for k in ("base", "frozen", "template")},
            comparator="Pareto-rerun, identical to published controls in L0",
        ),
    )
    return arrays


def label_events(events, record, *, phase):
    """Calibration-only lag; labels for 2 ms around a candidate, never inference."""
    from adapt_decomp.spikes import rate_of_agreement_paired

    truth = load_truth(record, phase=phase)
    n_cal = 5 * record["config"].fs
    _, _, lags = rate_of_agreement_paired(
        truth[:n_cal], events["initial"], fs=record["config"].fs, tol_spike_ms=2
    )
    labels = np.zeros(len(events["time"]), dtype=np.float32)
    refs = []
    for j in range(truth.shape[1]):
        ref = np.flatnonzero(truth[:, j]) - int(lags[j])
        ref = ref[(ref >= n_cal) & (ref < len(truth))]
        refs.append(ref)
        select = events["unit"] == j
        positions = events["time"][select]
        if len(ref):
            idx = np.searchsorted(ref, positions)
            left, right = ref[(idx - 1).clip(0, len(ref) - 1)], ref[idx.clip(0, len(ref) - 1)]
            labels[select] = np.minimum(np.abs(positions - left), np.abs(positions - right)) <= 4
    return labels, refs, lags
