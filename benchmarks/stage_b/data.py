"""Calibration-only input preparation; separate, guarded supervision extraction."""

import json
import time

import numpy as np
import torch

from benchmarks.stage_a.data import load_calibration

from .common import (
    DATA,
    DIMENSION,
    OUTPUT,
    RADIUS,
    SUBJECTS,
    TRAIN_CONDITIONS,
    calibration_path,
    reference,
    sha256,
    split,
    stub,
    write_json,
)


def windows_mean(emg, times):
    offsets = torch.arange(-RADIUS, RADIUS)
    positions = torch.as_tensor(times, dtype=torch.long)
    positions = positions[(positions >= RADIUS) & (positions < len(emg) - RADIUS)]
    if len(positions) < 3:
        raise ValueError("Insufficient calibration events for an MUAP template")
    return emg[positions[:, None] + offsets].mean(0).T


@torch.no_grad()
def prepare(subject, condition, snr):
    reference()
    from adapt_decomp.adaptation.data_structures import Data
    from adapt_decomp.preprocessing import extend_data

    name = stub(subject, condition, snr)
    folder = OUTPUT / "features" / name
    calibration, config = load_calibration(subject, condition, snr)
    if (folder / "metadata.json").exists():
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["code_sha256"] != sha256(__file__) or metadata["dimension"] != DIMENSION:
            raise ValueError("Preserve existing inputs and choose a new output directory")
        arrays = {
            key: torch.from_numpy(np.load(folder / f"{key}.npy", mmap_mode="c"))
            for key in ("x", "features", "q", "template", "amplitude", "projection", "x_cal")
        }
        return dict(**arrays, calibration=calibration, config=config, metadata=metadata)
    tick = time.perf_counter()
    cal_emg = torch.from_numpy(calibration.emg).float()
    extended_cal = extend_data(cal_emg, config.ext_fact, ext_mode=config.ext_mode)
    extended_cal -= extended_cal.mean(0)
    # Largest-variance calibration PCA axes are the smallest whitening eigenvectors.
    _, axes = torch.linalg.eigh(torch.from_numpy(calibration.whitening).float())
    axes = axes[:, :DIMENSION]
    cal_projected = extended_cal @ axes
    projection = axes / cal_projected.square().mean(0).sqrt().clamp_min(1e-6)
    x_cal = extended_cal @ projection
    captured = float(cal_projected.square().sum() / extended_cal.square().sum())
    template = torch.stack(
        [
            windows_mean(cal_emg, np.flatnonzero(calibration.spikes[:, j]))
            for j in range(calibration.spikes.shape[1])
        ]
    )
    amplitude = template.flatten(1).norm(dim=1).clamp_min(1e-7)
    # Flattening raw channels is row-major on the published 10x10 electrode grid.
    template = (template / amplitude[:, None, None]).reshape(-1, 10, 10, 2 * RADIUS)
    template = template.permute(0, 3, 1, 2)[:, None].contiguous()
    raw_path = DATA / "data" / subject / "noisy" / f"{name}_emg.npz"
    with np.load(raw_path) as archive:
        raw = np.asarray(archive["emg"], dtype=np.float32)
    signal = Data(torch.from_numpy(raw), True, config).emg_ext
    for start in range(0, len(signal), config.batch_size):
        trim = config.ext_fact if start == 0 else 0
        block = signal[start + trim : start + config.batch_size]
        block -= block.mean(0)
    signal[: config.ext_fact] = 0
    effective_fixed = (
        torch.from_numpy(calibration.whitening).float().T
        @ torch.from_numpy(calibration.sep_vectors).float()
    )
    projections = signal @ torch.cat([projection, effective_fixed], -1)
    x, frozen = projections[:, :DIMENSION], projections[:, DIMENSION:]
    with np.load(
        DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_angle.npz"
    ) as archive:
        angle = np.asarray(archive["angle"], dtype=np.float32)
    features, q, previous = [], [], torch.zeros(4)
    channel_scale = cal_emg.square().mean(0).sqrt().clamp_min(1e-6)
    source_scale = float(torch.from_numpy(calibration.sources).float().square().mean().sqrt())
    for start in range(0, len(signal), config.batch_size):
        stop = min(start + config.batch_size, len(signal))
        rms = signal[start:stop, :100].square().mean(0).sqrt().clamp_min(1e-6)
        pca = x[start:stop, :16]
        covariance = pca.T @ pca / len(pca)
        spectrum = torch.linalg.eigvalsh(covariance).clamp_min(1e-6).log()
        features.append(torch.cat([(rms / channel_scale).log(), spectrum, previous]))
        sources = frozen[start:stop] / max(source_scale, 1e-6)
        previous = torch.stack(
            [
                sources.abs().median(),
                sources.abs().quantile(0.95),
                sources.square().mean().sqrt(),
                (sources.abs() > 3).float().mean(),
            ]
        )
        q.append(float(angle[start:stop].mean()) / 40)
    features = torch.stack(features)
    n_cal = round(5 * config.fs / config.batch_size)
    feature_mean = features[:n_cal].mean(0)
    feature_scale = features[:n_cal].std(0).clamp_min(0.1)
    features = ((features - feature_mean) / feature_scale).clamp(-6, 6)
    arrays = dict(
        x=x,
        features=features,
        q=torch.tensor(q),
        template=template,
        amplitude=amplitude,
        projection=projection,
        x_cal=x_cal,
    )
    folder.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        np.save(folder / f"{key}.npy", value.numpy())
    metadata = dict(
        subject=subject,
        condition=condition,
        snr_db=snr,
        samples=len(x),
        units=template.shape[0],
        dimension=DIMENSION,
        features=120,
        calibration_variance_captured=captured,
        code_sha256=sha256(__file__),
        calibration_sha256=sha256(calibration_path(subject, condition, snr)),
        raw_sha256=sha256(raw_path),
        gt_spikes_loaded=False,
        elapsed_seconds=time.perf_counter() - tick,
    )
    write_json(folder / "metadata.json", metadata)
    return dict(**arrays, calibration=calibration, config=config, metadata=metadata)


@torch.no_grad()
def supervision(subject, condition, snr, *, outer, phase):
    partition = split(outer)
    allowed = partition["training"] if phase == "training" else partition["validation"]
    if phase not in ("training", "validation") or subject not in allowed:
        raise ValueError("Held-out subject supervision is forbidden")
    if condition not in TRAIN_CONDITIONS or snr not in ([30, 20] if phase == "training" else [25]):
        raise ValueError("Reserved speed or noise cannot enter model fitting/selection")
    record = prepare(subject, condition, snr)
    folder = OUTPUT / "targets" / stub(subject, condition, snr)
    if (folder / "metadata.json").exists():
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["code_sha256"] != sha256(__file__):
            raise ValueError("Supervision code changed")
        return dict(
            cal=record["template"],
            features=record["features"],
            angle=record["q"],
            **{k: torch.from_numpy(np.load(folder / f"{k}.npy")) for k in ("unit", "q", "target")},
        )
    from adapt_decomp.preprocessing import filter_kwargs, preprocess_emg
    from adapt_decomp.spikes import rate_of_agreement_paired

    calibration, config = record["calibration"], record["config"]
    clean = DATA / "data" / subject / "clean"
    gt_path = clean / f"{subject}_FDSI_{condition}_spikes.npz"
    with np.load(gt_path, allow_pickle=True) as archive:
        timings = archive["spikes"][calibration.gt_matched_indices]
    truth_cal = np.zeros_like(calibration.spikes)
    for unit, times in enumerate(timings):
        times = np.asarray(times, dtype=int)
        truth_cal[times[times < len(truth_cal)], unit] = 1
    _, _, lag = rate_of_agreement_paired(
        truth_cal, calibration.spikes, fs=config.fs, tol_spike_ms=2
    )
    with np.load(clean / f"{subject}_FDSI_{condition}_emg.npz") as archive:
        signal = np.asarray(archive["emg"], dtype=np.float32)
    signal = torch.as_tensor(
        preprocess_emg(signal, config.fs, **filter_kwargs(config)), dtype=torch.float32
    )
    with np.load(clean / f"{subject}_FDSI_{condition}_angle.npz") as archive:
        angles = archive["angle"] / 40
    unit_rows, q_rows, targets, counts = [], [], [], []
    for unit, times in enumerate(timings):
        times = np.asarray(times, dtype=int) - int(lag[unit])
        times = times[(times >= RADIUS) & (times < len(signal) - RADIUS)]
        for center in np.linspace(0, 1, 9):
            selected = times[np.abs(angles[times] - center) <= 1 / 16 + 1e-5]
            if len(selected) < 20:
                continue
            target = windows_mean(signal, selected) / record["amplitude"][unit]
            target = target.reshape(10, 10, 2 * RADIUS).permute(2, 0, 1)[None]
            unit_rows.append(unit)
            q_rows.append(float(angles[selected].mean()))
            targets.append(target)
            counts.append(len(selected))
    arrays = dict(unit=torch.tensor(unit_rows), q=torch.tensor(q_rows), target=torch.stack(targets))
    folder.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        np.save(folder / f"{key}.npy", value.numpy())
    write_json(
        folder / "metadata.json",
        dict(
            code_sha256=sha256(__file__),
            gt_sha256=sha256(gt_path),
            samples=len(unit_rows),
            spike_counts=counts,
            lag_from_initial_calibration_only=lag.tolist(),
            labels=(
                "estimated clean-EMG spike-triggered MUAP trajectories, not true generator MUAPs"
            ),
        ),
    )
    return dict(cal=record["template"], features=record["features"], angle=record["q"], **arrays)


def prepare_subject(subject):
    if subject not in SUBJECTS:
        raise ValueError("Unknown subject")
    from .common import CONDITIONS

    for condition in CONDITIONS:
        for snr in (30, 25, 20, 15):
            record = prepare(subject, condition, snr)
            print(
                "prepared",
                subject,
                condition,
                snr,
                record["metadata"]["calibration_variance_captured"],
                flush=True,
            )


def prepare_targets(subject):
    """One writer per subject, before parallel fold training starts."""
    if subject not in SUBJECTS:
        raise ValueError("Unknown subject")
    training_outer = next(s for s in SUBJECTS[1:] if subject in split(s)["training"])
    validation_outer = next((s for s in SUBJECTS[1:] if subject in split(s)["validation"]), None)
    for condition in TRAIN_CONDITIONS:
        for snr in (30, 20):
            data = supervision(subject, condition, snr, outer=training_outer, phase="training")
            print("targets", subject, condition, snr, len(data["q"]), flush=True)
        if validation_outer:
            data = supervision(subject, condition, 25, outer=validation_outer, phase="validation")
            print("targets", subject, condition, 25, len(data["q"]), flush=True)
