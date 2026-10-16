"""Whitelisted predicted sources and calibration only; no post-calibration GT."""

import json
import time

import numpy as np
import torch

from benchmarks.stage_a.common import calibration_path, pareto_path
from benchmarks.stage_a.data import load_calibration

from .common import CONDITIONS, DATA, OUTPUT, SUBJECTS, sha256, stub, write_json
from .detector import Detector


@torch.no_grad()
def prepare(subject, condition, snr):
    from adapt_decomp import AdaptationResult

    name = stub(subject, condition, snr)
    folder = OUTPUT / "features" / name
    calibration, config = load_calibration(subject, condition, snr)
    if (folder / "metadata.json").exists():
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["preparation_sha256"] != sha256(__file__):
            raise ValueError("Preparation changed: preserve this output and use a new folder")
        arrays = {
            k: torch.from_numpy(np.load(folder / f"{k}.npy", mmap_mode="c"))
            for k in ("base", "frozen", "features", "template")
        }
        return dict(**arrays, metadata=metadata, calibration=calibration, config=config)
    started = time.perf_counter()
    path = pareto_path(subject, condition, snr)
    baseline = AdaptationResult.load(path)
    frozen_path = DATA / "outputs/adaptation" / subject / "fixed" / f"{name}_adapt_fixed.pkl"
    fixed = AdaptationResult.load(frozen_path)
    # Cached objects also contain historical scores. Only predicted sources,
    # predicted spikes and unit ordering are whitelisted; .roa is not read here.
    base, frozen = baseline.sources.float(), fixed.sources.float()
    indices = calibration.gt_matched_indices
    if base.shape != frozen.shape or not np.array_equal(indices, baseline.gt_matched_indices):
        raise ValueError("Calibration and Pareto unit identities differ")
    if fixed.gt_matched_indices is not None and not np.array_equal(
        indices, fixed.gt_matched_indices
    ):
        raise ValueError("Frozen and Pareto unit identities differ")
    with np.load(
        DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_angle.npz"
    ) as archive:
        angle = np.asarray(archive["angle"], dtype=np.float32)
    sources_cal = torch.from_numpy(calibration.sources).float()
    spikes_cal = torch.from_numpy(calibration.spikes).bool()
    scale = sources_cal.square().mean(0).sqrt().clamp_min(1e-5)
    template = []
    radius = 10
    for unit in range(base.shape[1]):
        peaks = spikes_cal[:, unit].nonzero().flatten()
        peaks = peaks[(peaks >= radius) & (peaks < len(sources_cal) - radius)]
        if not len(peaks):
            raise ValueError("No calibration source waveform")
        waves = sources_cal[peaks[:, None] + torch.arange(-radius, radius + 1), unit]
        waves = waves / waves.square().sum(1, keepdim=True).sqrt().clamp_min(1e-6)
        template.append(waves.mean(0))
    template = torch.stack(template)
    template = template / template.square().sum(1, keepdim=True).sqrt().clamp_min(1e-6)
    batch = config.batch_size
    features, previous_q = [], 0.0
    detector = Detector(calibration, config)
    disagreements = 0
    calibration_peak = torch.from_numpy(calibration.spikes_centr).float().sqrt().clamp_min(1e-5)
    for index, start in enumerate(range(0, len(base), batch)):
        stop = min(start + batch, len(base))
        trim = config.ext_fact if index == 0 else 0
        p, f = base[start + trim : stop], frozen[start + trim : stop]
        # Shared features use current batch/previous state, never future samples.
        p_rms = (p.square().mean(0).sqrt() / scale).clamp_min(1e-6).log()
        f_rms = (f.square().mean(0).sqrt() / scale).clamp_min(1e-6).log()
        p_peak = (p.abs().amax(0) / calibration_peak).clamp_min(1e-6).log()
        f_peak = (f.abs().amax(0) / calibration_peak).clamp_min(1e-6).log()
        agreement = (p * f).mean(0) / (p.square().mean(0) * f.square().mean(0)).sqrt().clamp_min(
            1e-6
        )
        q = float(angle[start:stop].mean()) / 40
        speed = (q - previous_q) / ((stop - start) / config.fs) / 0.2
        context = torch.tensor([q, speed]).expand(base.shape[1], -1)
        pooled = p_rms.mean().expand(base.shape[1], 1)
        features.append(
            torch.cat(
                [
                    context,
                    torch.stack([p_rms, f_rms, p_peak, f_peak, agreement], -1),
                    pooled,
                ],
                -1,
            )
        )  # 8 fixed features + 2 detector-state features at replay time.
        binary, _ = detector.step(p, index)
        disagreements += int(torch.count_nonzero(binary != baseline.spikes[start + trim : stop]))
        previous_q = q
    if disagreements:
        raise ValueError(f"Identity detector differs from Pareto at {disagreements} samples")
    folder.mkdir(parents=True, exist_ok=True)
    for key, value in dict(
        base=base, frozen=frozen, features=torch.stack(features), template=template
    ).items():
        np.save(folder / f"{key}.npy", value.numpy())
    metadata = dict(
        subject=subject,
        condition=condition,
        snr_db=snr,
        samples=len(base),
        units=base.shape[1],
        identity_spike_disagreements=disagreements,
        calibration_sha256=sha256(calibration_path(subject, condition, snr)),
        pareto_sha256=sha256(path),
        frozen_sha256=sha256(frozen_path),
        preparation_sha256=sha256(__file__),
        labels_used=False,
        elapsed_seconds=time.perf_counter() - started,
    )
    write_json(folder / "metadata.json", metadata)
    return dict(
        base=base,
        frozen=frozen,
        features=torch.stack(features),
        template=template,
        metadata=metadata,
        calibration=calibration,
        config=config,
    )


def prepare_all():
    for subject in SUBJECTS:
        for condition in CONDITIONS:
            for snr in (30, 25, 20, 15):
                record = prepare(subject, condition, snr)
                print(
                    stub(subject, condition, snr),
                    record["metadata"]["identity_spike_disagreements"],
                    flush=True,
                )
