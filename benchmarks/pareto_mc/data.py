"""Raw EMG only for inference. Label loading is explicit and subject guarded."""

import numpy as np
import torch

from benchmarks.stage_a.data import load_calibration

from .common import DATA, calibration_path, guard, sha256, stub


def load_record(subject, condition, snr, *, phase):
    guard(subject, phase)
    calibration, config = load_calibration(subject, condition, snr)
    path = DATA / "data" / subject / "noisy" / f"{stub(subject, condition, snr)}_emg.npz"
    with np.load(path, allow_pickle=False) as archive:
        raw = torch.from_numpy(np.asarray(archive["emg"], dtype=np.float32))
    return dict(
        raw=raw,
        calibration=calibration,
        config=config,
        subject=subject,
        condition=condition,
        snr_db=snr,
        raw_sha256=sha256(path),
        calibration_sha256=sha256(calibration_path(subject, condition, snr)),
    )


def load_truth(record, *, phase):
    guard(record["subject"], phase)
    from adapt_decomp.utils import load_gt

    subject, condition = record["subject"], record["condition"]
    path = DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_spikes.npz"
    return load_gt(path, n_samples=len(record["raw"]))[:, record["calibration"].gt_matched_indices]


def load_angle(record, *, phase):
    guard(record["subject"], phase)
    subject, condition = record["subject"], record["condition"]
    path = DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_angle.npz"
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["angle"], dtype=np.float32)
