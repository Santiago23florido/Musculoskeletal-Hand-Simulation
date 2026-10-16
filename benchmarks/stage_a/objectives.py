"""Parameter-independent KL constants for the authors' calibrated squared loss."""

import json
import time

import numpy as np
import torch

from .common import OUTPUT, sha256, stub, write_json


@torch.no_grad()
def gaussian_kl(windows, shrinkage=0.001):
    windows = windows - windows.mean(-2, keepdim=True)
    dimension = windows.shape[-1]
    covariance = (windows.transpose(-1, -2) @ windows) / windows.shape[-2]
    covariance = (1 - shrinkage) * 0.5 * (covariance + covariance.transpose(-1, -2))
    covariance.diagonal(dim1=-2, dim2=-1).add_(shrinkage)
    factor, status = torch.linalg.cholesky_ex(covariance)
    if torch.any(status != 0):
        raise FloatingPointError("Invalid reference KL covariance")
    logdet = 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return 0.5 * (covariance.diagonal(dim1=-2, dim2=-1).sum(-1) - logdet - dimension)


def attach_constants(record):
    metadata = record["metadata"]
    cache = (
        OUTPUT / "features" / stub(metadata["subject"], metadata["condition"], metadata["snr_db"])
    )
    path = cache / "kl_constants.npz"
    info = cache / "kl_constants.json"
    if path.exists() and info.exists():
        meta = json.loads(info.read_text())
        if meta["objective_sha256"] != sha256(__file__):
            raise ValueError("KL constants predate objective code")
        with np.load(path) as archive:
            record["kl_baseline"] = torch.from_numpy(archive["baseline"])
            record["kl_cal_mean"] = float(archive["cal_mean"])
            record["kl_cal_std"] = float(archive["cal_std"])
        return record
    from adapt_decomp.preprocessing import extend_data

    started = time.perf_counter()
    calibration, config = record["calibration"], record["config"]
    x = extend_data(torch.from_numpy(calibration.emg).float(), config.ext_fact)
    x = x - x.mean(0, keepdim=True)
    calibration_z = x @ torch.from_numpy(calibration.whitening).float().T
    starts = list(range(0, len(calibration_z) - 2000 + 1, config.batch_size))
    calibration_kl = []
    for index in range(0, len(starts), 8):
        windows = torch.stack([calibration_z[s : s + 2000] for s in starts[index : index + 8]])
        calibration_kl.append(gaussian_kl(windows, config.shrinkage))
    calibration_kl = torch.cat(calibration_kl)
    baseline = torch.full((len(record["features"]),), float("nan"))
    indices = list(range(52, len(baseline), 4))
    for index in range(0, len(indices), 8):
        selected = indices[index : index + 8]
        stops = [min((i + 1) * config.batch_size, len(record["z"])) for i in selected]
        windows = torch.stack([record["z"][stop - 2000 : stop] for stop in stops])
        baseline[selected] = gaussian_kl(windows, config.shrinkage)
    record["kl_baseline"] = baseline
    record["kl_cal_mean"] = float(calibration_kl.mean())
    record["kl_cal_std"] = float(calibration_kl.std().clamp_min(1e-7))
    np.savez(
        path,
        baseline=baseline.numpy(),
        cal_mean=record["kl_cal_mean"],
        cal_std=record["kl_cal_std"],
    )
    write_json(
        info,
        dict(
            objective_sha256=sha256(__file__),
            calibration_mean=record["kl_cal_mean"],
            calibration_std=record["kl_cal_std"],
            baseline_kl_min=float(baseline[indices].min()),
            baseline_kl_max=float(baseline[indices].max()),
            elapsed_seconds=time.perf_counter() - started,
            loss_stride=4,
            fifo_samples=2000,
            labels_loaded=False,
        ),
    )
    print(f"KL constants ready: {metadata['condition']}", flush=True)
    return record
