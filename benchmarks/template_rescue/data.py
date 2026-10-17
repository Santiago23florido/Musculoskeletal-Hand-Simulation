"""Signal-only spatially whitened multichannel observations."""

import numpy as np
import torch

from benchmarks.pareto_residual.data import prepare as prepare_previous
from benchmarks.stage_a.common import pareto_path

from .common import DATA, stub
from .physics import spatial_whitener


@torch.no_grad()
def prepare(subject, condition, snr, *, physical=True):
    record = prepare_previous(subject, condition, snr)
    # Angle is supplied kinematics, not inferred from post-calibration spikes.
    with np.load(
        DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_angle.npz"
    ) as archive:
        angle = np.asarray(archive["angle"], dtype=np.float32)
    record["angle"] = angle
    record["reference_angle"] = float(angle[: 5 * record["config"].fs].mean())
    from adapt_decomp import AdaptationResult

    baseline = AdaptationResult.load(pareto_path(subject, condition, snr))
    record["baseline_spikes"] = baseline.spikes.bool()
    if not physical:
        return record
    from adapt_decomp.preprocessing import extend_data, filter_kwargs, preprocess_emg

    calibration, config = record["calibration"], record["config"]
    emg_cal = torch.from_numpy(calibration.emg).float()
    whiten = spatial_whitener(emg_cal)
    z_cal = extend_data(emg_cal @ whiten, config.ext_fact, ext_mode=config.ext_mode)
    z_cal -= z_cal.mean(0)
    name = stub(subject, condition, snr)
    with np.load(DATA / "data" / subject / "noisy" / f"{name}_emg.npz") as archive:
        raw = np.asarray(archive["emg"], dtype=np.float32)
    # Same offline bandpass/notch as the benchmark; streaming causality is only
    # claimed AFTER this supplied preprocessing, just as for the prior residual.
    filtered = preprocess_emg(raw, config.fs, **filter_kwargs(config))
    filtered -= filtered.mean(0)
    observed = torch.as_tensor(filtered, dtype=torch.float32) @ whiten
    z = extend_data(observed, config.ext_fact, ext_mode=config.ext_mode)
    for start in range(0, len(z), config.batch_size):
        trim = config.ext_fact if start == 0 else 0
        block = z[start + trim : start + config.batch_size]
        block -= block.mean(0)
    z[: config.ext_fact] = 0
    record.update(z=z, z_cal=z_cal)
    return record
