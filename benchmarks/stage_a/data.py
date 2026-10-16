"""Signal-only training preparation; simulated discharge labels are never loaded here."""

import json
import time

import numpy as np
import torch

from .common import DATA, OUTPUT, calibration_path, pareto_path, reference, sha256, stub, write_json


def load_calibration(subject, condition, snr):
    reference()
    from adapt_decomp import CBSSResult
    from adapt_decomp.adaptation import AdaptConfig

    path = calibration_path(subject, condition, snr)
    calibration = CBSSResult.load(path)
    cfg_path = pareto_path(subject, condition, snr).with_name(
        pareto_path(subject, condition, snr).stem + "_config.yaml"
    )
    config = AdaptConfig.from_yaml(cfg_path)
    config.device, config.compute_loss, config.debug = "cpu", False, False
    return calibration, config


def decoder(calibration, config):
    """Reuse reference SD/SV state initializers; omit obsolete dense whitening KL setup."""
    from adapt_decomp.adaptation.data_structures import Decomposition

    d = Decomposition.__new__(Decomposition)
    for name in (
        "device",
        "batch_size",
        "contrast_scope",
        "eps",
        "spike_det_exp",
        "source_fifo_batches",
    ):
        setattr(d, name, getattr(config, name))
    tensors = calibration.to_adapt_tensors()
    d.sep_vectors = tensors["sep_vectors"].clone()
    d.whitening = tensors["whitening"].clone()
    d.sources_calib, d.spikes_calib = tensors["sources_calib"], tensors["spikes_calib"]
    d.spikes_centr_cal = tensors["spikes_centr"].clone()
    d.base_centr_cal = tensors["base_centr"].clone()
    d.init_sv_update()
    d.init_sd_update()
    return d


@torch.no_grad()
def prepare(subject, condition, snr=30, *, persist=False):
    """Precompute W0 x once; batching and centering match reference offline Data.

    All signal features are causal after the provided 5-s calibration. Covariance
    features are 16 eigenvalues in the calibration PCA basis; source features
    refer to the PREVIOUS batch of frozen calibrated sources.
    """
    calibration, config = load_calibration(subject, condition, snr)
    name = stub(subject, condition, snr)
    cache = OUTPUT / "features" / name
    if persist and (cache / "metadata.json").exists():
        metadata = json.loads((cache / "metadata.json").read_text())
        if metadata["preparation_sha256"] != sha256(__file__):
            raise ValueError("Feature cache predates data preparation code; use a new output")
        return dict(
            z=torch.from_numpy(np.load(cache / "z.npy", mmap_mode="c")),
            features=torch.from_numpy(np.load(cache / "features.npy")),
            calibration=calibration,
            config=config,
            metadata=metadata,
        )
    from adapt_decomp.adaptation.data_structures import Data

    started = time.perf_counter()
    raw_path = DATA / "data" / subject / "noisy" / f"{name}_emg.npz"
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = np.asarray(archive["emg"], dtype=np.float32)
    extended = Data(torch.from_numpy(raw), True, config).emg_ext
    w0 = torch.from_numpy(calibration.whitening).float()
    n, dimension = extended.shape
    batch = config.batch_size
    # Calibration PCA axes for largest original variances (smallest W0 eigenvalues).
    _, axes = torch.linalg.eigh(w0)
    pca = axes[:, :16]
    z = torch.empty_like(extended)
    for start in range(0, n, batch):
        trim = config.ext_fact if start == 0 else 0
        block = extended[start + trim : min(start + batch, n)]
        extended[start + trim : min(start + batch, n)] = block - block.mean(0, keepdim=True)
    # Large GEMM is materially faster than a separate D-by-D GEMM per 100-ms batch.
    torch.mm(extended, w0.T, out=z)
    z[: config.ext_fact] = 0
    del extended
    clean = DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_angle.npz"
    with np.load(clean, allow_pickle=False) as archive:
        angle = np.asarray(archive["angle"], dtype=np.float32)
    features = []
    previous_stats = torch.zeros(4)
    previous_angle = 0.0
    separators = torch.from_numpy(calibration.sep_vectors).float()
    for start in range(0, n, batch):
        end = min(start + batch, n)
        current = z[start:end]
        rms = torch.from_numpy(np.sqrt(np.mean(raw[start:end] ** 2, axis=0) + 1e-12)).log()
        projected = current @ pca
        projected = projected - projected.mean(0, keepdim=True)
        cov = projected.T @ projected / max(1, len(projected))
        spectrum = torch.linalg.eigvalsh(cov).clamp_min(1e-7).log()
        q = float(angle[start:end].mean()) / 40
        velocity = (q - previous_angle) / ((end - start) / config.fs) / 0.2
        features.append(torch.cat([rms, spectrum, previous_stats, torch.tensor([q, velocity])]))
        sources = current @ separators
        absolute = sources.abs()
        previous_stats = torch.stack(
            [
                absolute.median(),
                torch.quantile(absolute, 0.95),
                sources.square().mean().sqrt(),
                (sources.square() > torch.from_numpy(calibration.spikes_centr).float())
                .float()
                .mean(),
            ]
        )
        previous_angle = q
    features = torch.stack(features)
    metadata = dict(
        subject=subject,
        condition=condition,
        snr_db=snr,
        samples=n,
        dimension=dimension,
        batch_size=batch,
        batch_count=len(features),
        features=features.shape[-1],
        calibration_sha256=sha256(calibration_path(subject, condition, snr)),
        raw_signal_sha256=sha256(raw_path),
        preparation_sha256=sha256(__file__),
        elapsed_seconds=time.perf_counter() - started,
        centering="reference offline Data and per-batch centering; first 10 samples padded",
        labels_loaded=False,
    )
    if persist:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "z.npy", z.numpy())
        np.save(cache / "features.npy", features.numpy())
        write_json(cache / "metadata.json", metadata)
    return dict(z=z, features=features, calibration=calibration, config=config, metadata=metadata)
