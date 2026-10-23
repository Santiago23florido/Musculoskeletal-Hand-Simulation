"""Own causal process_batch loop, using the pinned Pareto numerical primitives."""

import copy
import time

import numpy as np
import torch

from .common import reference

reference()
from adapt_decomp import AdaptDecomp  # noqa: E402
from adapt_decomp.adaptation.ops import gate_spikes_by_iqr  # noqa: E402


class ParetoMC(AdaptDecomp):
    """C0 is exactly Pareto; a head changes decisions BEFORE centroid/SV feedback.

    The public batch interface accepts only raw EMG and its sequential index.
    Full-record Data/filtfilt and angular traces are never used in this loop.
    """

    head = None
    collector = None
    observer = None

    @torch.no_grad()
    def process_batch(self, emg_batch, batch_idx=None):
        n_called = emg_batch.shape[0]
        if self.data_preprocessed:
            pad = self.config.ext_fact if batch_idx == 0 else 0
            x = emg_batch[pad:]
        else:
            x, pad = self._preprocess_batch_raw(emg_batch)
        x = x - x.mean(0, keepdim=True)
        z, coupling = self._whiten(x, batch_idx)
        self._apply_wh_sv_coupling(coupling)
        sources = z @ self.decomp.sep_vectors.T
        spikes, peaks = self._detect_spikes(sources, len(sources))
        if self.head is not None or self.collector is not None:
            spikes, peaks = self.decide(sources, spikes, peaks, x, pad, batch_idx)
        if self.config.adapt_sd or self.config.adapt_sv:
            trusted = gate_spikes_by_iqr(
                sources,
                spikes,
                self.decomp.Q75_cal,
                self.decomp.IQR_cal,
                gate_factor=3.0,
                peak_power=self.config.spike_det_exp,
                use_abs_for_detection=True,
                eps=self.config.eps,
            )
        else:
            trusted = spikes
        self._update_spike_det(sources, spikes, peaks, trusted, batch_idx)
        self._update_sep_vectors(z, sources, trusted, batch_idx)
        self.last_trusted = trusted.sum(0).clone()
        self.last_spikes = spikes.sum(0).clone()
        binary = torch.zeros(n_called, sources.shape[1], dtype=torch.int32)
        output = torch.zeros_like(binary, dtype=torch.float32)
        binary[pad:], output[pad:] = spikes.int(), sources
        return binary, output

    def decide(self, sources, spikes, peaks, x, pad, index):
        raise NotImplementedError("C4 is installed only after the C0 identity gate")


def build(calibration, config, *, cls=ParetoMC):
    config = copy.deepcopy(config)
    config.device, config.compute_loss, config.debug = "cpu", False, False
    if config.lr_mode != "fixed" or config.contrast_scope != "spike_based":
        raise ValueError("Expected the pinned fixed-rate spike-based Pareto configuration")
    tensors = calibration.to_adapt_tensors()
    adapter = cls(
        whitening=tensors["whitening"],
        sep_vectors=tensors["sep_vectors"],
        base_centr=tensors["base_centr"],
        spikes_centr=tensors["spikes_centr"],
        emg_calib=tensors["emg_calib"],
        sources_calib=tensors["sources_calib"],
        spikes_calib=tensors["spikes_calib"],
        adapt_config=config,
        pca_components=tensors["pca_components"],
        pca_mean=tensors["pca_mean"],
    )
    adapter.data_preprocessed = False
    return adapter


@torch.no_grad()
def replay(raw, adapter, *, collect_sources=True, progress=False):
    """CPU timing includes filtering, extension, whitening, detection and feedback."""
    fs, batch = adapter.config.fs, adapter.config.batch_size
    binary, sources, duration, centroids, trusted = [], [], [], [], []
    for index, start in enumerate(range(0, len(raw), batch)):
        tick = time.perf_counter()
        spikes, values = adapter.process_batch(raw[start : start + batch], index)
        duration.append((time.perf_counter() - tick) * 1000)
        binary.append(spikes.numpy().astype(np.uint8))
        if collect_sources:
            sources.append(values.numpy())
        centroids.append(
            torch.stack([adapter.decomp.base_centr, adapter.decomp.spikes_centr]).numpy()
        )
        trusted.append(getattr(adapter, "last_trusted", torch.zeros(values.shape[1])).numpy())
        if progress and index % 500 == 0:
            print("batch", index, "seconds", round(start / fs, 2), flush=True)
    durations = np.asarray(duration)
    return dict(
        spikes=np.concatenate(binary),
        sources=np.concatenate(sources) if collect_sources else None,
        batch_ms=durations,
        centroid=np.stack(centroids),
        trusted_count=np.stack(trusted),
        median_batch_ms=float(np.median(durations)),
        p95_batch_ms=float(np.percentile(durations, 95)),
    )
