"""Keep the reference detector/SV updates, replacing only the whitening operation."""

import time

import numpy as np
import torch

from .common import reference
from .data import decoder
from .model import apply_correction

reference()
from adapt_decomp import AdaptDecomp  # noqa: E402
from adapt_decomp.adaptation.ops import clip_global_delta  # noqa: E402


def frame_delta(separators, u, v, previous_u, previous_v):
    """-B (A_new-A_old) A_old^-1, evaluated via rank-r Woodbury solves."""
    eye = torch.eye(u.shape[-1], dtype=u.dtype, device=u.device)
    small = eye + previous_u.T @ previous_v
    right_v = v - previous_v @ torch.linalg.solve(small, previous_u.T @ v)
    right_old = previous_v - previous_v @ torch.linalg.solve(small, previous_u.T @ previous_v)
    return -(separators @ u) @ right_v.T + (separators @ previous_u) @ right_old.T


class CorrectedAdapter(AdaptDecomp):
    def __init__(self, record, u, v, *, coupling=True, adapt_sv=True, adapt_sd=True):
        # Reference initialization of source/centroid state, without its superseded
        # dense whitening FIFO/calibration-KL calculations.
        import copy

        self.config = copy.deepcopy(record["config"])
        self.config.adapt_sv, self.config.adapt_sd = adapt_sv, adapt_sd
        self.config.compute_loss, self.config.debug = False, False
        self.decomp = decoder(record["calibration"], self.config)
        self.data_preprocessed = True
        self.correction_u, self.correction_v = u, v
        self.previous_u, self.previous_v = torch.zeros_like(u[0]), torch.zeros_like(v[0])
        self.coupling = coupling
        self._init_exe_time()

    @torch.no_grad()
    def _whiten(self, x, batch_idx):
        index = int(batch_idx)
        u, v = self.correction_u[index], self.correction_v[index]
        if self.coupling:
            delta = frame_delta(self.decomp.sep_vectors, u, v, self.previous_u, self.previous_v)
            delta = clip_global_delta(
                delta,
                self.decomp.sep_vectors,
                self.config.safety_clip_multiplier_sv * self.config.sv_learning_rate,
                self.config.eps,
            )
            self.decomp.sep_vectors += delta
        self.previous_u, self.previous_v = u, v
        return apply_correction(x, u, v), None


@torch.no_grad()
def predict(record, model=None, *, coupling=True, adapt_sv=True, adapt_sd=True):
    features, z0 = record["features"], record["z"]
    batch, dimension = record["config"].batch_size, z0.shape[1]
    factor_started = time.perf_counter()
    if model is None:
        u = v = torch.zeros(len(features), dimension, 3)
    else:
        model.eval()
        u, v, _, _ = model(features[None])
        u, v = u[0], v[0]
    factor_seconds = time.perf_counter() - factor_started
    adapter = CorrectedAdapter(
        record, u, v, coupling=coupling, adapt_sv=adapt_sv, adapt_sd=adapt_sd
    )
    spikes = np.zeros((len(z0), adapter.decomp.sep_vectors.shape[0]), dtype=np.uint8)
    sources_array = np.zeros(spikes.shape, dtype=np.float32)
    timings = []
    for index, start in enumerate(range(0, len(z0), batch)):
        before = time.perf_counter()
        binary, sources = adapter.process_batch(z0[start : start + batch], index)
        timings.append((time.perf_counter() - before) * 1000)
        spikes[start : start + len(binary)] = binary.numpy().astype(np.uint8)
        sources_array[start : start + len(binary)] = sources.numpy()
    return (
        spikes,
        dict(
            median_batch_ms=float(np.median(timings)),
            p95_batch_ms=float(np.percentile(timings, 95)),
            total_batch_seconds=float(np.sum(timings) / 1000),
            factor_seconds=factor_seconds,
            whitening_precompute_seconds=record["metadata"]["elapsed_seconds"],
            latency_scope="cached W0*x and causal encoder factors; not live end-to-end latency",
        ),
        sources_array,
    )
