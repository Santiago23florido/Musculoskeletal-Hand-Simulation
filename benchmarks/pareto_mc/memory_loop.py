"""C2 extensions to the frozen L1 loop, with record-local causal retrieval."""

import copy
from dataclasses import dataclass

import numpy as np
import torch

from .backbone import build
from .closed_loop import ClosedLoop
from .memory import PosturalMemory, confidence, effective_filters, limited_blend, project_filters


@dataclass(frozen=True)
class MemoryConfig:
    bins: int = 16
    h: float = 0.06
    threshold: float = 0.8
    kappa: float = 0.1
    beta_max: float = 0.0
    momentum: float = 0.95
    calibration_only: bool = True
    conservative: bool = False

    def __post_init__(self):
        if self.bins not in (8, 16, 32) or self.h not in (0.03, 0.06, 0.12):
            raise ValueError("Undeclared structural memory choice")
        if not 0.5 <= self.threshold <= 0.95 or not 0.02 <= self.kappa <= 0.3:
            raise ValueError("Invalid memory confidence or step limit")
        if not 0 <= self.beta_max <= 0.8 or not 0.8 <= self.momentum <= 0.99:
            raise ValueError("Invalid memory gain or momentum")


class MemoryEvents:
    """Install the retrieved view after C1, before C4 and all current inserts."""

    def __init__(self, owner, builder):
        self.owner, self.builder = owner, builder

    def build(self, pareto, fallback, primary, *, speed):
        owner = self.owner
        q, logvar, _ = owner.state_trace[-1]
        filters, centroids, mass = owner.memory.retrieve(q, batch=owner.current_index)
        owner.retrieved, owner.retrieved_centroids, owner.mass = filters, centroids, mass
        # No changed feature or arithmetic on the identity path, including m=0.
        if owner.memory_config.beta_max == 0 or owner.in_calibration:
            return self.builder.build(pareto, fallback, primary, speed=speed)
        owner.query_reliability = (
            float(np.exp(-0.5 * np.exp(logvar) / owner.memory_config.h**2))
            if owner.observer is not None
            else 1.0
        )
        valid = mass >= owner.memory_config.threshold
        view = (owner.current_x @ filters.T).numpy()
        # At unsupported states keep the calibration view already used at L1.
        view[:, ~valid.numpy()] = fallback[:, ~valid.numpy()]
        return self.builder.build(pareto, view, primary, speed=speed, mass=mass.numpy())


class MemoryLoop(ClosedLoop):
    def configure_memory(self, calibration, config):
        self.memory_config = config
        filters = self.fixed_filters.T.clone()
        centroids = torch.stack(
            [torch.from_numpy(calibration.base_centr), torch.from_numpy(calibration.spikes_centr)]
        ).float()
        self.memory = PosturalMemory(
            filters,
            centroids,
            bins=config.bins,
            h=config.h,
            threshold=config.threshold,
            momentum=config.momentum,
        )
        self.memory.calibration_only = config.calibration_only
        self.calibration_gap = (centroids[1] - centroids[0]).clamp_min(1e-7)
        self.expected_spikes = torch.from_numpy(calibration.spikes.mean(0)).float()
        self.expected_spikes *= self.config.batch_size
        self.builder = MemoryEvents(self, self.builder)
        self.memory_trace = []
        self.query_reliability = 0.0
        self.current_index = -1
        self.mass = torch.zeros(filters.shape[0])
        return self

    @property
    def in_calibration(self):
        return self.current_index * self.config.batch_size < 5 * self.config.fs

    def decide(self, sources, spikes, peaks, x, pad, index):
        self.current_index, self.current_x = index, x
        return super().decide(sources, spikes, peaks, x, pad, index)

    def _update_sep_vectors(self, z, sources, trusted, index):
        super()._update_sep_vectors(z, sources, trusted, index)
        cfg = self.memory_config
        pareto = self.decomp.sep_vectors
        beta = cfg.beta_max * self.mass * self.query_reliability
        beta = torch.where(self.mass >= cfg.threshold, beta, 0.0)
        if self.in_calibration:
            beta.zero_()
        if cfg.conservative:
            # The single preregistered alternative also requires locally reliable
            # decisions. It changes no saved centroids, even when returning to rest.
            gap = self.decomp.spikes_centr - self.decomp.base_centr
            quality = confidence(
                trusted.sum(0),
                self.expected_spikes,
                gap,
                self.calibration_gap,
                trusted.sum(0),
            )
            beta = beta * quality.square()
        if bool(torch.any(beta != 0)):
            projected = project_filters(self.decomp.whitening, self.retrieved)
            if not bool(torch.isfinite(projected).all()):
                raise FloatingPointError("Non-finite projected memory filter")
            self.decomp.sep_vectors = limited_blend(pareto, projected, beta, cfg.kappa)
        delta = (self.decomp.sep_vectors - pareto).norm(dim=1)
        delta /= pareto.norm(dim=1).clamp_min(1e-7)
        if bool(torch.any(delta > cfg.kappa + 1e-6)):
            raise AssertionError("Post-QR memory step exceeded the registered limit")
        self.memory_trace.append(torch.stack([self.mass, beta, delta, (beta > 0).float()]).numpy())
        # Store only the state available AFTER this batch; it is not retrievable
        # until the next batch. L2 insert() advances time but preserves bin 0 only.
        gap = self.decomp.spikes_centr - self.decomp.base_centr
        if self.feedback_mask is not None:
            trusted = trusted & self.feedback_mask
        quality = confidence(
            self.last_candidate_count,
            self.expected_spikes,
            gap,
            self.calibration_gap,
            trusted.sum(0),
            variance=float(np.exp(self.state_trace[-1][1])),
        )
        if not self.in_calibration:
            self.memory.insert(
                self.state_trace[-1][0],
                self.memory.filters[0]
                if cfg.calibration_only
                else effective_filters(self.decomp.whitening, self.decomp.sep_vectors),
                torch.stack([self.decomp.base_centr, self.decomp.spikes_centr]),
                quality,
                batch=index,
            )

    def _update_spike_det(self, sources, spikes, peaks, trusted, index):
        self.last_candidate_count = spikes.sum(0)
        return super()._update_spike_det(sources, spikes, peaks, trusted, index)


def make_memory_adapter(record, head, observer=None, *, memory_config=None, **selection):
    config = copy.deepcopy(record["config"])
    config.max_sigma_batches = 1  # Fixed-mode statistics only, as verified at L1.
    adapter = build(record["calibration"], config, cls=MemoryLoop)
    adapter.configure(record["calibration"], head, observer=observer, **selection)
    return adapter.configure_memory(record["calibration"], memory_config or MemoryConfig())
