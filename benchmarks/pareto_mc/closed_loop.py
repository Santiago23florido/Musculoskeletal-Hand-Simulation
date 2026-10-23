"""C4 is evaluated on the same evolving filters/centroids that it controls."""

import copy

import numpy as np
import torch

from .backbone import ParetoMC, build
from .events import EventFeatures, decode_block
from .observer import EMGFeatures


def calibration_template(calibration):
    sources = torch.from_numpy(calibration.sources).float()
    templates = []
    for j in range(sources.shape[1]):
        peaks = np.flatnonzero(calibration.spikes[:, j])
        peaks = peaks[(peaks >= 10) & (peaks < len(sources) - 10)]
        waves = sources[torch.as_tensor(peaks)[:, None] + torch.arange(-10, 11), j]
        waves = waves / waves.norm(dim=1, keepdim=True).clamp_min(1e-6)
        template = waves.mean(0)
        templates.append(template / template.norm().clamp_min(1e-6))
    return torch.stack(templates).numpy()


class ClosedLoop(ParetoMC):
    def configure(
        self,
        calibration,
        head,
        *,
        observer=None,
        threshold=0.5,
        strength=1.0,
        guarded_feedback=False,
        collect=True,
    ):
        self.head, self.observer = head, observer
        self.collector = collect
        self.threshold, self.strength = threshold, strength
        self.guarded_feedback = guarded_feedback
        self.events = []
        self.builder = EventFeatures(calibration, calibration_template(calibration))
        self.fixed_filters = (
            torch.from_numpy(calibration.whitening).float().T
            @ torch.from_numpy(calibration.sep_vectors).float()
        )
        self.extractor = EMGFeatures(calibration, self.config) if observer is not None else None
        self.hidden, self.previous_q = None, 0.0
        self.state_trace = []
        self.feedback_mask = None
        return self

    @torch.no_grad()
    def decide(self, sources, spikes, peaks, x, pad, index):
        q, logvar, speed = 0.0, 0.0, 0.0
        if self.observer is not None:
            features = self.extractor.step(x)
            mean, variance, self.hidden = self.observer(features[None, None], self.hidden)
            q, logvar = float(mean[0, 0]), float(variance[0, 0])
            speed = abs(q - self.previous_q) / (self.config.batch_size / self.config.fs)
            self.previous_q = q
        self.state_trace.append([q, logvar, speed])
        memory = (x @ self.fixed_filters).numpy()
        event = self.builder.build(sources.numpy(), memory, spikes.numpy(), speed=speed)
        original = spikes.clone()
        probability = np.where(event["primary"], 0.9, 0.1)
        if self.head is not None:
            prior = (event["primary"].astype(np.float32) * 2 - 1) * np.log(9)
            logits = self.head(
                torch.from_numpy(event["features"]), torch.from_numpy(event["primary"])
            ).numpy()
            probability = 1 / (1 + np.exp(-(prior + self.strength * (logits - prior))))
        identity = (
            self.head is None
            or self.strength == 0
            or (self.head.is_identity() and 0.1 < self.threshold < 0.9)
        )
        if index * self.config.batch_size < 5 * self.config.fs or identity:
            self.feedback_mask = torch.ones_like(spikes, dtype=torch.bool)
        else:
            accepted = decode_block(
                event,
                probability,
                self.threshold,
                len(sources),
                sources.shape[1],
                self.config.spike_min_dist,
            )
            spikes = torch.from_numpy(accepted)
            peaks = peaks | spikes
            self.feedback_mask = torch.ones_like(spikes, dtype=torch.bool)
            if self.guarded_feedback:
                high = torch.zeros_like(spikes, dtype=torch.bool)
                select = probability >= 0.8
                high[event["time"][select], event["unit"][select]] = True
                self.feedback_mask = original | high
        if self.collector:
            event["time"] = event["time"] + index * self.config.batch_size + pad
            self.events.append(event)
        return spikes, peaks

    def _update_spike_det(self, sources, spikes, peaks, trusted, index):
        if self.feedback_mask is not None:
            trusted = trusted & self.feedback_mask
        return super()._update_spike_det(sources, spikes, peaks, trusted, index)

    def _update_sep_vectors(self, z, sources, trusted, index):
        if self.feedback_mask is not None:
            trusted = trusted & self.feedback_mask
        return super()._update_sep_vectors(z, sources, trusted, index)

    def collected(self, initial):
        events = {
            key: np.concatenate([value[key] for value in self.events])
            for key in ("features", "time", "unit", "primary")
        }
        events["initial"] = np.asarray(initial).copy()
        return events


def make_adapter(
    record,
    head,
    observer=None,
    *,
    threshold=0.5,
    strength=1.0,
    guarded_feedback=False,
    fast_init=True,
):
    config = copy.deepcopy(record["config"])
    if fast_init:
        # Only unused whitening-error normalizers change. In fixed mode neither
        # K_cal nor sigma_K appears in the whitening/SV update or decisions.
        # One window also avoids the Windows MKL batched-LU SLASWP failure.
        config.max_sigma_batches = 1
    adapter = build(record["calibration"], config, cls=ClosedLoop)
    return adapter.configure(
        record["calibration"],
        head,
        observer=observer,
        threshold=threshold,
        strength=strength,
        guarded_feedback=guarded_feedback,
    )
