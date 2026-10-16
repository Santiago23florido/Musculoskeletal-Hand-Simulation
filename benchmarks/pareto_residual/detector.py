"""Replay the reference adaptive detector on actual evolving Pareto sources."""

import copy

import torch

from benchmarks.stage_a.data import decoder

from .common import reference

reference()
from adapt_decomp import AdaptDecomp  # noqa: E402
from adapt_decomp.adaptation.ops import gate_spikes_by_iqr  # noqa: E402


class Detector(AdaptDecomp):
    def __init__(self, calibration, config):
        self.config = copy.deepcopy(config)
        self.config.debug, self.config.compute_loss = False, False
        self.decomp = decoder(calibration, self.config)

    @torch.no_grad()
    def step(self, sources, index):
        # Called with the first batch's extension padding already removed.
        spikes, peaks = self._detect_spikes(sources.detach(), len(sources))
        trusted = gate_spikes_by_iqr(
            sources.detach(),
            spikes,
            self.decomp.Q75_cal,
            self.decomp.IQR_cal,
            gate_factor=3,
            peak_power=self.config.spike_det_exp,
            use_abs_for_detection=True,
            eps=self.config.eps,
        )
        self._update_spike_det(sources.detach(), spikes, peaks, trusted, index)
        return spikes, trusted
