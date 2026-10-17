"""Physical directions, dormant-unit recovery and an independent Pareto fallback."""

import copy
import time

import numpy as np
import torch

from benchmarks.pareto_residual.detector import Detector
from benchmarks.pareto_residual.model import correct

from .physics import TemplateState, merge_recovery, reference_envelope, synthesize


@torch.no_grad()
def physical_sources(record):
    """Estimate morphs from past reliable Pareto events; retain inactive identities."""
    config, calibration = record["config"], record["calibration"]
    z, z_cal = record["z"], record["z_cal"]
    spikes_cal = torch.from_numpy(calibration.spikes).bool()
    state = TemplateState(z_cal, spikes_cal)
    initial = z_cal @ state.initial_filter
    peak_rms = torch.stack(
        [initial[spikes_cal[:, j], j].square().mean().sqrt() for j in range(initial.shape[1])]
    ).clamp_min(1e-6)
    gain = torch.from_numpy(calibration.spikes_centr).float().sqrt() / peak_rms
    initial *= gain
    template_calibration = copy.copy(calibration)
    template_calibration.sources = initial.numpy()
    template_calibration.spikes_centr = np.asarray(
        [float(initial[spikes_cal[:, j], j].square().mean()) for j in range(initial.shape[1])],
        dtype=np.float32,
    )
    template_calibration.base_centr = np.asarray(
        [float(initial[~spikes_cal[:, j], j].square().mean()) for j in range(initial.shape[1])],
        dtype=np.float32,
    )
    observer = Detector(calibration, config)
    candidate_detector = Detector(template_calibration, config)
    n, units = record["base"].shape
    delta, template_sources = torch.zeros(n, units), torch.zeros(n, units)
    evidence = torch.zeros(n, units, dtype=torch.bool)
    candidates = torch.zeros_like(evidence)
    scale = torch.from_numpy(calibration.sources).float().square().mean(0).sqrt().clamp_min(1e-6)
    observer_disagreements, trusted_updates = 0, 0
    durations = []
    for index, start in enumerate(range(0, n, config.batch_size)):
        tick = time.perf_counter()
        stop = min(start + config.batch_size, n)
        trim = config.ext_fact if index == 0 else 0
        block = slice(start + trim, stop)
        q = (float(record["angle"][start:stop].mean()) - record["reference_angle"]) / 40
        h = state.templates(q)
        filters = synthesize(h)
        current = z[block] @ filters * gain
        residual = z[block] @ (filters - state.initial_filter) * gain
        # Per-unit RMS displacement is bounded by 0.25 calibration source RMS.
        rms = residual.square().mean(0).sqrt().clamp_min(1e-8)
        residual *= (0.25 * scale / rms).clamp_max(1)
        delta[block], template_sources[block] = residual, current
        evidence[block] = state.evidence(z[block], h)
        candidate, _ = candidate_detector.step(current, index)
        candidates[block] = candidate
        binary, trusted = observer.step(record["base"][block], index)
        observer_disagreements += int(
            torch.count_nonzero(binary != record["baseline_spikes"][block])
        )
        # Current decisions precede current observations: no same-batch refit.
        state.observe(z[block], trusted, q)
        trusted_updates += int(trusted.sum()) if abs(q) >= 0.05 else 0
        durations.append((time.perf_counter() - tick) * 1000)
    if observer_disagreements:
        raise ValueError("Independent Pareto observer no longer reproduces the cached baseline")
    return dict(
        delta=delta,
        template_sources=template_sources,
        evidence=evidence,
        candidates=candidates,
        diagnostics=dict(
            observer_spike_disagreements=observer_disagreements,
            potential_trusted_updates=trusted_updates,
            median_physical_batch_ms=float(np.median(durations)),
            p95_physical_batch_ms=float(np.percentile(durations, 95)),
        ),
    )


@torch.no_grad()
def replay(record, models, physical, *, method):
    config, calibration = record["config"], record["calibration"]
    detector = Detector(calibration, config)
    state = detector.decomp
    cal_spike = state.spikes_centr_cal.clamp_min(1e-7)
    cal_gap = (state.spikes_centr_cal - state.base_centr_cal).clamp_min(1e-7)
    n, units = record["base"].shape
    binary = torch.zeros(n, units, dtype=torch.bool)
    last_output = torch.full((units,), -config.fs * 10, dtype=torch.long)
    last_primary = last_output.clone()
    restored, rest_samples, rest_differences, candidate_count = 0, 0, 0, 0
    times = []
    for index, start in enumerate(range(0, n, config.batch_size)):
        tick = time.perf_counter()
        stop = min(start + config.batch_size, n)
        trim = config.ext_fact if index == 0 else 0
        block = slice(start + trim, stop)
        p, f = record["base"][block], record["frozen"][block]
        centroid = torch.stack(
            [
                (state.spikes_centr.clamp_min(1e-7) / cal_spike).log(),
                ((state.spikes_centr - state.base_centr).clamp_min(1e-7) / cal_gap).log(),
            ],
            -1,
        )
        features = torch.cat([record["features"][index], centroid], -1)
        coefficient = torch.stack([m(features) for m in models]).mean(0)
        q = (float(record["angle"][start:stop].mean()) - record["reference_angle"]) / 40
        speed = float(record["features"][index, 0, 1])
        envelope, rest = reference_envelope(q, speed)
        if start < 5 * config.fs:
            envelope, rest = 0.0, True
        sources = correct(p, f, coefficient * envelope)
        if method in ("physical", "rescue"):
            sources += envelope * physical["delta"][block]
        prediction, _ = detector.step(sources, index)
        # The fallback is the complete independent Pareto prediction, rather
        # than a zero gate using the already altered detector's centroids/FIFO.
        if rest:
            prediction = record["baseline_spikes"][block].clone()
            rest_samples += (stop - start - trim) * units
        if method == "rescue" and not rest:
            candidate_count += int(physical["candidates"][block].sum())
            prediction, count = merge_recovery(
                prediction,
                physical["candidates"][block],
                physical["evidence"][block],
                start=start + trim,
                last_output=last_output,
                last_primary=last_primary,
                refractory=config.spike_min_dist,
                silence=round(0.25 * config.fs),
            )
            restored += count
        else:
            for unit in range(units):
                peaks = prediction[:, unit].nonzero().flatten()
                if len(peaks):
                    last_output[unit] = last_primary[unit] = start + trim + int(peaks[-1])
        binary[block] = prediction
        if rest:
            rest_differences += int(
                torch.count_nonzero(prediction != record["baseline_spikes"][block])
            )
        times.append((time.perf_counter() - tick) * 1000)
    return binary.numpy().astype(np.uint8), dict(
        recovered_candidate_events=restored,
        template_candidate_events=candidate_count,
        fallback_sample_unit_count=rest_samples,
        fallback_spike_disagreements=rest_differences,
        median_head_batch_ms=float(np.median(times)),
        p95_head_batch_ms=float(np.percentile(times, 95)),
    )
