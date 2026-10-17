"""Causal template sufficient statistics and a small regularized inverse.

Templates live in spatially whitened, time-extended EMG coordinates. This is
a separable covariance approximation, not the original full adaptive whitener.
No post-calibration true spikes or true MUAP trajectories enter this module.
"""

import numpy as np
import torch


def spatial_whitener(emg, shrinkage=0.05):
    centered = emg - emg.mean(0)
    covariance = centered.T @ centered / len(centered)
    ridge = shrinkage * covariance.trace() / covariance.shape[0]
    values, vectors = torch.linalg.eigh(covariance)
    return (vectors * (values + ridge).clamp_min(1e-10).rsqrt()) @ vectors.T


def synthesize(templates, ridge=0.1):
    """H (H^T H + ridge I)^-1: regularized template-model inverse.

    Eigenvalues of the small SPD system are >= ridge. This ensures a unique,
    finite solve; it guarantees neither true source identifiability nor RoA.
    """
    normalized = templates / templates.norm(dim=0).clamp_min(1e-8)
    gram = normalized.T @ normalized
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    factor = torch.linalg.cholesky(gram + ridge * eye)
    return normalized @ torch.cholesky_solve(eye, factor)


def reference_envelope(q, speed):
    """Continuous attenuation near calibration posture; exact rest fallback.

    q is displacement from calibration posture divided by 40 degrees.
    speed is normalized as in the previous experiment. No recording clock or
    condition name enters the decision.
    """
    rest = abs(q) <= 0.025 and abs(speed) <= 0.1
    amplitude = float(np.clip((abs(q) - 0.025) / 0.1, 0, 1))
    return amplitude, rest


class TemplateState:
    """Anchored first-order morph fitted from previous predicted discharges.

    Minimize sum ||z_peak - H0_j - q B_j||^2 + prior ||B_j||^2.
    The sufficient statistics update AFTER emitting a batch. No observation
    from an inactive unit is required to synthesize its retained filter.
    """

    def __init__(self, z_cal, spikes, *, prior=20.0, morph_bound=0.5):
        count = spikes.sum(0).clamp_min(1)
        self.anchor = z_cal.T @ spikes.float() / count
        self.numerator = torch.zeros_like(self.anchor)
        self.denominator = torch.full((spikes.shape[1],), prior, dtype=z_cal.dtype)
        self.bound = morph_bound
        self.initial_filter = synthesize(self.anchor)
        self.threshold = []
        norm = z_cal.norm(dim=1).clamp_min(1e-8)
        unit_templates = self.anchor / self.anchor.norm(dim=0).clamp_min(1e-8)
        cosine = z_cal @ unit_templates / norm[:, None]
        for unit in range(spikes.shape[1]):
            evidence = cosine[spikes[:, unit].bool(), unit]
            self.threshold.append(max(0.05, float(torch.quantile(evidence, 0.1))))
        self.threshold = torch.tensor(self.threshold)

    def templates(self, q):
        change = q * self.numerator / self.denominator
        # A norm bound on the physical template change, not an invertibility claim.
        limit = self.bound * self.anchor.norm(dim=0).clamp_min(1e-8)
        change = change * (limit / change.norm(dim=0).clamp_min(1e-8)).clamp_max(1)
        return self.anchor + change

    def observe(self, z, trusted, q):
        if abs(q) < 0.05:
            return
        templates = self.templates(q)
        cosine = z @ (templates / templates.norm(dim=0).clamp_min(1e-8))
        cosine = cosine / z.norm(dim=1).clamp_min(1e-8)[:, None]
        accepted = trusted & (cosine >= self.threshold[None])
        count = accepted.sum(0)
        residual_sum = z.T @ accepted.float() - self.anchor * count
        self.numerator += q * residual_sum
        self.denominator += q * q * count

    def evidence(self, z, templates):
        normalized = templates / templates.norm(dim=0).clamp_min(1e-8)
        cosine = z @ normalized / z.norm(dim=1).clamp_min(1e-8)[:, None]
        # Distinct identities must win against other calibrated templates.
        maximum = cosine.max(1, keepdim=True).values
        # Ambiguous exact ties must not depend on the unit's array index.
        best = cosine == maximum
        unique = best & (best.sum(1, keepdim=True) == 1)
        return (cosine >= self.threshold[None]) & unique


def merge_recovery(
    primary, candidates, evidence, *, start, last_output, last_primary, refractory, silence
):
    """Add template-supported peaks only to genuinely silent primary units.

    Output history crosses batch boundaries. Existing primary events are never
    deleted. This cannot guarantee precision: false positive rescue is measured.
    """
    result = primary.clone()
    added = 0
    for unit in range(primary.shape[1]):
        existing = primary[:, unit].nonzero().flatten().tolist()
        latest = int(last_output[unit])
        previous_primary = int(last_primary[unit])
        for index in (candidates[:, unit] & evidence[:, unit]).nonzero().flatten().tolist():
            absolute = start + index
            before = [start + t for t in existing if t <= index]
            recent_primary = max([previous_primary] + before)
            if absolute - recent_primary < silence:
                continue
            if absolute - latest < refractory:
                continue
            if existing and min(abs(index - t) for t in existing) < refractory:
                continue
            result[index, unit] = True
            latest = absolute
            added += 1
        output_times = result[:, unit].nonzero().flatten()
        if len(output_times):
            last_output[unit] = start + int(output_times[-1])
        if existing:
            last_primary[unit] = start + existing[-1]
    return result, added
