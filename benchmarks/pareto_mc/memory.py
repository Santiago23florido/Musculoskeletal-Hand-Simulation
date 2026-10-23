"""C2 record-local effective-filter memory; retrieval precedes any current insert."""

import torch


def confidence(spikes, expected, gap, calibration_gap, trusted, variance=0.0):
    rate = (spikes / expected.clamp_min(1e-7)).clamp(0, 1)
    separation = (gap / calibration_gap.clamp_min(1e-7)).clamp(0, 1)
    fraction = (trusted / spikes.clamp_min(1)).clamp(0, 1)
    return (rate * separation * fraction).pow(1 / 3) * torch.exp(-torch.as_tensor(variance))


def effective_filters(whitening, separators):
    return separators @ whitening


def project_filters(whitening, effective):
    return torch.linalg.solve(whitening.T, effective.T).T


def limited_blend(pareto, memory, beta, kappa):
    """Bound the FINAL post-QR residual, including rotations of other unit rows.

    Backtracking on the blend prevents QR from silently violating the row bound.
    Sign alignment follows each Pareto row; beta=0 returns bitwise identity.
    """
    if not 0 < kappa <= 1:
        raise ValueError("kappa must lie in (0,1]")
    if not bool(torch.any(beta != 0)):
        return pareto.clone()
    memory = memory * torch.where((pareto * memory).sum(1) < 0, -1.0, 1.0)[:, None]
    memory = torch.nn.functional.normalize(memory, dim=1)
    scale = 1.0
    for _ in range(24):
        proposal = torch.nn.functional.normalize(
            pareto * (1 - scale * beta[:, None]) + memory * (scale * beta[:, None]),
            dim=1,
        )
        proposal = torch.linalg.qr(proposal.T, mode="reduced")[0].T
        signs = torch.where((proposal * pareto).sum(1) < 0, -1.0, 1.0)
        proposal = proposal * signs[:, None]
        change = (proposal - pareto).norm(dim=1) / pareto.norm(dim=1).clamp_min(1e-7)
        if bool(torch.all(change <= kappa)):
            return proposal
        scale *= 0.5
    return pareto.clone()


class PosturalMemory:
    def __init__(self, filters, centroids, *, bins=16, h=0.06, momentum=0.95, threshold=0.8):
        if bins not in (8, 16, 32) or h <= 0 or not 0 <= momentum < 1:
            raise ValueError("Invalid memory configuration")
        self.knots = torch.linspace(0, 1, bins)
        self.h, self.momentum, self.threshold = h, momentum, threshold
        self.filters = torch.zeros(bins, *filters.shape)
        self.centroids = torch.zeros(bins, *centroids.shape)
        self.quality = torch.zeros(bins, filters.shape[0])
        self.last_batch = -1
        self.filters[0], self.centroids[0], self.quality[0] = filters.clone(), centroids.clone(), 1
        self.calibration_only = False

    def retrieve(self, state, *, batch):
        if batch <= self.last_batch:
            raise ValueError("Retrieve before inserting this batch; future memory is forbidden")
        kernel = torch.exp(-0.5 * ((self.knots - float(state)) / self.h).square())
        weights = kernel[:, None] * self.quality
        total = weights.sum(0)
        normalized = weights / total.clamp_min(1e-12)
        filters = (self.filters * normalized[:, :, None]).sum(0)
        centroids = (self.centroids * normalized[:, None, :]).sum(0)
        mass = (total / kernel.sum().clamp_min(1e-12)).clamp(0, 1)
        return filters, centroids, mass

    def insert(self, state, filters, centroids, quality, *, batch):
        if batch <= self.last_batch:
            raise ValueError("Memory batches must increase strictly")
        self.last_batch = batch
        if self.calibration_only:
            return
        index = int(torch.argmin((self.knots - float(state)).abs()))
        keep = quality >= self.threshold
        # The calibration snapshot is immutable and retains maximal confidence.
        if index == 0:
            return
        for j in keep.nonzero().flatten().tolist():
            alpha = self.momentum if self.quality[index, j] > 0 else 0.0
            sign = -1.0 if (self.filters[index, j] @ filters[j]) < 0 else 1.0
            self.filters[index, j].mul_(alpha).add_(filters[j] * (1 - alpha) * sign)
            self.centroids[index, :, j].mul_(alpha).add_(centroids[:, j] * (1 - alpha))
            self.quality[index, j] = alpha * self.quality[index, j] + (1 - alpha) * quality[j]
