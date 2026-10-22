"""The learned decisions must actually change adaptation, with calibration-only labels."""

import copy
from types import SimpleNamespace

import numpy as np
import torch

from .closed_loop import ClosedLoop
from .events import label_events
from .models import SpikeLikelihood
from .test_backbone import fixture


def configured(adapter, model, *, guarded=False):
    result = copy.deepcopy(adapter)
    result.__class__ = ClosedLoop
    c = SimpleNamespace(
        sources=adapter.decomp.sources_calib.numpy(),
        spikes=adapter.decomp.spikes_calib.numpy(),
        spikes_centr=adapter.decomp.spikes_centr_cal.numpy(),
        whitening=adapter.decomp.whitening.numpy(),
        sep_vectors=adapter.decomp.sep_vectors.T.numpy(),
        sil=np.ones(2),
    )
    return result.configure(c, model, guarded_feedback=guarded)


def test_accepted_events_feed_back_into_centroids_and_separation_vectors():
    original, signal = fixture()
    model = SpikeLikelihood()
    with torch.no_grad():
        model.layers[-1].bias.fill_(100)
    active = configured(original, model)
    baseline = configured(original, SpikeLikelihood())
    n = original.config.batch_size
    a, _ = active.process_batch(signal[:n], 60)  # strictly after the calibration hold
    b, _ = baseline.process_batch(signal[:n], 60)
    assert a.sum() > b.sum()
    assert torch.linalg.norm(active.decomp.spikes_centr - baseline.decomp.spikes_centr) > 0
    assert torch.linalg.norm(active.decomp.sep_vectors - baseline.decomp.sep_vectors) > 0


def test_guarded_variant_does_not_train_centroids_on_weak_added_events():
    original, signal = fixture()
    model = SpikeLikelihood()
    with torch.no_grad():
        model.layers[-1].bias.fill_(1.2)  # posterior about 0.76 for non-Pareto candidates
    full = configured(original, model)
    guarded = configured(original, model, guarded=True)
    n = original.config.batch_size
    a, _ = full.process_batch(signal[:n], 60)
    b, _ = guarded.process_batch(signal[:n], 60)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.all(guarded.decomp.spikes_centr >= full.decomp.spikes_centr)
    assert torch.any(guarded.decomp.spikes_centr > full.decomp.spikes_centr)


def test_candidate_labels_use_only_calibration_lag_and_four_sample_tolerance(monkeypatch):
    import adapt_decomp.spikes

    from . import events

    truth = np.zeros((13000, 1), dtype=np.uint8)
    truth[11000, 0] = 1
    monkeypatch.setattr(events, "load_truth", lambda *args, **kwargs: truth)

    def calibration_scorer(ref, pred, **kwargs):
        assert len(ref) == 5 * 2048 and len(pred) == 5 * 2048
        return np.ones(1), None, np.array([2])

    monkeypatch.setattr(adapt_decomp.spikes, "rate_of_agreement_paired", calibration_scorer)
    candidates = dict(
        time=np.array([10998, 11002, 11003]),
        unit=np.zeros(3, dtype=int),
        initial=np.zeros((10240, 1), dtype=np.uint8),
    )
    record = dict(raw=np.empty((13000, 0)), config=SimpleNamespace(fs=2048))
    labels, references, lag = label_events(candidates, record, phase="training")
    np.testing.assert_array_equal(labels, [1, 1, 0])
    np.testing.assert_array_equal(references[0], [10998])
    assert lag[0] == 2
