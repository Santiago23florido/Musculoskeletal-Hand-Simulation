"""Context heads start at the prior, are causal across batches and chunk-invariant."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from . import context_c4
from .context_c4 import BATCH, ContextLikelihood, Layout, chunk_tensors, decode_labels, predict


@pytest.fixture(autouse=True)
def short_calibration(monkeypatch):
    """Synthetic records are 4 s long: calibration ends after five batches here."""
    monkeypatch.setattr(context_c4, "KEEP", 5 * BATCH)


def synthetic(batches=40, units=3, seed=0):
    rng = np.random.default_rng(seed)
    time, unit = [], []
    for b in range(batches):
        for j in range(units):
            k = rng.integers(0, 9)
            t = np.unique(rng.integers(0, BATCH, k)) + b * BATCH
            time += t.tolist()
            unit += [j] * len(t)
    n = len(time)
    events = dict(
        time=np.array(time, dtype=np.int32),
        unit=np.array(unit, dtype=np.int16),
        primary=rng.random(n) < 0.2,
        record=dict(calibration=SimpleNamespace(spikes=np.zeros((0, units)))),
    )
    x24 = rng.normal(size=(n, 24)).astype(np.float32)
    offset = ((events["time"] % BATCH) / BATCH).astype(np.float32)
    posture = rng.uniform(0, 1, (batches, 2)).astype(np.float32)
    return events, (x24, offset, events["primary"], posture)


def active(kind, seed=0):
    torch.manual_seed(seed)
    model = ContextLikelihood(kind)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear | torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.3)
    return model.eval()


def test_layout_maps_every_candidate_once():
    events, _ = synthetic()
    layout = Layout(events, 40)
    index = layout.index[layout.index >= 0]
    assert sorted(index.tolist()) == list(range(len(events["time"])))
    j, b, _ = np.nonzero(layout.index >= 0)
    flat = layout.index[layout.index >= 0]
    assert np.array_equal(events["unit"][flat], j)
    assert np.array_equal(events["time"][flat] // BATCH, b)


@pytest.mark.parametrize("kind", ["wide", "set", "setrec", "mdiff"])
def test_heads_start_at_the_pareto_prior(kind):
    events, values = synthetic()
    logits = predict(ContextLikelihood(kind), None, events, values=values)
    prior = (events["primary"].astype(np.float32) * 2 - 1) * math.log(9)
    np.testing.assert_allclose(logits, prior, atol=1e-6)


@pytest.mark.parametrize("kind", ["set", "setrec", "mdiff"])
def test_future_batches_never_change_past_decisions(kind):
    events, values = synthetic()
    model = active(kind)
    before = predict(model, None, events, values=values)
    x24, offset, primary, posture = (v.copy() for v in values)
    later = events["time"] >= 25 * BATCH
    x24[later] += 3.0
    posture[25:] += 0.5
    after = predict(model, None, events, values=(x24, offset, primary, posture))
    np.testing.assert_array_equal(before[~later], after[~later])
    assert np.abs(before[later] - after[later]).max() > 1e-3


@pytest.mark.parametrize("kind", ["set", "setrec", "mdiff"])
def test_predictions_do_not_depend_on_the_chunk_length(kind, monkeypatch):
    events, values = synthetic()
    model = active(kind)
    reference = predict(model, None, events, values=values)
    monkeypatch.setattr(context_c4, "CHUNK", 7)
    np.testing.assert_allclose(predict(model, None, events, values=values), reference, atol=2e-5)


def test_units_are_independent():
    events, values = synthetic()
    model = active("setrec")
    reference = predict(model, None, events, values=values)
    keep = events["unit"] != 1
    reduced = {k: v[keep] if isinstance(v, np.ndarray) else v for k, v in events.items()}
    x24, offset, primary, posture = values
    logits = predict(model, None, reduced, values=(x24[keep], offset[keep], primary[keep], posture))
    np.testing.assert_allclose(logits, reference[keep], atol=2e-5)


def test_history_matters_only_for_recurrent_heads():
    events, values = synthetic()
    for kind, differs in (("set", False), ("setrec", True)):
        model = active(kind)
        full = predict(model, None, events, values=values)
        cut = predict(model, None, events, values=values, history=False)
        assert (np.abs(full - cut).max() > 1e-4) == differs


def test_a_candidate_never_sees_its_own_label():
    events, values = synthetic()
    model = active("mdiff")
    chunk = chunk_tensors(Layout(events, 40), values, 0, 40, "cpu")
    rng = np.random.default_rng(3)
    m, c, k = chunk["valid"].shape
    labels = [torch.from_numpy(rng.integers(0, 3, (m * c, k))) for _ in range(2)]
    with torch.no_grad():
        parts, _ = model.trunk(chunk, None, True)
        before = model.readout(parts, labels=tuple(labels)).view(m * c, k)
        valid = chunk["valid"].view(m * c, k)
        s, i = (int(v) for v in torch.nonzero(valid & (valid.sum(1, keepdim=True) > 2))[0])
        flipped = labels[1].clone()
        flipped[s, i] = (flipped[s, i] + 1) % 3
        after = model.readout(parts, labels=(labels[0], flipped)).view(m * c, k)
    assert torch.equal(before[s, i], after[s, i])
    others = valid[s].clone()
    others[i] = False
    assert (before[s][others] - after[s][others]).abs().max() > 1e-5


def test_label_decoding_scores_every_candidate_and_uses_labels():
    events, values = synthetic()
    model = active("mdiff")
    chunk = chunk_tensors(Layout(events, 40), values, 0, 40, "cpu")
    with torch.no_grad():
        decoded, _, carry = decode_labels(model, chunk, None)
        masked, _ = model(chunk, None)
    kept = chunk["kept"]
    assert torch.isfinite(decoded[chunk["valid"]]).all()
    assert (decoded[kept] - masked[kept]).abs().max() > 1e-4
    assert carry.shape == kept[:, -1].shape and bool((carry[~kept[:, -1]] == 0).all())


def test_budget_bounded_chunks_cover_the_record_and_keep_predictions(monkeypatch):
    events, values = synthetic()
    model = active("setrec")
    reference = predict(model, None, events, values=values)
    monkeypatch.setattr(context_c4, "BUDGET", 2_000)
    ranges = context_c4.chunks(Layout(events, 40))
    assert ranges[0][0] == 0 and ranges[-1][1] == 40
    assert all(a < b and b == c for (a, b), (c, _) in zip(ranges, ranges[1:], strict=False))
    assert len(ranges) > 40 // context_c4.CHUNK + 1
    np.testing.assert_allclose(predict(model, None, events, values=values), reference, atol=2e-5)
