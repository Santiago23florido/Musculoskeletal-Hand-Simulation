"""The streamed in-loop context head reproduces P2's chunked open-loop predictor."""

from types import SimpleNamespace

import numpy as np
import torch

from .closed_context import ContextHead, EventRecorder
from .context_c4 import BATCH, ContextLikelihood, predict
from .posture_c4 import LAG_BATCHES, posture_columns

PAD = 7  # batch 0 starts after the extension samples, as in the loop


def active(seed=0):
    torch.manual_seed(seed)
    model = ContextLikelihood("setrec")
    for module in model.modules():
        if isinstance(module, torch.nn.Linear | torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.3)
    return model.eval()


def synthetic(batches=45, units=4, seed=0, empty=(3, 17, 18)):
    """Per-batch loop events (local times) and the whole-record arrays P2 would see."""
    rng = np.random.default_rng(seed)
    q = np.cumsum(rng.normal(0, 0.05, batches))
    per_batch, time, unit, features, primary = [], [], [], [], []
    for b in range(batches):
        start = b * BATCH + (PAD if b == 0 else 0)
        local_t, local_u = [], []
        for j in range(units):
            k = 0 if b in empty else int(rng.integers(0, 9))
            t = np.unique(rng.integers(start, (b + 1) * BATCH, k))
            local_t += (t - b * BATCH - (PAD if b == 0 else 0)).tolist()
            local_u += [j] * len(t)
            time += t.tolist()
            unit += [j] * len(t)
        n = len(local_t)
        f = rng.normal(size=(n, 22)).astype(np.float32)
        p = rng.random(n) < 0.2
        features.append(f)
        primary.append(p)
        per_batch.append(
            dict(
                features=f,
                time=np.array(local_t, dtype=np.int32),
                unit=np.array(local_u, dtype=np.int16),
                primary=p,
            )
        )
    time = np.array(time, dtype=np.int32)
    f22 = np.concatenate(features)
    events = dict(
        time=time,
        unit=np.array(unit, dtype=np.int16),
        primary=np.concatenate(primary),
        record=dict(calibration=SimpleNamespace(spikes=np.zeros((0, units)))),
    )
    x24 = np.concatenate([f22, posture_columns(q, time, BATCH)], 1).astype(np.float32)
    displacement = q - q[np.maximum(np.arange(len(q)) - LAG_BATCHES, 0)]
    values = (
        x24,
        ((time % BATCH) / BATCH).astype(np.float32),
        events["primary"],
        np.column_stack([q, displacement]).astype(np.float32),
    )
    return per_batch, q, events, values


def stream(model, per_batch, q, units):
    owner = SimpleNamespace(
        config=SimpleNamespace(batch_size=BATCH, ext_fact=PAD), state_trace=[], current_index=-1
    )
    head = ContextHead(model, owner, units)
    outputs = []
    for b, event in enumerate(per_batch):
        owner.current_index = b
        owner.state_trace.append([float(q[b]), 0.0, 0.0])
        owner.last_event = event
        logits = head(torch.from_numpy(event["features"]), torch.from_numpy(event["primary"]))
        outputs.append(logits.numpy())
    return np.concatenate(outputs)


def test_streamed_head_equals_chunked_predictor():
    per_batch, q, events, values = synthetic()
    model = active()
    streamed = stream(model, per_batch, q, units=4)
    chunked = predict(model, None, events, values=values)
    np.testing.assert_allclose(streamed, chunked, atol=2e-5)
    assert np.abs(streamed).max() > 1e-3


def test_streamed_head_is_the_prior_at_initialisation():
    per_batch, q, events, _ = synthetic(seed=1)
    streamed = stream(ContextLikelihood("setrec").eval(), per_batch, q, units=4)
    prior = (events["primary"].astype(np.float32) * 2 - 1) * np.log(9)
    np.testing.assert_allclose(streamed, prior, atol=1e-6)


def test_history_reaches_later_batches_only():
    per_batch, q, _, _ = synthetic(seed=2)
    model = active(1)
    before = stream(model, per_batch, q, units=4)
    changed = [dict(e) for e in per_batch]
    changed[30] = dict(changed[30], features=changed[30]["features"] + 2.0)
    after = stream(model, changed, q, units=4)
    cut = sum(len(e["time"]) for e in per_batch[:30])
    np.testing.assert_array_equal(before[:cut], after[:cut])
    assert np.abs(before[cut:] - after[cut:]).max() > 1e-4


def test_recorder_passes_events_through():
    owner = SimpleNamespace()
    inner = SimpleNamespace(build=lambda *a, **k: dict(args=a, kwargs=k))
    recorder = EventRecorder(inner, owner)
    event = recorder.build(1, 2, speed=3)
    assert event == dict(args=(1, 2), kwargs=dict(speed=3)) and owner.last_event is event
