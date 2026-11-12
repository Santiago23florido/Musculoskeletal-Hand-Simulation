"""P3: P2's context head (setrec24) deciding inside the real L2 loop, trained by DAgger.

P2 showed, open loop on L2's sealed final candidates, that a C4 head which attends to a
unit's competing candidates and carries a per-unit GRU over the posture trajectory
raises RoA from 92.01% (P1) to 94.06%. P3 asks whether that holds when the head's
decisions also drive the adaptation. Everything except the decision head is the
accepted L2 loop: Pareto backbone, C1 observer, calibration-only memory with L2's
selected settings, threshold/strength decoding and the calibration hold. The head
streams batch by batch with P2's chunk semantics (tested against P2's predictor).

Three DAgger rounds start from P2's selected setrec24 head. Each round's closed-loop
trajectories join P2's 420, three seeds are retrained with P2's recipe and selected
with P2's TPE. Final sub-04 decisions of all three seeds are sealed before the
original evaluator. A separately registered, single opening of sub-05 then evaluates
the frozen systems (Pareto, L2 and, if it improves on L2 on sub-04, this head).
"""

import argparse
import copy
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from functools import cache

import numpy as np
import torch
from torch.nn import functional as F

from .backbone import build, replay
from .common import (
    CONDITIONS,
    OUTPUT,
    ROOT,
    TRAIN,
    VALIDATION,
    now,
    pareto_path,
    records,
    register,
    sha256,
    stub,
    write_new,
)
from .context_c4 import (
    BATCH,
    CONTEXT_EPOCHS,
    Layout,
    chunk_tensors,
    chunks,
    load_context,
    predict,
    trajectory_values,
    unit_weights,
)
from .context_c4 import FOLDER as P2
from .experiment import checked
from .memory_experiment import SEEDS, objective_value, read
from .memory_loop import MemoryConfig, make_memory_adapter
from .observer import load_observer
from .posture_c4 import LAG_BATCHES, l2_paths, posture_columns
from .train_likelihood import labelled, load_model, quick_scores

FOLDER = OUTPUT / "P3_closed_context"
SMOKE = OUTPUT / "P3_smoke"
HELDOUT = FOLDER / "heldout"
ARM = "setrec24"
ROUNDS = (1, 2, 3)
BOOTSTRAP = 2000
HELDOUT_SUBJECT = "sub-05"
GPU_PYTHON = ROOT / ".venv-stage-b-gpu/Scripts/python.exe"


# --------------------------------------------------------------------------------------
# Streaming head inside the loop
# --------------------------------------------------------------------------------------


class EventRecorder:
    """Pass-through around the loop's event builder; keeps the batch's candidates."""

    def __init__(self, builder, owner):
        self.builder, self.owner = builder, owner

    def build(self, *args, **kwargs):
        event = self.builder.build(*args, **kwargs)
        self.owner.last_event = event
        return event


class ContextHead:
    """A P2 context head called by ClosedLoop.decide once per batch.

    The batch's times and units come from the EventRecorder, posture from the C1 trace
    that decide() has just extended. Every batch, calibration included, advances the
    GRU and becomes the next batch's keys, exactly as in P2's open-loop predictor.
    """

    def __init__(self, model, owner, units, *, keep_logits=False):
        if model.spec["kind"] != "setrec":
            raise ValueError("P3 streams the setrec context head only")
        self.model, self.owner, self.units = model.eval(), owner, units
        self.hidden, self.previous = None, None
        self.keep_logits, self.logits = keep_logits, []

    def is_identity(self):
        return False

    def _cells(self, unit, n):
        counts = np.bincount(unit, minlength=self.units) if n else np.zeros(self.units, int)
        width = int(counts.max(initial=0))
        index = np.full((self.units, max(width, 1)), -1)
        for j in range(self.units):
            idx = np.flatnonzero(unit == j)
            index[j, : len(idx)] = idx
        return index, width

    @staticmethod
    def _pad(array, width, fill=0):
        extra = width - array.shape[1]
        if extra <= 0:
            return array[:, :width]
        shape = (array.shape[0], extra) + array.shape[2:]
        return np.concatenate([array, np.full(shape, fill, dtype=array.dtype)], 1)

    @torch.no_grad()
    def __call__(self, features, primary):
        owner = self.owner
        event, index, trace = owner.last_event, owner.current_index, owner.state_trace
        if len(trace) != index + 1 or owner.config.batch_size != BATCH:
            raise AssertionError("Head called out of step with the loop")
        x22 = np.asarray(event["features"], dtype=np.float32)
        if x22.shape != tuple(features.shape):
            raise AssertionError("Head features differ from the recorded event")
        q = float(trace[-1][0])
        posture = np.array([q, q - float(trace[max(index - LAG_BATCHES, 0)][0])], np.float32)
        n = len(x22)
        pad = owner.config.ext_fact if index == 0 else 0
        time = np.asarray(event["time"], dtype=np.int64) + index * BATCH + pad
        if n and (time.min() < index * BATCH or time.max() >= (index + 1) * BATCH):
            raise AssertionError("Candidate outside its batch")
        if n:
            x24 = np.concatenate([x22, np.broadcast_to(posture, (n, 2))], 1).astype(np.float32)
            offset = ((time % BATCH) / BATCH).astype(np.float32)
            prim = np.asarray(event["primary"], dtype=bool)
        else:  # masked placeholders, as chunk_tensors' clipped padding
            x24, offset, prim = np.zeros((1, 24), np.float32), np.zeros(1, np.float32), [False]
            prim = np.asarray(prim)
        cells, width = self._cells(np.asarray(event["unit"]), n)
        previous = self.previous
        k = max(width, previous["width"] if previous else 0, 1)
        cells = self._pad(cells, k, -1)
        valid, safe = cells >= 0, cells.clip(0)
        x = x24[safe] * valid[..., None]
        off = offset[safe]
        if previous is None:
            px = np.zeros((self.units, k, 24), np.float32)
            poff = np.zeros((self.units, k), np.float32)
            pvalid = np.zeros((self.units, k), bool)
        else:
            px = self._pad(previous["x"], k)
            poff = self._pad(previous["offset"], k)
            pvalid = self._pad(previous["valid"], k, False)
        t = np.rint(off * BATCH).astype(np.int32)
        pt = np.rint(poff * BATCH).astype(np.int32) - BATCH
        arrays = dict(
            x=x,
            offset=off * valid,
            primary=prim[safe] & valid,
            valid=valid,
            px=px * pvalid[..., None],
            poffset=poff * pvalid,
            time=t,
            keys_time=np.concatenate([pt, t], 1),
            keys_valid=np.concatenate([pvalid, valid], 1),
            posture=np.broadcast_to(posture, (self.units, 2)),
        )
        chunk = {
            name: torch.from_numpy(np.ascontiguousarray(value[:, None]))
            for name, value in arrays.items()
        }
        out, self.hidden = self.model(chunk, self.hidden)
        out = out[:, 0].numpy()
        logits = np.zeros(n, dtype=np.float32)
        if n:
            logits[cells[valid]] = out[valid]
        self.previous = dict(
            x=x[:, : max(width, 1)],
            offset=off[:, : max(width, 1)],
            valid=valid[:, : max(width, 1)],
            width=width,
        )
        if self.keep_logits:
            self.logits.append(logits.copy())
        return torch.from_numpy(logits)


def make_context_adapter(record, model, *, settings, threshold, strength, keep_logits=False):
    """The accepted L2 loop (make_memory_adapter) with the context head installed."""
    adapter = make_memory_adapter(
        record,
        None,
        load_observer(),
        memory_config=MemoryConfig(**settings),
        threshold=threshold,
        strength=strength,
    )
    adapter.builder = EventRecorder(adapter.builder, adapter)
    adapter.head = ContextHead(
        model, adapter, record["calibration"].spikes.shape[1], keep_logits=keep_logits
    )
    return adapter


def make_l2_adapter(record, checkpoint, *, settings, threshold, strength):
    """The L2 loop exactly as its final validation built it."""
    return make_memory_adapter(
        record,
        load_model(checkpoint),
        load_observer(),
        memory_config=MemoryConfig(**settings),
        threshold=threshold,
        strength=strength,
    )


def l2_settings():
    retained = read(OUTPUT / "L2/retained.json")
    if not retained["passed"]:
        raise ValueError("L2 must be the retained level")
    return retained["settings"]


def l2_heads():
    retained = read(OUTPUT / "L2/retained.json")
    parameters = retained["selection"]["selection"]["parameters"]
    heads = {}
    for seed in SEEDS:
        path = OUTPUT / f"L2/standard/round3/seed{seed}/model.pt"
        if sha256(path) != read(path.with_name("complete.json"))["checkpoint_sha256"]:
            raise ValueError("L2 round-3 checkpoint changed")
        heads[seed] = dict(
            kind="l2",
            checkpoint=path.relative_to(OUTPUT).as_posix(),
            checkpoint_sha256=sha256(path),
            parameters=parameters,
        )
    return heads


def context_head_spec(path, parameters):
    return dict(
        kind="context",
        checkpoint=path.relative_to(OUTPUT).as_posix(),
        checkpoint_sha256=sha256(path),
        parameters=dict(parameters),
    )


def p2_head():
    selected = read(P2 / ARM / "selected.json")["selection"]
    path = P2 / ARM / f"seed{selected['selected_seed']}/model.pt"
    if sha256(path) != read(path.with_name("complete.json"))["checkpoint_sha256"]:
        raise ValueError("P2 setrec24 checkpoint changed")
    return context_head_spec(path, selected["parameters"])


# --------------------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------------------


@contextmanager
def heldout_guard():
    """The original loaders, with the subject guard opened for sub-05 only, phase 'test',
    and only once the held-out protocol is registered."""
    from . import data, events

    original = data.guard

    def guard(subject, phase):
        if phase == "test":
            if subject != HELDOUT_SUBJECT or not (HELDOUT / "protocol.json").exists():
                raise ValueError("Held-out access requires the registered P3 protocol")
            return
        original(subject, phase)

    data.guard, events.guard = guard, guard
    try:
        yield
    finally:
        data.guard, events.guard = original, original


def phase_of(case):
    if case[0] == HELDOUT_SUBJECT:
        return "test"
    return "training" if case[0] in TRAIN else "validation"


def run_system(record, spec, *, keep_logits=False):
    """Replay one record with a frozen system; returns the prediction and the adapter."""
    settings = l2_settings()
    if spec["kind"] == "pareto":
        config = copy.deepcopy(record["config"])
        adapter = build(record["calibration"], config)
        return replay(record["raw"], adapter, collect_sources=False), adapter
    checkpoint = OUTPUT / spec["checkpoint"]
    if sha256(checkpoint) != spec["checkpoint_sha256"]:
        raise ValueError("Classifier checkpoint changed")
    if spec["kind"] == "l2":
        adapter = make_l2_adapter(record, checkpoint, settings=settings, **spec["parameters"])
    else:
        adapter = make_context_adapter(
            record,
            load_context(checkpoint),
            settings=settings,
            keep_logits=keep_logits,
            **spec["parameters"],
        )
    adapter.collector = True
    prediction = replay(record["raw"], adapter, collect_sources=False)
    prediction["state"] = np.asarray(adapter.state_trace)
    prediction["memory"] = np.asarray(adapter.memory_trace)
    return prediction, adapter


def rollout_task(arguments):
    folder, case, spec = arguments
    torch.set_num_threads(4)
    name = stub(*case)
    metadata_path = folder / "metadata" / f"{name}.json"
    if metadata_path.exists():
        metadata = read(metadata_path)
        if metadata["spec"] != spec:
            raise ValueError("Cannot reuse a rollout made with a different system")
        for key, digest in metadata["files"].items():
            if sha256(folder / key) != digest:
                raise ValueError("Completed rollout changed")
        return name
    destination = folder / "predictions" / f"{name}.npz"
    if destination.exists():
        raise ValueError("Orphaned prediction: preserve this run and use a fresh directory")
    with heldout_guard():
        from .data import load_record

        record = load_record(*case, phase=phase_of(case))
        if spec["kind"] == "published":
            from adapt_decomp import AdaptationResult

            spikes = AdaptationResult.load(pareto_path(*case)).spikes.numpy().astype(np.uint8)
            prediction, adapter = dict(spikes=spikes), None
        else:
            prediction, adapter = run_system(record, spec)
    prediction.pop("sources", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **prediction)
    files = [destination]
    if adapter is not None and spec["kind"] != "pareto":
        events = adapter.collected(prediction["spikes"][: 5 * record["config"].fs])
        event_path = folder / "events" / f"{name}.npz"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(event_path, **events)
        files.append(event_path)
    write_new(
        metadata_path,
        dict(
            predicted_at_utc=now(),
            record=list(case),
            spec=spec,
            labels_opened=False,
            raw_sha256=record["raw_sha256"],
            calibration_sha256=record["calibration_sha256"],
            files={path.relative_to(folder).as_posix(): sha256(path) for path in files},
        ),
    )
    return name


def rollout(folder, cases, spec, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(rollout_task, (folder, case, spec)) for case in cases]
        for future in as_completed(futures):
            print("P3 rollout", folder.relative_to(OUTPUT).as_posix(), future.result(), flush=True)
    files = {}
    for case in cases:
        metadata = read(folder / "metadata" / f"{stub(*case)}.json")
        for key, digest in metadata["files"].items():
            if sha256(folder / key) != digest:
                raise ValueError("Prediction changed before seal")
            files[key] = digest
    seal_path = folder / "sealed.json"
    if not seal_path.exists():
        write_new(
            seal_path,
            dict(sealed_at_utc=now(), files=files, cases=cases, no_labels_during_rollout=True),
        )
    elif read(seal_path)["files"] != files:
        raise ValueError("Existing seal disagrees with completed rollouts")


# --------------------------------------------------------------------------------------
# Training data: P2's 420 trajectories plus P3's closed-loop rounds
# --------------------------------------------------------------------------------------


@cache
def sealed_files(folder):
    return read(folder / "sealed.json")["files"]


def values_for(path, events):
    """P2's trajectory values; for P3 rollouts posture comes from the loop's own C1 trace."""
    if FOLDER not in path.parents:
        return trajectory_values(path, events)
    rollout_folder = path.parent.parent
    prediction = rollout_folder / "predictions" / path.name
    seal = sealed_files(rollout_folder)
    for file in (prediction, path):
        if sha256(file) != seal[file.relative_to(rollout_folder).as_posix()]:
            raise ValueError("P3 trajectory changed after its seal")
    with np.load(prediction) as archive:
        state = archive["state"]
    q, speed = state[:, 0], state[:, 2]
    time = np.asarray(events["time"])
    expected = np.clip(speed[time // BATCH], -6, 6).astype(np.float32)
    if np.max(np.abs(events["features"][:, 21] - expected), initial=0.0) > 0:
        raise AssertionError(f"Candidate/batch misalignment in {path}")
    x24 = np.concatenate([events["features"], posture_columns(q, time, BATCH)], 1)
    displacement = q - q[np.maximum(np.arange(len(q)) - LAG_BATCHES, 0)]
    posture = np.column_stack([q, displacement]).astype(np.float32)
    offset = ((time % BATCH) / BATCH).astype(np.float32)
    return np.ascontiguousarray(x24, np.float32), offset, events["primary"].astype(bool), posture


def round_paths(index, phase):
    subjects = TRAIN if phase == "training" else VALIDATION
    folder = FOLDER / f"round{index}/rollout/events"
    return [(folder / f"{stub(*case)}.npz", case) for case in records(subjects)]


def training_paths(index):
    training = l2_paths("training")
    for k in range(1, index + 1):
        training += round_paths(k, "training")
    return training, round_paths(index, "validation")


def validation_bce(model, paths, validation, device):
    values = []
    for (path, _), events in zip(paths, validation, strict=True):
        logits = predict(model, path, events, device, values=values_for(path, events))
        keep = events["keep"]
        loss = np.logaddexp(0, logits[keep]) - logits[keep] * events["target"][keep]
        m = events["record"]["calibration"].spikes.shape[1]
        values.extend(float(loss[events["unit"][keep] == j].mean()) for j in range(m))
    return float(np.mean(values))


def chunk_epoch(model, optimizer, values, events, keep, weights, target):
    """context_c4.chunk_epoch for the setrec head, with explicit trajectory values."""
    layout = Layout(events, len(values[3]))
    losses, hidden = [], None
    for a, b in chunks(layout):
        chunk = chunk_tensors(layout, values, a, b, "cuda")
        for name, mask in (("x", "valid"), ("px", None)):
            noise = torch.randn_like(chunk[name]) * 0.02
            if mask is not None:
                noise = noise * chunk[mask][..., None]
            chunk[name] = chunk[name] + noise
        safe, valid = chunk["index"].clip(0), chunk["valid_np"]
        kept = torch.from_numpy(keep[safe] & valid).cuda()
        if not bool(kept.any()):
            with torch.no_grad():
                _, hidden = model(chunk, hidden)
            continue
        goal = torch.from_numpy(target[safe]).cuda()
        w = torch.from_numpy(weights[safe]).cuda()
        logits, hidden = model(chunk, hidden)
        loss = (F.binary_cross_entropy_with_logits(logits, goal, reduction="none") * w)[kept]
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        hidden = hidden.detach()
        losses.append(float(loss.detach()))
    return losses


def fit(index, seed):
    """context_c4.fit for setrec24 on P2's data plus P3 rounds 1..index."""
    from .context_c4 import ContextLikelihood

    folder = FOLDER / f"round{index}/seed{seed}"
    if (folder / "complete.json").exists():
        if sha256(folder / "model.pt") != read(folder / "complete.json")["checkpoint_sha256"]:
            raise ValueError("Completed P3 classifier changed")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Use the authorized CUDA training environment")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    model = ContextLikelihood("setrec").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    training, validation_paths = training_paths(index)
    validation = [labelled(path, case, "validation") for path, case in validation_paths]
    rng = np.random.default_rng(seed)
    history, best = [], float("inf")
    folder.mkdir(parents=True, exist_ok=True)
    for epoch in range(CONTEXT_EPOCHS):
        losses = []
        model.train()
        for order in rng.permutation(len(training)):
            path, case = training[order]
            events = labelled(path, case, "training")
            keep = np.asarray(events["keep"])
            weights = unit_weights(events, keep)
            target = np.asarray(events["target"], dtype=np.float32)
            values = values_for(path, events)
            losses += chunk_epoch(model, optimizer, values, events, keep, weights, target)
        model.eval()
        value = validation_bce(model, validation_paths, validation, "cuda")
        history.append(dict(epoch=epoch, train_bce=float(np.mean(losses)), validation_bce=value))
        if value < best:
            best = value
            torch.save(
                dict(model=model.cpu().state_dict(), spec=model.spec, epoch=epoch),
                folder / "model.pt",
            )
            model.cuda()
        print("P3 C4", index, seed, history[-1], now(), flush=True)
    write_new(folder / "history.json", history)
    write_new(
        folder / "complete.json",
        dict(
            completed_at_utc=now(),
            checkpoint_sha256=sha256(folder / "model.pt"),
            validation_bce=best,
            training_trajectories=len(training),
            epochs=CONTEXT_EPOCHS,
            seed=seed,
            gpu=torch.cuda.get_device_name(),
        ),
    )


def select(index):
    """context_c4.select on this round's closed-loop sub-04 candidates."""
    import optuna

    folder = FOLDER / f"round{index}"
    if (folder / "selected.json").exists():
        return read(folder / "selected.json")
    _, validation_paths = training_paths(index)
    validation = [labelled(path, case, "validation") for path, case in validation_paths]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = [folder / f"seed{seed}/model.pt" for seed in SEEDS]
    models = [load_context(path) for path in paths]
    logits = [
        [
            predict(model, path, events, device, values=values_for(path, events))
            for (path, _), events in zip(validation_paths, validation, strict=True)
        ]
        for model in models
    ]
    baseline_precision = read(OUTPUT / "L0/validation/summary.json")["precision_pct"]

    def measure(seed, threshold, strength):
        metrics = quick_scores(validation, logits[seed], threshold, strength)
        return objective_value(metrics, baseline_precision), metrics

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2026),
        storage=f"sqlite:///{(folder / 'tpe.sqlite3').as_posix()}",
        study_name="closed_context_C4",
        load_if_exists=True,
    )

    def evaluate(trial):
        threshold = trial.suggest_float("threshold", 0.25, 0.85)
        strength = trial.suggest_float("strength", 0.0, 1.0)
        value, metrics = measure(0, threshold, strength)
        trial.set_user_attr("metrics", metrics)
        return value

    if not study.trials:
        study.enqueue_trial(dict(threshold=0.5, strength=0.0))
    study.optimize(evaluate, n_trials=max(0, 40 - len(study.trials)))
    top = sorted(
        (t for t in study.trials if t.value is not None), key=lambda t: t.value, reverse=True
    )[:3]
    repeats = []
    for trial in top:
        scores = [measure(seed, **trial.params) for seed in SEEDS]
        seed = int(np.argsort([value for value, _ in scores])[1])
        repeats.append(
            dict(
                trial=trial.number,
                parameters=trial.params,
                seed_metrics=[metrics for _, metrics in scores],
                median_objective=float(np.median([value for value, _ in scores])),
                selected_seed=seed,
            )
        )
    winner = max(repeats, key=lambda row: row["median_objective"])
    result = dict(
        selected_at_utc=now(),
        repeats=repeats,
        selection=winner,
        head=context_head_spec(paths[winner["selected_seed"]], winner["parameters"]),
        scope="this round's closed-loop sub-04 candidates; final comparison uses new rollouts",
    )
    write_new(folder / "selected.json", result)
    return result


def round_head(index):
    return p2_head() if index == 1 else read(FOLDER / f"round{index - 1}/selected.json")["head"]


def final_heads():
    selected = read(FOLDER / "round3/selected.json")
    parameters = selected["selection"]["parameters"]
    return {
        seed: context_head_spec(FOLDER / f"round3/seed{seed}/model.pt", parameters)
        for seed in SEEDS
    }


# --------------------------------------------------------------------------------------
# Scoring, analysis and decisions
# --------------------------------------------------------------------------------------


def score_task(arguments):
    """Original evaluator plus P1's posture strata for one sealed decision file."""
    from .evaluate import score_record

    folder, case = arguments
    torch.set_num_threads(1)
    path = folder / "scores" / f"{stub(*case)}.json"
    if path.exists():
        return
    phase = phase_of(case)
    with heldout_guard():
        from .data import load_angle, load_truth
        from .events import descriptor

        record = descriptor(*case, phase=phase)
        with np.load(folder / "predictions" / f"{stub(*case)}.npz") as archive:
            binary = archive["spikes"]
        result = score_record(record, dict(spikes=binary), phase=phase)
        truth, angle = load_truth(record, phase=phase), load_angle(record, phase=phase)
    strata = strata_for(truth, angle, binary)
    for row in result["rows"]:
        row["strata"] = {k: v[row["unit"]] for k, v in strata.items()}
    write_new(path, result)


def strata_for(truth, angle, binary):
    """posture_c4.posture_strata with the labels passed in (any guarded phase)."""
    from adapt_decomp.spikes import rate_of_agreement_paired

    from benchmarks.robust_mvdr.phases import masked

    after = np.arange(len(angle)) >= 5 * 2048
    result = {}
    for label, mask in (
        ("near_reference", (angle < 5) & after),
        ("far_20deg", (angle >= 20) & after),
    ):
        if mask.sum() < 2048:
            result[label] = [None] * truth.shape[1]
            continue
        roa = rate_of_agreement_paired(
            masked(truth, mask), masked(binary, mask), fs=2048, tol_spike_ms=2
        )[0]
        result[label] = [float(v) for v in roa]
    return result


def score_folders(folders, cases, workers):
    for folder in folders:
        for key, digest in read(folder / "sealed.json")["files"].items():
            if sha256(folder / key) != digest:
                raise ValueError(f"Sealed decision changed in {folder}")
    tasks = [(folder, case) for folder in folders for case in cases]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(score_task, task) for task in tasks]):
            future.result()


def load_rows(scores, cases):
    """Evaluator rows of one decision set, keyed by (condition, SNR, unit)."""
    rows = {}
    for case in cases:
        for row in read(scores / f"{stub(*case)}.json")["rows"]:
            rows[(row["condition"], row["snr_db"], row["unit"])] = row
    return rows


def bootstrap(rows, clusters, table):
    """Seed-mean contrasts with P2's paired cluster bootstrap (2000 draws, seed 2026)."""
    from .l2_ablation import combine, metrics

    names = sorted(clusters)
    labels = list(rows)

    def summary(selected):
        values = {}
        for label in labels:
            chosen = [rows[label][k] for k in selected]
            value = metrics(chosen)
            for stratum in ("near_reference", "far_20deg"):
                found = [
                    r["strata"][stratum]
                    for r in chosen
                    if "strata" in r and r["strata"][stratum] is not None
                ]
                value[stratum] = float(np.mean(found) * 100) if found else None
            values[label] = value
        return values, {name: combine(w, values) for name, w in table.items()}

    keys = sorted(rows[labels[0]])
    point_values, point = summary(keys)
    rng = np.random.default_rng(2026)
    draws = defaultdict(lambda: defaultdict(list))
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, len(names), len(names))
        _, contrast = summary([k for i in pick for k in clusters[names[i]]])
        for name, effect in contrast.items():
            for metric, value in effect.items():
                draws[name][metric].append(np.nan if value is None else value)
    effects = {
        name: {
            metric: dict(
                estimate=value,
                ci95=(
                    None
                    if value is None
                    else [float(v) for v in np.nanpercentile(draws[name][metric], [2.5, 97.5])]
                ),
            )
            for metric, value in effect.items()
        }
        for name, effect in point.items()
    }
    return point_values, effects


def clusters_for(cases, phase):
    clusters = defaultdict(list)
    with heldout_guard():
        from .events import descriptor

        for case in cases:
            calibration = descriptor(*case, phase=phase)["calibration"]
            for unit, index in enumerate(calibration.gt_matched_indices):
                clusters[(case[1], int(index))].append((case[1], case[2], unit))
    return clusters


def mean_of(name, sign):
    return {f"{name}_seed{s}": sign / len(SEEDS) for s in SEEDS}


def analyse(workers):
    path = FOLDER / "analysis.json"
    if path.exists():
        return read(path)
    cases = records(VALIDATION)
    folders = {f"cl_seed{s}": FOLDER / f"final_validation/seed{s}" for s in SEEDS}
    score_folders(list(folders.values()), cases, workers)
    scores = {label: folder / "scores" for label, folder in folders.items()}
    scores.update(
        {f"l2_seed{s}": OUTPUT / f"L2_ablation/arms/l2_seed{s}_full/scores" for s in SEEDS}
    )
    scores.update({f"p2_seed{s}": P2 / f"scores/{ARM}/seed{s}" for s in SEEDS})
    scores["pareto"] = OUTPUT / "L0/validation/scores"
    rows = {label: load_rows(folder, cases) for label, folder in scores.items()}
    table = {
        "closed_vs_l2": {**mean_of("cl", 1), **mean_of("l2", -1)},
        "closed_vs_open_p2": {**mean_of("cl", 1), **mean_of("p2", -1)},
        "closed_vs_pareto": {**mean_of("cl", 1), "pareto": -1.0},
        "l2_vs_pareto": {**mean_of("l2", 1), "pareto": -1.0},
    }
    point_values, effects = bootstrap(rows, clusters_for(cases, "validation"), table)

    def above(name):
        return effects[name]["global_roa_pct"]["ci95"][0] > 0

    def below(name):
        return effects[name]["global_roa_pct"]["ci95"][1] < 0

    decisions = dict(
        improves_over_l2=above("closed_vs_l2"),
        closed_below_open_loop=below("closed_vs_open_p2"),
        closed_above_open_loop=above("closed_vs_open_p2"),
        improves_over_pareto=above("closed_vs_pareto"),
        heldout_eligible=above("closed_vs_l2"),
    )
    result = dict(
        analysed_at_utc=now(),
        instances=len(rows["pareto"]),
        arms=point_values,
        effects=effects,
        rounds={
            k: dict(
                selected=read(FOLDER / f"round{k}/selected.json")["selection"],
                validation_bce=[
                    read(FOLDER / f"round{k}/seed{s}/complete.json")["validation_bce"]
                    for s in SEEDS
                ],
            )
            for k in ROUNDS
        },
        decisions=decisions,
    )
    write_new(path, result)
    return result


# --------------------------------------------------------------------------------------
# Latency (isolated, one record at a time) and the single sub-05 opening
# --------------------------------------------------------------------------------------


def latency():
    """Isolated CPU timing per 100 ms batch: one record at a time, four Torch threads."""
    path = FOLDER / "latency.json"
    if path.exists():
        return read(path)
    from .data import load_record

    torch.set_num_threads(4)
    selected = read(FOLDER / "round3/selected.json")["head"]
    retained = read(OUTPUT / "L2/retained.json")["selection"]
    l2 = dict(
        kind="l2",
        checkpoint=retained["checkpoint"],
        checkpoint_sha256=retained["checkpoint_sha256"],
        parameters=retained["selection"]["parameters"],
    )
    systems = dict(pareto=dict(kind="pareto"), l2=l2, context=selected)
    cases = [("sub-04", condition, 20) for condition in CONDITIONS]
    timings = defaultdict(list)
    for case in cases:
        record = load_record(*case, phase="validation")
        for name, spec in systems.items():
            prediction, _ = run_system(record, spec)
            timings[name].append(prediction["batch_ms"])
            print("latency", name, stub(*case), round(float(np.median(timings[name][-1])), 2))
    summary = {}
    for name, parts in timings.items():
        values = np.concatenate(parts)
        summary[name] = dict(
            batches=len(values),
            median_ms=float(np.median(values)),
            p95_ms=float(np.percentile(values, 95)),
            mean_ms=float(np.mean(values)),
            max_ms=float(np.max(values)),
        )
    result = dict(
        measured_at_utc=now(),
        scope="native Windows CPU, one record at a time, no concurrent work, four Torch "
        "threads; raw filtering through closed-loop update; 204-sample (99.6 ms) batches",
        records=[stub(*case) for case in cases],
        target_ms=22,
        systems=summary,
        cpu=os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
    )
    write_new(path, result)
    return result


def heldout_arms(eligible):
    arms = {"pareto_published": dict(kind="published"), "pareto_rerun": dict(kind="pareto")}
    arms.update({f"l2_seed{s}": spec for s, spec in l2_heads().items()})
    if eligible:
        arms.update({f"cl_seed{s}": spec for s, spec in final_heads().items()})
    return arms


def register_heldout():
    analysis = read(FOLDER / "analysis.json")
    eligible = analysis["decisions"]["heldout_eligible"]
    if (HELDOUT / "protocol.json").exists():
        return read(HELDOUT / "protocol.json")
    register(
        HELDOUT,
        dict(
            purpose="Single opening of the reserved subject sub-05 with frozen systems",
            subject=HELDOUT_SUBJECT,
            cases=records((HELDOUT_SUBJECT,)),
            arms=heldout_arms(eligible),
            eligibility=dict(
                rule="the P3 head enters only if it improved on L2 on sub-04 (CI above zero)",
                eligible=eligible,
                sub04_effect=analysis["effects"]["closed_vs_l2"]["global_roa_pct"],
            ),
            frozen="every checkpoint, threshold, strength and memory setting fixed on sub-01..04",
            evaluator="original score_record; P1 posture strata; labels read only after seal",
            statistics="seed-mean contrasts, paired cluster bootstrap 2000 draws seed 2026, "
            "clusters condition x ground-truth unit",
            contrasts=[
                "l2_vs_pareto (L2 seed mean - Pareto rerun)",
                "l2_accepted_vs_pareto (L2 seed 2, the accepted selection - Pareto rerun)",
                "closed_vs_l2 and closed_vs_pareto if eligible",
            ],
            rerun_check="Pareto rerun compared sample by sample with the published run",
            once="sub-05 is opened once; this protocol is never re-registered or rerun with "
            "other systems; any later change is reported as post hoc",
        ),
    )
    return read(HELDOUT / "protocol.json")


def heldout(workers):
    path = HELDOUT / "analysis.json"
    if path.exists():
        return read(path)
    protocol = checked(HELDOUT)
    opened = HELDOUT / "opened.json"
    if not opened.exists():
        write_new(opened, dict(opened_at_utc=now(), arms=sorted(protocol["spec"]["arms"])))
    cases = [tuple(case) for case in protocol["spec"]["cases"]]
    arms = protocol["spec"]["arms"]
    for name, spec in arms.items():
        rollout(HELDOUT / "arms" / name, cases, spec, workers)
    folders = [HELDOUT / "arms" / name for name in arms]
    rerun_identity = {}
    for case in cases:
        with np.load(HELDOUT / f"arms/pareto_rerun/predictions/{stub(*case)}.npz") as a:
            rerun = a["spikes"]
        with np.load(HELDOUT / f"arms/pareto_published/predictions/{stub(*case)}.npz") as b:
            published = b["spikes"]
        n = min(len(rerun), len(published))
        rerun_identity[stub(*case)] = float(np.mean(rerun[:n] == published[:n]))
    score_folders(folders, cases, workers)
    rows = {name: load_rows(HELDOUT / "arms" / name / "scores", cases) for name in arms}
    rows["pareto"] = rows.pop("pareto_rerun")
    table = {
        "l2_vs_pareto": {**mean_of("l2", 1), "pareto": -1.0},
        "l2_accepted_vs_pareto": {"l2_seed2": 1.0, "pareto": -1.0},
        "published_vs_rerun": {"pareto_published": 1.0, "pareto": -1.0},
    }
    if any(name.startswith("cl_") for name in arms):
        table["closed_vs_l2"] = {**mean_of("cl", 1), **mean_of("l2", -1)}
        table["closed_vs_pareto"] = {**mean_of("cl", 1), "pareto": -1.0}
    point_values, effects = bootstrap(rows, clusters_for(cases, "test"), table)
    result = dict(
        analysed_at_utc=now(),
        subject=HELDOUT_SUBJECT,
        instances=len(rows["pareto"]),
        arms=point_values,
        effects=effects,
        pareto_rerun_sample_identity=rerun_identity,
    )
    write_new(path, result)
    return result


# --------------------------------------------------------------------------------------
# Registration, smoke test and the automatic pipeline
# --------------------------------------------------------------------------------------


def smoke(case=("sub-04", "triangular-ramp5s", 20)):
    """Unregistered check on one real record: the streamed in-loop logits equal P2's
    chunked predictor on the very candidates the loop produced."""
    from .data import load_record

    torch.set_num_threads(4)
    record = load_record(*case, phase="validation")
    spec = p2_head()
    prediction, adapter = run_system(record, spec, keep_logits=True)
    events = adapter.collected(prediction["spikes"][: 5 * record["config"].fs])
    events["record"] = dict(calibration=record["calibration"])
    streamed = np.concatenate(adapter.head.logits)
    state = prediction["state"]
    q = state[:, 0]
    time = events["time"]
    x24 = np.concatenate([events["features"], posture_columns(q, time, BATCH)], 1)
    displacement = q - q[np.maximum(np.arange(len(q)) - LAG_BATCHES, 0)]
    values = (
        np.ascontiguousarray(x24, np.float32),
        ((time % BATCH) / BATCH).astype(np.float32),
        events["primary"].astype(bool),
        np.column_stack([q, displacement]).astype(np.float32),
    )
    chunked = predict(load_context(OUTPUT / spec["checkpoint"]), None, events, values=values)
    difference = float(np.max(np.abs(streamed - chunked)))
    result = dict(
        checked_at_utc=now(),
        record=list(case),
        candidates=len(streamed),
        max_abs_logit_difference=difference,
        median_batch_ms=float(np.median(prediction["batch_ms"])),
        p95_batch_ms=float(np.percentile(prediction["batch_ms"], 95)),
        passed=difference < 1e-4,
        labels_opened=False,
    )
    SMOKE.mkdir(parents=True, exist_ok=True)
    (SMOKE / f"smoke_{stub(*case)}.json").write_text(json.dumps(result, indent=2), "utf-8")
    print(result, flush=True)
    return result


def register_protocol():
    if (FOLDER / "protocol.json").exists():
        return read(FOLDER / "protocol.json")
    smoke_files = list(SMOKE.glob("smoke_*.json"))
    if not smoke_files or not all(read(p)["passed"] for p in smoke_files):
        raise ValueError("Run the streaming smoke check before registration")
    register(
        FOLDER,
        dict(
            purpose="Does P2's open-loop context-head gain hold when its decisions drive the "
            "adaptation? setrec24 replaces the L2 C4 head inside the accepted L2 loop",
            loop="L2: Pareto backbone, C1 observer, calibration-only memory with L2's selected "
            "settings, threshold/strength decoding, Pareto held during calibration",
            head="P2 setrec24 streamed batch by batch with P2's chunk semantics; in-loop logits "
            "equal P2's predictor on the loop's candidates (smoke check, max |diff| < 1e-4)",
            start="round 1 rolls out P2's selected setrec24 (seed 2, its TPE parameters)",
            dagger="three rounds; each adds 60 closed-loop training trajectories to P2's 420; "
            "three seeds retrained from scratch with P2's recipe (AdamW 1e-3, wd 0.01, three "
            "epochs, unit-balanced BCE, feature noise 0.02, deterministic CUDA); epoch by "
            "validation BCE and TPE (40 trials + top-3 seed repeats) on the round's own "
            "closed-loop sub-04 candidates",
            final="round-3 checkpoints of all three seeds with the round-3 selected threshold "
            "and strength, rolled out on sub-04 and sealed before the original evaluator",
            primary="seed-mean global RoA, P3 - L2 full loop (the three L2 seeds of the L2 "
            "ablation, which reproduced the accepted run), paired cluster bootstrap (2000 "
            "draws, seed 2026, condition x ground-truth unit)",
            decisions=[
                "P3 improves on L2 only if the primary 95% CI lies above zero",
                "closed loop keeps the open-loop level unless the P3 - P2 open-loop CI lies "
                "below zero; above zero means feedback adds to it",
                "the P3 head enters the single sub-05 opening only if it improves on L2",
                "latency is measured in isolation and reported against the 22 ms target",
                "nothing is deployed and no accepted level changes without the user",
            ],
            known_bias="sub-04 was used for P1/P2 selection and diagnostics; sub-05 answers it",
            pilot="none; a one-record streaming smoke check only (no labels)",
        ),
    )
    return read(FOLDER / "protocol.json")


def gpu(*arguments):
    """Run a GPU step in the authorized CUDA environment, logged, deterministic."""
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONPATH=str(ROOT))
    return subprocess.Popen(
        [str(GPU_PYTHON), "-u", "-m", "benchmarks.pareto_mc.closed_context", *arguments],
        cwd=ROOT,
        env=environment,
    )


def run_round(index, workers):
    checked(FOLDER)
    folder = FOLDER / f"round{index}"
    rollout(folder / "rollout", records(TRAIN) + records(VALIDATION), round_head(index), workers)
    if not (folder / "selected.json").exists():
        processes = [gpu("fit", str(index), str(seed)) for seed in SEEDS]
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"Round {index} training failed: {codes}")
        if gpu("select", str(index)).wait():
            raise RuntimeError(f"Round {index} selection failed")
    print("P3 round", index, read(folder / "selected.json")["selection"], now(), flush=True)


def run_all(workers):
    register_protocol()
    for index in ROUNDS:
        run_round(index, workers)
    checked(FOLDER)
    for seed, spec in final_heads().items():
        rollout(FOLDER / f"final_validation/seed{seed}", records(VALIDATION), spec, workers)
    result = analyse(workers)
    print("P3 decisions", result["decisions"], now(), flush=True)
    print("P3 latency", latency()["systems"], now(), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["smoke", "register", "run", "fit", "select", "analyse", "latency", "heldout"],
    )
    parser.add_argument("args", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    options = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if options.command == "smoke":
        smoke()
    elif options.command == "register":
        print(sorted(register_protocol()["spec"]))
    elif options.command == "run":
        run_all(options.workers)
    elif options.command == "fit":
        torch.set_num_threads(2)
        checked(FOLDER)
        fit(int(options.args[0]), int(options.args[1]))
    elif options.command == "select":
        torch.set_num_threads(4)
        checked(FOLDER)
        select(int(options.args[0]))
    elif options.command == "analyse":
        checked(FOLDER)
        print(analyse(options.workers)["decisions"])
    elif options.command == "latency":
        checked(FOLDER)
        print(latency())
    elif options.command == "heldout":
        checked(FOLDER)
        register_heldout()
        print(json.dumps(heldout(options.workers)["effects"], indent=1)[:4000])
    sys.exit(0)
