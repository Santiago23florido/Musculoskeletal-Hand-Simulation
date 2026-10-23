"""TPE in batches of nine, sharing only head-independent causal computations."""

from concurrent.futures import as_completed

import numpy as np
import torch

from .common import OUTPUT, VALIDATION, now, records, sha256, stub, write_new
from .data import load_record
from .evaluate import score_record, summarize
from .events import descriptor
from .experiment import checked
from .memory_experiment import (
    SEEDS,
    TRIALS,
    l1_selection,
    objective_value,
    parameters,
    read,
    seed_selection,
)
from .shared_rollout import prepare, shared_replay


def verify_completed(folder, case, selection, settings):
    path = folder / "metadata" / f"{stub(*case)}.json"
    if not path.exists():
        return False
    metadata = read(path)
    if metadata["settings"] != settings or metadata["head"] != selection:
        raise ValueError("Search resume parameters changed")
    for key, digest in metadata["files"].items():
        if sha256(folder / key) != digest:
            raise ValueError("Completed search prediction changed")
    return True


def shared_task(arguments):
    case, runs = arguments
    torch.set_num_threads(4)
    pending = [(f, s, c) for f, s, c in runs if not verify_completed(f, case, s, c)]
    if not pending:
        return stub(*case)
    record = load_record(*case, phase="validation")
    input_metadata = read(OUTPUT / "C1/features" / f"{stub(*case)}.json")
    for key in ("raw_sha256", "calibration_sha256"):
        if record[key] != input_metadata[key]:
            raise ValueError("Search inputs changed since C1")
    for _, selection, _ in pending:
        if sha256(OUTPUT / selection["checkpoint"]) != selection["checkpoint_sha256"]:
            raise ValueError("Search classifier changed")
    master, adapters, observer, extractor = prepare(record, [(s, c) for _, s, c in pending])
    predictions = shared_replay(
        record["raw"], master, adapters, observer=observer, extractor=extractor
    )
    for (folder, selection, settings), prediction in zip(pending, predictions, strict=True):
        destination = folder / "predictions" / f"{stub(*case)}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError("Orphan search prediction: preserve and use a fresh run")
        np.savez_compressed(destination, **prediction)
        write_new(
            folder / "metadata" / f"{stub(*case)}.json",
            dict(
                predicted_at_utc=now(),
                settings=settings,
                head=selection,
                labels_opened=False,
                shared="raw preprocessing, whitening and blind C1 only; per-batch causal reuse",
                memory_hit_rate=float(prediction["memory"][:, 3].mean()),
                files={destination.relative_to(folder).as_posix(): sha256(destination)},
            ),
        )
    return stub(*case)


def group_rollout(pool, runs):
    futures = [pool.submit(shared_task, (case, runs)) for case in records(VALIDATION)]
    for future in as_completed(futures):
        print("search group", len(runs), future.result(), flush=True)
    for folder, _, _ in runs:
        files = {}
        for case in records(VALIDATION):
            metadata = read(folder / "metadata" / f"{stub(*case)}.json")
            for key, digest in metadata["files"].items():
                if sha256(folder / key) != digest:
                    raise ValueError("Search prediction changed before seal")
                files[key] = digest
        if not (folder / "sealed.json").exists():
            write_new(
                folder / "sealed.json",
                dict(sealed_at_utc=now(), files=files, no_labels_during_rollout=True),
            )
        elif read(folder / "sealed.json")["files"] != files:
            raise ValueError("Existing search seal changed")


def score_search(folder):
    for key, digest in read(folder / "sealed.json")["files"].items():
        if sha256(folder / key) != digest:
            raise ValueError("Search seal violated")
    summary_path = folder / "summary.json"
    if summary_path.exists():
        return read(summary_path)
    results = []
    for case in records(VALIDATION):
        name = stub(*case)
        path = folder / "scores" / f"{name}.json"
        if not path.exists():
            record = descriptor(*case, phase="validation")
            with np.load(folder / "predictions" / f"{name}.npz") as archive:
                prediction = dict(spikes=archive["spikes"])
            result = score_record(record, prediction, phase="validation")
            write_new(path, result)
        results.append(read(path))
    metrics = summarize(results)
    metrics["timing_scope"] = "shared search work; final gate measures the full ordinary CPU loop"
    metrics["memory_hit_rate"] = float(
        np.mean([read(p)["memory_hit_rate"] for p in (folder / "metadata").glob("*.json")])
    )
    write_new(summary_path, metrics)
    return metrics


def search(root, pool, previous, *, calibration_only, conservative):
    import optuna

    identity = read(OUTPUT / "C2_shared_identity_v1/result.json")
    if not identity["passed"] or identity["memory_hit_rate"] <= 0:
        raise ValueError("Shared search has not passed real active-memory equivalence")
    destination = root / "selected_memory.json"
    if destination.exists():
        return read(destination)
    baseline = read(OUTPUT / "L0/validation/summary.json")["precision_pct"]
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2026),
        storage=f"sqlite:///{(root / 'tpe.sqlite3').as_posix()}",
        study_name="memory",
        load_if_exists=True,
    )
    if not study.trials:
        for index in range(TRIALS):
            study.enqueue_trial(dict(structure=index % 9))

    def selection(seed):
        return (
            l1_selection(seed) if calibration_only else seed_selection(previous["selection"], seed)
        )

    while True:
        checked(root)
        active = [
            optuna.trial.Trial(study, t._trial_id)
            for t in study.trials
            if t.state == optuna.trial.TrialState.RUNNING
        ]
        if not active:
            remaining = sum(t.state == optuna.trial.TrialState.WAITING for t in study.trials)
            active = [study.ask() for _ in range(min(9, remaining))]
        if not active:
            break
        runs = []
        for trial in active:
            settings = parameters(
                trial, calibration_only=calibration_only, conservative=conservative
            )
            trial.set_user_attr("settings", settings)
            folder = root / "search" / f"trial{trial.number}/seed0"
            runs.append((folder, selection(0), settings))
        try:
            group_rollout(pool, runs)
        except FloatingPointError as error:
            for trial in active:
                trial.set_user_attr("error", str(error))
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue
        for trial, (folder, _, _) in zip(active, runs, strict=True):
            metrics = score_search(folder)
            trial.set_user_attr("metrics", metrics)
            value = objective_value(metrics, baseline)
            study.tell(trial, value)
            print("memory trial", trial.number, value, metrics, flush=True)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) < 3:
        raise RuntimeError("Insufficient complete trials; failed trials remain in the study")
    top = sorted(complete, key=lambda t: t.value, reverse=True)[:3]
    runs = [
        (
            root / "search" / f"trial{trial.number}/seed{seed}",
            selection(seed),
            trial.user_attrs["settings"],
        )
        for trial in top
        for seed in SEEDS
    ]
    group_rollout(pool, runs)
    repeated = []
    for trial in top:
        metrics = [
            score_search(root / "search" / f"trial{trial.number}/seed{seed}") for seed in SEEDS
        ]
        values = [objective_value(value, baseline) for value in metrics]
        seed = int(np.argsort(values)[1])
        repeated.append(
            dict(
                trial=trial.number,
                settings=trial.user_attrs["settings"],
                metrics_by_seed=metrics,
                median_objective=float(np.median(values)),
                seed=seed,
                selection=selection(seed),
            )
        )
    result = dict(
        selected_at_utc=now(),
        winner=max(repeated, key=lambda value: value["median_objective"]),
        repeats=repeated,
    )
    write_new(destination, result)
    write_new(
        root / "trials.json",
        [
            dict(
                number=t.number,
                state=t.state.name,
                value=t.value,
                params=t.params,
                attrs=t.user_attrs,
            )
            for t in study.trials
        ],
    )
    return result
