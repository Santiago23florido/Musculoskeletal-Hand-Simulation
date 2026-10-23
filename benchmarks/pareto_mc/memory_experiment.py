"""Preregistered L2/L3 memory search, DAgger trajectories, and sequential gates."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .backbone import replay
from .common import OUTPUT, ROOT, TRAIN, VALIDATION, now, records, register, sha256, stub, write_new
from .data import load_record
from .experiment import checked, config_hash, score_validation
from .memory_loop import MemoryConfig, make_memory_adapter
from .observer import load_observer
from .train_likelihood import load_model

STRUCTURES = [(bins, h) for bins in (8, 16, 32) for h in (0.03, 0.06, 0.12)]
TRIALS = 45
SEEDS = (0, 1, 2)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def l1_selection(seed=None):
    selection = read(OUTPUT / "L1/all_feedback/round3/selected.json")
    if seed is not None:
        selection = seed_selection(selection, seed)
    return selection


def seed_selection(selection, seed):
    selection = dict(selection)
    checkpoint = (OUTPUT / selection["checkpoint"]).parent.parent / f"seed{seed}/model.pt"
    completed = read(checkpoint.with_name("complete.json"))
    if sha256(checkpoint) != completed["checkpoint_sha256"]:
        raise ValueError("Previously trained seed checkpoint changed")
    selection["checkpoint"] = checkpoint.relative_to(OUTPUT).as_posix()
    selection["checkpoint_sha256"] = sha256(checkpoint)
    return selection


def objective_value(metrics, baseline_precision):
    loss = max(0.0, baseline_precision - metrics["precision_pct"] - 1)
    return 0.5 * (metrics["global_roa_pct"] + metrics["ramp5s_roa_pct"]) - 5 * loss


def rollout_task(arguments):
    folder, case, selection, settings, detailed = arguments
    torch.set_num_threads(4)
    name = stub(*case)
    metadata_path = folder / "metadata" / f"{name}.json"
    if metadata_path.exists():
        metadata = read(metadata_path)
        if metadata["settings"] != settings or metadata["head"] != selection:
            raise ValueError("Cannot reuse a rollout with different memory or weights")
        for key, digest in metadata["files"].items():
            if sha256(folder / key) != digest:
                raise ValueError("Completed memory rollout changed")
        return name
    phase = "training" if case[0] in TRAIN else "validation"
    record = load_record(*case, phase=phase)
    checkpoint = OUTPUT / selection["checkpoint"]
    if sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("Classifier checkpoint changed")
    input_metadata = read(OUTPUT / "C1/features" / f"{name}.json")
    for key in ("raw_sha256", "calibration_sha256"):
        if record[key] != input_metadata[key]:
            raise ValueError("Raw EMG or calibration changed since C1")
    adapter = make_memory_adapter(
        record,
        load_model(checkpoint),
        load_observer(),
        memory_config=MemoryConfig(**settings),
        **selection["selection"]["parameters"],
    )
    adapter.collector = detailed
    prediction = replay(record["raw"], adapter, collect_sources=detailed)
    if not detailed:
        prediction.pop("sources")
    prediction["state"] = np.asarray(adapter.state_trace)
    prediction["memory"] = np.asarray(adapter.memory_trace)
    destination = folder / "predictions" / f"{name}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("Orphaned prediction: preserve this run and use a fresh directory")
    np.savez_compressed(destination, **prediction)
    files = [destination]
    if detailed:
        events = adapter.collected(prediction["spikes"][: 5 * record["config"].fs])
        event_path = folder / "events" / f"{name}.npz"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(event_path, **events)
        files.append(event_path)
    write_new(
        metadata_path,
        dict(
            predicted_at_utc=now(),
            record=case,
            labels_opened=False,
            settings=settings,
            head=selection,
            raw_sha256=record["raw_sha256"],
            calibration_sha256=record["calibration_sha256"],
            memory_hit_rate=float(prediction["memory"][:, 3].mean()),
            max_memory_relative_step=float(prediction["memory"][:, 2].max()),
            files={path.relative_to(folder).as_posix(): sha256(path) for path in files},
        ),
    )
    return name


def rollout(pool, folder, cases, selection, settings, *, detailed):
    futures = [
        pool.submit(rollout_task, (folder, case, selection, settings, detailed)) for case in cases
    ]
    for future in as_completed(futures):
        print("memory rollout", folder, future.result(), flush=True)
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


def parameters(trial, *, calibration_only, conservative):
    structure = trial.suggest_categorical("structure", list(range(9)))
    bins, h = STRUCTURES[structure]
    return asdict(
        MemoryConfig(
            bins=bins,
            h=h,
            threshold=trial.suggest_float("threshold", 0.5, 0.95),
            kappa=trial.suggest_float("kappa", 0.02, 0.3),
            beta_max=trial.suggest_float("beta_max", 0.1, 0.8),
            # Bin momentum has no effect on a read-only calibration snapshot.
            momentum=0.95 if calibration_only else trial.suggest_float("momentum", 0.8, 0.99),
            calibration_only=calibration_only,
            conservative=conservative,
        )
    )


def search_independent(root, pool, previous, *, calibration_only, conservative):
    import optuna

    destination = root / "selected_memory.json"
    if destination.exists():
        return read(destination)
    baseline_precision = read(OUTPUT / "L0/validation/summary.json")["precision_pct"]
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2026),
        storage=f"sqlite:///{(root / 'tpe.sqlite3').as_posix()}",
        study_name="memory",
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        # Five continuous TPE trials for each declared structural grid point.
        for index in range(TRIALS):
            study.enqueue_trial(dict(structure=index % len(STRUCTURES)))

    def evaluate(trial):
        checked(root)
        settings = parameters(trial, calibration_only=calibration_only, conservative=conservative)
        trial.set_user_attr("settings", settings)
        selection = (
            l1_selection(0) if calibration_only else seed_selection(previous["selection"], 0)
        )
        folder = root / "search" / f"trial{trial.number}" / "seed0"
        rollout(pool, folder, records(VALIDATION), selection, settings, detailed=False)
        metrics = score_validation(folder)
        trial.set_user_attr("metrics", metrics)
        return objective_value(metrics, baseline_precision)

    # RUNNING trials remain auditable after a process interruption; an incomplete
    # one is resumed through the fixed parameter values, never silently discarded.
    for trial in study.get_trials(deepcopy=False):
        if trial.state == optuna.trial.TrialState.RUNNING:
            active = optuna.trial.Trial(study, trial._trial_id)
            try:
                value = evaluate(active)
            except Exception:
                study.tell(active, state=optuna.trial.TrialState.FAIL)
                raise
            study.tell(active, value)
    remaining = sum(t.state == optuna.trial.TrialState.WAITING for t in study.trials)
    study.optimize(evaluate, n_trials=remaining, catch=(FloatingPointError,))
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) < 3:
        raise RuntimeError("Fewer than three successful full-record memory trials")
    repeated = []
    for trial in sorted(complete, key=lambda t: t.value, reverse=True)[:3]:
        values, metrics = [], []
        settings = trial.user_attrs["settings"]
        for seed in SEEDS:
            selection = (
                l1_selection(seed)
                if calibration_only
                else seed_selection(previous["selection"], seed)
            )
            folder = root / "search" / f"trial{trial.number}" / f"seed{seed}"
            rollout(pool, folder, records(VALIDATION), selection, settings, detailed=False)
            actual = score_validation(folder)
            metrics.append(actual)
            values.append(objective_value(actual, baseline_precision))
        seed = int(np.argsort(values)[1])
        repeated.append(
            dict(
                trial=trial.number,
                settings=settings,
                metrics_by_seed=metrics,
                median_objective=float(np.median(values)),
                seed=seed,
                selection=(
                    l1_selection(seed)
                    if calibration_only
                    else seed_selection(previous["selection"], seed)
                ),
            )
        )
    winner = max(repeated, key=lambda item: item["median_objective"])
    result = dict(selected_at_utc=now(), winner=winner, repeats=repeated)
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


def train(root, index):
    folder = root / f"round{index}"
    selection_path = folder / "selected.json"
    if not selection_path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "training.log").open("a", encoding="utf-8") as log:
            subprocess.run(
                [
                    str(ROOT / ".venv-stage-b-gpu/Scripts/python.exe"),
                    "-u",
                    "-m",
                    "benchmarks.pareto_mc.memory_training",
                    str(root),
                    str(index),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    return read(selection_path)


def run_variant(root, previous, *, calibration_only, conservative, workers):
    if (root / "gate.json").exists():
        return read(root / "gate.json")
    if not (root / "protocol.json").exists():
        register(
            root,
            dict(
                level="L2" if calibration_only else "L3",
                cases=records(TRAIN) + records(VALIDATION),
                reference=previous,
                structures=STRUCTURES,
                tpe_trials=TRIALS,
                tpe_execution="ask/tell batches of nine; current-batch whitening/C1 reused exactly",
                search_equivalence_sha256=sha256(OUTPUT / "C2_shared_identity_v1/result.json"),
                continuous="threshold [.5,.95], kappa [.02,.3], beta_max [.1,.8]",
                momentum="fixed .95 (inactive)" if calibration_only else "TPE [.8,.99]",
                confidence_threshold=(
                    "minimum supported retrieval mass; also insertion quality at L3"
                ),
                uncertainty="beta attenuated by exp(-0.5*predicted_variance/h**2)",
                conservative=conservative,
                repeats="top three, C4 training seeds 0/1/2; median objective",
                search_classifier="frozen L1 round3 seeds, before memory-specific DAgger",
                observer_sha256=sha256(OUTPUT / "C1/seed0/observer.pt"),
                classifier_sha256={seed: l1_selection(seed)["checkpoint_sha256"] for seed in SEEDS},
                dagger="three new rounds; aggregate all L1 and completed memory trajectories",
                objective=".5 global + .5 ramp5s RoA; -5*precision loss beyond 1pp vs Pareto",
                gate=(
                    "reference_rest >=97 and moving >=retained L1"
                    if calibration_only
                    else "ramp5s cycles2-8 >=cycle1+5 and global >=retained L2+1"
                ),
                limiter="post-QR residual relative to the current Pareto proposal <=kappa",
                centroids="saved and retrieved; live centroid updates remain gradual L1 EMA",
                workers=workers,
                torch_threads=4,
                openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
                sub05="never opened",
                input_config_sha256={
                    stub(*case): config_hash(case) for case in records(TRAIN) + records(VALIDATION)
                },
            ),
        )
    checked(root)
    identity = read(OUTPUT / "C2_identity_v1/result.json")
    if not identity["passed"]:
        raise ValueError("C2 real-record identity gate failed")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        from .memory_search import search

        selected = search(
            root, pool, previous, calibration_only=calibration_only, conservative=conservative
        )["winner"]
        selection, settings = selected["selection"], selected["settings"]
        for index in (1, 2, 3):
            checked(root)
            rollout(
                pool,
                root / f"round{index}/rollout",
                records(TRAIN) + records(VALIDATION),
                selection,
                settings,
                detailed=True,
            )
            selection = train(root, index)
        checked(root)
        rollout(
            pool,
            root / "final_validation",
            records(VALIDATION),
            selection,
            settings,
            detailed=True,
        )
    actual = score_validation(root / "final_validation")
    metadata = [read(p) for p in (root / "final_validation/metadata").glob("*.json")]
    actual["memory_hit_rate"] = float(np.mean([m["memory_hit_rate"] for m in metadata]))
    if calibration_only:
        passes = dict(
            reference_rest=actual["phases"]["reference_rest"] >= 97,
            moving=actual["phases"]["moving"] >= previous["actual"]["phases"]["moving"],
        )
    else:
        passes = dict(
            cycles=actual["ramp5s_cycles2_8_roa_pct"] >= actual["ramp5s_cycle1_roa_pct"] + 5,
            global_gain=actual["global_roa_pct"] >= previous["actual"]["global_roa_pct"] + 1,
        )
    result = dict(
        decided_at_utc=now(),
        run_root=root.relative_to(OUTPUT).as_posix(),
        actual=actual,
        gates=passes,
        passed=all(passes.values()),
        selection=selection,
        settings=settings,
        reference=previous["actual"],
    )
    write_new(root / "gate.json", result)
    from .memory_report import publish

    publish(root)
    with (ROOT / "docs/benchmarks/pareto_mc_journal.md").open("a", encoding="utf-8") as journal:
        journal.write(
            f"\n### {root.parent.name}/{root.name} — resultado automático\n\n"
            f"RoA global {actual['global_roa_pct']:.4f} %, ramp5s "
            f"{actual['ramp5s_roa_pct']:.4f} %, precisión {actual['precision_pct']:.4f} %. "
            f"Reposo {actual['phases']['reference_rest']:.4f} %, movimiento "
            f"{actual['phases']['moving']:.4f} %. Puertas: {passes}. "
            f"Superada: {result['passed']}. Diagnósticos y sello en `{root}`.\n"
        )
    print("memory gate", result, flush=True)
    return result


def run(level, workers):
    l1 = read(OUTPUT / "L1/all_feedback/gate.json")
    if not l1["passed"]:
        raise ValueError("L1 gate must pass before memory development")
    if level == "L2":
        previous = l1
    else:
        path = OUTPUT / "L2/retained.json"
        if not path.exists() or not read(path)["passed"]:
            raise ValueError("L2 must pass before full-memory development")
        previous = read(path)
    root = OUTPUT / level
    for conservative in (False, True):
        result = run_variant(
            root / ("confidence_gated" if conservative else "standard"),
            previous,
            calibration_only=level == "L2",
            conservative=conservative,
            workers=workers,
        )
        if result["passed"]:
            if not (root / "retained.json").exists():
                write_new(root / "retained.json", result)
            return result
        if not conservative:
            analysis_path = root / "standard/failure_analysis.json"
            if not analysis_path.exists():
                write_new(
                    analysis_path,
                    dict(
                        analysed_at_utc=now(),
                        failed_gates=result["gates"],
                        metrics=result["actual"],
                        hypothesis=(
                            "retrieval near rest can override poorly supported local decisions"
                        ),
                        single_change="multiply beta by squared local rate/gap/IQR confidence",
                        decision="one further registered structure, then stop if it fails",
                    ),
                )
    result = dict(
        completed_at_utc=now(),
        passed=False,
        retained_level="L1" if level == "L2" else "L2",
        reason="Both preregistered memory structures failed; stop the ladder",
    )
    if not (root / "stopped.json").exists():
        write_new(root / "stopped.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", choices=["L2", "L3"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.level, args.workers)
