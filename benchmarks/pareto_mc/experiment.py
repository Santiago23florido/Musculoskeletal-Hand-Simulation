"""L1 DAgger experiment. All rollouts and timing run in native Windows CPU Python."""

import json
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from .backbone import replay
from .closed_loop import make_adapter
from .common import (
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
from .data import load_record
from .evaluate import score_record, summarize
from .events import descriptor, prepare_open
from .observer import load_observer
from .train_likelihood import load_model


def checked(folder):
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    for relative, expected in protocol["sources"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Registered source changed: {relative}; use a fresh run")
    return protocol


def config_hash(case):
    path = pareto_path(*case)
    return sha256(path.with_name(path.stem + "_config.yaml"))


def open_task(case):
    torch.set_num_threads(2)
    phase = "training" if case[0] in TRAIN else "validation"
    arrays = prepare_open(*case, phase=phase)
    return stub(*case), len(arrays["time"])


def prepare_all_open():
    cases = records(TRAIN) + records(VALIDATION)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(open_task, case) for case in cases]):
            print("prepared open", *future.result(), flush=True)


def rollout_task(args):
    folder, case, selection, guarded = args
    torch.set_num_threads(4)
    name = stub(*case)
    metadata_path = folder / "metadata" / f"{name}.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, digest in metadata["files"].items():
            if sha256(folder / key) != digest:
                raise ValueError("Changed rollout artifact")
        return name
    phase = "training" if case[0] in TRAIN else "validation"
    record = load_record(*case, phase=phase)
    input_metadata = json.loads((OUTPUT / "C1/features" / f"{name}.json").read_text())
    if any(record[key] != input_metadata[key] for key in ("raw_sha256", "calibration_sha256")):
        raise ValueError("Raw EMG or calibration changed after observer feature preparation")
    checkpoint = OUTPUT / selection["checkpoint"]
    if sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("Selected classifier checkpoint changed")
    params = selection["selection"]["parameters"]
    head, observer = load_model(checkpoint), load_observer()
    adapter = make_adapter(record, head, observer, **params, guarded_feedback=guarded)
    prediction = replay(record["raw"], adapter)
    events = adapter.collected(prediction["spikes"][: 5 * record["config"].fs])
    destination = folder / "predictions" / f"{name}.npz"
    event_path = folder / "events" / f"{name}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **prediction, state=np.asarray(adapter.state_trace))
    np.savez_compressed(event_path, **events)
    write_new(
        metadata_path,
        dict(
            predicted_at_utc=now(),
            record=case,
            labels_opened=False,
            checkpoint_sha256=sha256(checkpoint),
            observer_sha256=sha256(OUTPUT / "C1/seed0/observer.pt"),
            raw_sha256=record["raw_sha256"],
            calibration_sha256=record["calibration_sha256"],
            files={p.relative_to(folder).as_posix(): sha256(p) for p in (destination, event_path)},
            median_batch_ms=prediction["median_batch_ms"],
            p95_batch_ms=prediction["p95_batch_ms"],
            guarded_feedback=guarded,
            memory_hits="not active at L1",
            control="identity at L1",
        ),
    )
    return name


def rollout(folder, cases, selection, guarded):
    folder.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(rollout_task, (folder, case, selection, guarded)) for case in cases]
        for future in as_completed(tasks):
            print("rollout", folder.name, future.result(), flush=True)
    seal_path = folder / "sealed.json"
    if not seal_path.exists():
        files = {}
        for case in cases:
            metadata = json.loads((folder / "metadata" / f"{stub(*case)}.json").read_text())
            for key, digest in metadata["files"].items():
                if sha256(folder / key) != digest:
                    raise ValueError("Predictions changed before seal")
                files[key] = digest
        write_new(
            seal_path,
            dict(
                sealed_at_utc=now(),
                cases=cases,
                files=files,
                no_true_spikes_opened_during_rollout=True,
            ),
        )


def score_validation(folder):
    seal = json.loads((folder / "sealed.json").read_text(encoding="utf-8"))
    for key, digest in seal["files"].items():
        if sha256(folder / key) != digest:
            raise ValueError("A sealed prediction was changed")
    results, durations = [], []
    for case in records(VALIDATION):
        name = stub(*case)
        record = descriptor(*case, phase="validation")
        with np.load(folder / "predictions" / f"{name}.npz") as archive:
            prediction = {key: archive[key] for key in archive.files}
        durations.extend(prediction["batch_ms"].tolist())
        path = folder / "scores" / f"{name}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            result = score_record(record, prediction, phase="validation")
            write_new(path, result)
        results.append(result)
    summary = summarize(results)
    summary.update(
        median_batch_ms=float(np.median(durations)),
        p95_batch_ms=float(np.percentile(durations, 95)),
        timing_scope="native Windows CPU; raw filtering through closed-loop update; "
        "four independent record workers, four Torch threads each",
    )
    if not (folder / "summary.json").exists():
        write_new(folder / "summary.json", summary)
    return summary


def train_round(variant, index):
    folder = OUTPUT / "L1" / variant / f"round{index}"
    folder.mkdir(parents=True, exist_ok=True)
    selected = folder / "selected.json"
    if variant == "guarded_feedback" and index == 0 and not selected.exists():
        original = OUTPUT / "L1/all_feedback/round0/selected.json"
        selection = json.loads(original.read_text(encoding="utf-8"))
        selection["reused_from"] = str(original.relative_to(OUTPUT))
        write_new(selected, selection)
    if not selected.exists():
        with (folder / "training.log").open("a", encoding="utf-8") as log:
            subprocess.run(
                [
                    str(ROOT / ".venv-stage-b-gpu/Scripts/python.exe"),
                    "-m",
                    "benchmarks.pareto_mc.train_likelihood",
                    variant,
                    str(index),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    return json.loads(selected.read_text(encoding="utf-8"))


def score_open(selection):
    from .train_likelihood import decode_record, predict_logits

    folder = OUTPUT / "L1/open/evaluation"
    if (folder / "summary.json").exists():
        return json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    params = selection["selection"]["parameters"]
    model = load_model(OUTPUT / selection["checkpoint"])
    files = {}
    for case in records(VALIDATION):
        name = stub(*case)
        with np.load(OUTPUT / "L1/open/features" / f"{name}.npz") as archive:
            events = {key: archive[key] for key in archive.files}
        record = descriptor(*case, phase="validation")
        events["record"] = record
        logits = predict_logits(model, events)
        binary = decode_record(events, logits, **params)
        path = folder / "predictions" / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, spikes=binary)
        files[path.relative_to(folder).as_posix()] = sha256(path)
    write_new(
        folder / "sealed.json",
        dict(sealed_at_utc=now(), files=files, checkpoint_sha256=selection["checkpoint_sha256"]),
    )
    results = []
    for case in records(VALIDATION):
        name = stub(*case)
        record = descriptor(*case, phase="validation")
        with np.load(folder / "predictions" / f"{name}.npz") as archive:
            prediction = dict(spikes=archive["spikes"])
        result = score_record(record, prediction, phase="validation")
        write_new(folder / "scores" / f"{name}.json", result)
        results.append(result)
    summary = summarize(results)
    write_new(folder / "summary.json", summary)
    return summary


def run_l1():
    root = OUTPUT / "L1"
    if not (root / "protocol.json").exists():
        l0 = json.loads((OUTPUT / "L0/result.json").read_text())
        if not l0["passed"]:
            raise ValueError("L0 gate has not passed")
        identity = json.loads((OUTPUT / "C4_identity/result.json").read_text())
        if not identity["passed"]:
            raise ValueError("Untrained C4 must reproduce Pareto on all three real controls")
        register(
            root,
            dict(
                level="L1",
                training=records(TRAIN),
                validation=records(VALIDATION),
                observer_sha256=sha256(OUTPUT / "C1/seed0/observer.pt"),
                reused_inputs_manifest_sha256=(
                    sha256(OUTPUT / "input_reuse.json")
                    if (OUTPUT / "input_reuse.json").exists()
                    else None
                ),
                dagger_rounds=[0, 1, 2, 3],
                epochs=10,
                seeds=[0, 1, 2],
                tpe_trials_per_round=40,
                repeat_top=3,
                repeat_seeds=[0, 1, 2],
                active_search_parameters="likelihood threshold [0.25,0.85] and strength [0,1]",
                memory_parameters="not active until L2; never search parameters with no effect",
                objective="0.5 global + 0.5 ramp5s Jaccard; -5 * precision loss beyond 1 pp",
                gate="global >= open likelihood; ramp5s >= open likelihood + 1 pp",
                structural_variants=["all_feedback", "guarded_feedback"],
                structural_change="second variant requires posterior >=0.8 for added events to "
                "update SV/centroids; all accepted events remain predictions",
                maximum_level_iterations=2,
                cpu_workers=4,
                cpu_threads_per_worker=4,
                reference="Pareto-rerun",
                sub05="never opened in development",
                environment=dict(
                    platform=platform.platform(),
                    torch=torch.__version__,
                    numpy=np.__version__,
                    openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
                ),
                input_config_sha256={
                    stub(*case): config_hash(case) for case in records(TRAIN) + records(VALIDATION)
                },
            ),
        )
    checked(root)
    prepare_all_open()
    baseline = None
    if (root / "open/evaluation/summary.json").exists():
        baseline = json.loads((root / "open/evaluation/summary.json").read_text(encoding="utf-8"))
    for variant in ("all_feedback", "guarded_feedback"):
        folder = root / variant
        if (folder / "gate.json").exists():
            result = json.loads((folder / "gate.json").read_text())
            if result["passed"]:
                return result
            continue
        selection = train_round(variant, 0)
        if baseline is None:
            baseline = score_open(selection)
        for index in (1, 2, 3):
            checked(root)
            rollout(
                folder / f"round{index}/rollout",
                records(TRAIN) + records(VALIDATION),
                selection,
                variant == "guarded_feedback",
            )
            selection = train_round(variant, index)
        checked(root)
        rollout(
            folder / "final_validation",
            records(VALIDATION),
            selection,
            variant == "guarded_feedback",
        )
        actual = score_validation(folder / "final_validation")
        result = dict(
            decided_at_utc=now(),
            variant=variant,
            reference=baseline,
            actual=actual,
            global_pass=actual["global_roa_pct"] >= baseline["global_roa_pct"],
            ramp5s_pass=actual["ramp5s_roa_pct"] >= baseline["ramp5s_roa_pct"] + 1,
        )
        result["passed"] = result["global_pass"] and result["ramp5s_pass"]
        write_new(folder / "gate.json", result)
        print("L1 gate", result, flush=True)
        if result["passed"]:
            return result
    result = dict(
        completed_at_utc=now(),
        passed=False,
        retained_level="L0",
        reason="Both preregistered L1 structures failed; stop before L2",
    )
    write_new(root / "stopped.json", result)
    return result
