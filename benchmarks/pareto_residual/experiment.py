"""Predeclared subject/noise/speed separation and label-free model selection."""

import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .common import (
    OUTPUT,
    ROOT,
    SUBJECTS,
    TRAIN_CONDITIONS,
    sha256,
    source_hashes,
    split,
    write_json,
)
from .data import prepare
from .model import ResidualGate
from .replay import replay

SPEC = dict(
    seeds=[0, 1],
    epochs=5,
    learning_rate=0.003,
    weight_decay=0.01,
    training_conditions=TRAIN_CONDITIONS,
    training_snrs=[30, 20],
    validation_snrs=[25],
    validation_conditions=TRAIN_CONDITIONS,
    feature_scaling="within-record calibration; no fitted global normalizer",
    loss="0.1 contrast + waveform + 0.2 duplicates + 0.1 displacement + 0.02 size + 0.01 smooth",
    selection="label-free validation only; initial zero head is eligible",
    deployment="mean coefficients of both seeds, never best test seed",
    labels_used=False,
    additional_minimum_validation_improvement_fraction=0.005,
    maximum_validation_condition_proxy_regression=0.01,
)


def register():
    path = OUTPUT / "protocol.json"
    protocol = dict(
        registered_at_utc=datetime.now(UTC).isoformat(),
        spec=SPEC,
        folds={s: split(s) for s in SUBJECTS[1:]},
        source_sha256=source_hashes(),
        guide_sha256=(
            sha256(ROOT / "MUAP_dynamic_DL_research_report.pdf")
            if (ROOT / "MUAP_dynamic_DL_research_report.pdf").is_file()
            else None
        ),
        evaluation="80 outer-subject records; compare to SAME 80 Pareto records",
        unseen_during_weight_fit_and_selection="outer subject, 15dB, ramp5s and ramp20s",
        prior_exposure=(
            "All original P1 outcomes were previously examined. This is exploratory "
            "grouped validation, not a fresh blind confirmatory test."
        ),
        baseline_tuning="sub-01 was the published Pareto tuning subject and stays in training",
        success=dict(
            ramp5s_delta_pp=5,
            ramp10s_delta_pp=5,
            staircase_delta_pp=0,
            final_phase_delta_pp=0,
            final_tracked_unit_delta=0,
        ),
    )
    if path.exists():
        raise ValueError("Protocol already registered; preserve it")
    write_json(path, protocol)
    destination = OUTPUT / "source"
    destination.mkdir(exist_ok=True)
    for file in Path(__file__).parent.glob("*.py"):
        shutil.copy2(file, destination / file.name)
    return protocol


def checked_protocol():
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    if protocol["source_sha256"] != source_hashes() or protocol["spec"] != SPEC:
        raise ValueError("Source or specification changed after registration")
    return protocol


def records(subjects, conditions, snrs):
    return [prepare(s, c, n) for s in subjects for c in conditions for n in snrs]


@torch.no_grad()
def validate(model, validation):
    values = []
    for record in validation:
        metrics = replay(record, [model])
        metrics.pop("predictions")
        metrics.pop("sources")
        values.append(dict(condition=record["metadata"]["condition"], **metrics))
    return dict(objective=float(np.mean([r["objective"] for r in values])), records=values)


def train(test, seed):
    checked_protocol()
    if (OUTPUT / "locked.json").exists():
        raise ValueError("All candidates are frozen; test results cannot trigger retraining")
    if seed not in SPEC["seeds"]:
        raise ValueError("Undeclared seed")
    output = OUTPUT / "folds" / test / f"seed{seed}"
    if output.exists():
        raise ValueError("Training exists; preserve it and use a new experiment folder")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    partition = split(test)
    training = records(partition["training"], TRAIN_CONDITIONS, SPEC["training_snrs"])
    validation = records(partition["validation"], TRAIN_CONDITIONS, [25])
    model = ResidualGate()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SPEC["learning_rate"], weight_decay=SPEC["weight_decay"]
    )
    output.mkdir(parents=True)
    baseline = validate(model, validation)
    best = baseline["objective"]
    best_epoch = 0
    best_validation = baseline

    def save(path, epoch, value):
        torch.save(
            dict(
                model=model.state_dict(),
                spec=model.spec,
                epoch=epoch,
                validation=value,
                fold=partition,
                seed=seed,
                labels_used=False,
            ),
            path,
        )

    save(output / "initial.pt", 0, baseline)
    save(output / "best.pt", 0, baseline)
    history = [dict(epoch=0, validation=baseline, training=None)]
    for epoch in range(1, SPEC["epochs"] + 1):
        rng.shuffle(training)
        losses = []
        for record in training:
            metrics = replay(record, [model], optimizer=optimizer)
            losses.append(metrics["objective"])
        val = validate(model, validation)
        if val["objective"] < best:
            best, best_epoch, best_validation = val["objective"], epoch, val
            save(output / "best.pt", epoch, val)
        history.append(dict(epoch=epoch, training=float(np.mean(losses)), validation=val))
        write_json(output / "history.json", dict(seed=seed, fold=partition, epochs=history))
        print(test, seed, epoch, "validation", val["objective"], flush=True)
    improved = best <= baseline["objective"] * (
        1 - SPEC["additional_minimum_validation_improvement_fraction"]
    )
    safe = all(
        r["objective"] <= b["objective"] + SPEC["maximum_validation_condition_proxy_regression"]
        for r, b in zip(best_validation["records"], baseline["records"], strict=True)
    )
    chosen = "best.pt" if improved and safe else "initial.pt"
    shutil.copy2(output / chosen, output / "selected.pt")
    write_json(
        output / "selection.json",
        dict(
            seed=seed,
            fold=partition,
            best_epoch=best_epoch,
            selected=chosen,
            selected_sha256=sha256(output / "selected.pt"),
            baseline_validation=baseline["objective"],
            best_validation=best,
            minimum_improvement_passed=improved,
            condition_safety_passed=safe,
            criterion_uses_ground_truth=False,
        ),
    )


def freeze():
    protocol = checked_protocol()
    if (OUTPUT / "locked.json").exists():
        raise ValueError("Already frozen")
    models = {}
    for subject in SUBJECTS[1:]:
        models[subject] = {}
        for seed in SPEC["seeds"]:
            path = OUTPUT / "folds" / subject / f"seed{seed}" / "selected.pt"
            selection = json.loads(path.with_name("selection.json").read_text())
            if selection["selected_sha256"] != sha256(path):
                raise ValueError("Checkpoint changed")
            models[subject][str(seed)] = dict(sha256=sha256(path), selection=selection)
    write_json(
        OUTPUT / "locked.json",
        dict(
            frozen_at_utc=datetime.now(UTC).isoformat(),
            protocol_sha256=sha256(OUTPUT / "protocol.json"),
            models=models,
            source_sha256=protocol["source_sha256"],
            test_ground_truth_used_for_selection=False,
        ),
    )
