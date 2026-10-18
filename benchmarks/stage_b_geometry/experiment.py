"""Registered source-risk refinement; the original observer remains frozen."""

import json
import random
import shutil
import time
from datetime import UTC, datetime

import numpy as np
import torch

from benchmarks.stage_b.common import OUTPUT as ORIGINAL_OUTPUT
from benchmarks.stage_b.experiment import load_models as original_models
from benchmarks.stage_b.experiment import training_device

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
from .decoder import objective, teacher
from .model import GeometricMorph
from .states import prepare_state

SPEC = dict(
    seeds=[0, 1],
    epochs=20,
    learning_rate=0.0003,
    weight_decay=0.001,
    optimizer="AdamW; gradient norm <=1",
    training_device="cuda",
    precision="float32; no AMP/TF32",
    decoder="MVDR; w^T h=1; same direction as original; gain calibrated once",
    source_loss="covariance-whitened projected NMSE + 0.25 normalized output-source MSE",
    auxiliary_loss="0.1 covariance-whitened angular + 0.01 waveform NMSE + 0.001 size prior",
    selection="unit-weighted validation normalized source error; no outer RoA",
    parameterization=(
        "240 shared coefficients; posture polynomial times calibration moments; "
        "no additive waveform"
    ),
    candidates="geometric identity epoch 0; shared physical warp epochs 1..20",
    input_noise_fraction=0.03,
    state_dimension=1,
    observer="original frozen observer for each fold/seed",
    training_conditions=TRAIN_CONDITIONS,
    training_snrs=[30, 20],
    validation_snrs=[25],
    covariance_shrinkage=0.5,
    covariance_ridge=0.001,
    fifo_batches=10,
    pca_dimension=512,
    interpolation_knots=129,
    primary_method="aligned_b",
    incomplete_tests=[
        "independent new data",
        "1s/2.5s true-generator ramps",
        "full raw-stream CPU latency",
    ],
)


def register():
    if (OUTPUT / "protocol.json").exists():
        raise ValueError("Preserve the registered experiment")
    write_json(
        OUTPUT / "protocol.json",
        dict(
            registered_at_utc=datetime.now(UTC).isoformat(),
            spec=SPEC,
            source_sha256=source_hashes(),
            guide_sha256=(
                sha256(ROOT / "MUAP_dynamic_DL_research_report.pdf")
                if (ROOT / "MUAP_dynamic_DL_research_report.pdf").is_file()
                else None
            ),
            original_locked_sha256=sha256(ORIGINAL_OUTPUT / "locked.json"),
            folds={s: split(s) for s in SUBJECTS[1:]},
            prior_exposure=(
                "Theory correction follows previously observed P1 failure; "
                "exploratory, no independent confirmation"
            ),
            targets="same estimated clean-EMG GT-triggered trajectories; not generator MUAP truth",
            test=(
                "all 80 original outer records and 591 calibrated instances; "
                "save all 320 new predictions before scoring"
            ),
            methods=["aligned_b", "aligned_k", "gain_only_b", "static_mvdr"],
            controls=(
                "original CNN with MVDR isolates normalization; static MVDR isolates learned warp"
            ),
        ),
    )
    for relative in source_hashes():
        target = OUTPUT / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def checked():
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    if protocol["spec"] != SPEC or protocol["source_sha256"] != source_hashes():
        raise ValueError("Code/spec changed; preserve this run and register a new experiment")
    if protocol["original_locked_sha256"] != sha256(ORIGINAL_OUTPUT / "locked.json"):
        raise ValueError("Original frozen weights changed")
    return protocol


def cases(test, phase, device):
    snrs = (30, 20) if phase == "training" else (25,)
    result = []
    for subject in split(test)[phase]:
        for condition in TRAIN_CONDITIONS:
            for snr in snrs:
                case = prepare_state(subject, condition, snr, outer=test, phase=phase)
                case = {k: v.to(device) for k, v in case.items()}
                with torch.no_grad():
                    fixed = [teacher(case, j) for j in range(9)]
                result.append((case, fixed))
    return result


@torch.no_grad()
def validate(model, records):
    totals, units = dict(source=0.0, projected=0.0, angular=0.0, waveform=0.0), 0
    for case, fixed in records:
        for index in range(9):
            _, metrics = objective(model, case, index, fixed[index])
            count = len(case["cal"])
            for key, value in metrics.items():
                totals[key] += float(value) * count
            units += count
    return {k: v / units for k, v in totals.items()}


def train(test, seed):
    checked()
    if test not in SUBJECTS[1:] or seed not in SPEC["seeds"] or (OUTPUT / "locked.json").exists():
        raise ValueError("Undeclared or frozen training")
    folder = OUTPUT / "folds" / test / f"seed{seed}"
    if (folder / "complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    device = training_device(seed)
    training, validation = cases(test, "training", device), cases(test, "validation", device)
    model = GeometricMorph().to(device)
    best, history = float("inf"), []

    def candidate(network, epoch):
        nonlocal best
        network.eval()
        values = validate(network, validation)
        history.append(dict(epoch=epoch, validation=values))
        if values["source"] < best:
            best = values["source"]
            torch.save(
                dict(spec=network.spec, model=network.state_dict(), epoch=epoch, validation=values),
                folder / "morph.pt",
            )
        write_json(folder / "history.json", history)
        print(test, seed, epoch, values, flush=True)

    candidate(model, 0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SPEC["learning_rate"], weight_decay=SPEC["weight_decay"]
    )
    indices = [(i, j) for i in range(len(training)) for j in range(9)]
    for epoch in range(1, SPEC["epochs"] + 1):
        model.train()
        for order in torch.randperm(len(indices)).tolist():
            i, j = indices[order]
            case, fixed = training[i]
            loss, _ = objective(model, case, j, fixed[j], noise=SPEC["input_noise_fraction"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite source objective")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        candidate(model, epoch)
    write_json(
        folder / "complete.json",
        dict(
            completed_at_utc=datetime.now(UTC).isoformat(),
            protocol_sha256=sha256(OUTPUT / "protocol.json"),
            morph_sha256=sha256(folder / "morph.pt"),
            original_observer_sha256=sha256(
                ORIGINAL_OUTPUT / "folds" / test / f"seed{seed}" / "observer.pt"
            ),
            training_records=len(training),
            validation_records=len(validation),
            device=str(device),
            gpu=torch.cuda.get_device_name(device),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),
            elapsed_seconds=time.perf_counter() - tick,
        ),
    )


def freeze():
    checked()
    if (OUTPUT / "locked.json").exists():
        raise ValueError("Already frozen")
    models = {}
    for test in SUBJECTS[1:]:
        models[test] = {}
        for seed in SPEC["seeds"]:
            folder = OUTPUT / "folds" / test / f"seed{seed}"
            entry = json.loads((folder / "complete.json").read_text())
            if entry["protocol_sha256"] != sha256(OUTPUT / "protocol.json"):
                raise ValueError("Training protocol changed")
            if entry["morph_sha256"] != sha256(folder / "morph.pt"):
                raise ValueError("Selected weights changed")
            models[test][str(seed)] = entry
    write_json(
        OUTPUT / "locked.json",
        dict(
            frozen_at_utc=datetime.now(UTC).isoformat(),
            protocol_sha256=sha256(OUTPUT / "protocol.json"),
            models=models,
        ),
    )


def load_models(test):
    checked()
    locked = json.loads((OUTPUT / "locked.json").read_text())
    if locked["protocol_sha256"] != sha256(OUTPUT / "protocol.json"):
        raise ValueError("Frozen protocol changed")
    original = original_models(test)
    result = []
    for seed in SPEC["seeds"]:
        path = OUTPUT / "folds" / test / f"seed{seed}" / "morph.pt"
        if sha256(path) != locked["models"][test][str(seed)]["morph_sha256"]:
            raise ValueError("Frozen weights changed")
        observer_path = ORIGINAL_OUTPUT / "folds" / test / f"seed{seed}" / "observer.pt"
        if sha256(observer_path) != locked["models"][test][str(seed)]["original_observer_sha256"]:
            raise ValueError("Frozen observer changed")
        saved = torch.load(path, weights_only=True, map_location="cpu")
        morph = GeometricMorph(**saved["spec"])
        morph.load_state_dict(saved["model"])
        result.append((morph.eval(), original[seed][1]))
    return result
