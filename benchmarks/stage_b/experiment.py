"""Estimated-trajectory supervision, privileged training and frozen outer evaluation."""

import json
import os
import random
import shutil
import time
from datetime import UTC, datetime

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
from .data import supervision
from .model import BlindState, MorphNet, template_objective

SPEC = dict(
    seeds=[0, 1],
    training_device="cuda",
    training_precision="float32; no AMP; TF32 disabled",
    training_determinism="seeded CUDA/CPU; deterministic algorithms; cuBLAS :4096:8",
    inference_device="cpu",
    morph_epochs=20,
    state_epochs=30,
    morph_batch_size=16,
    morph_learning_rate=0.001,
    state_learning_rate=0.003,
    weight_decay=0.001,
    morph_channels=8,
    morph_relative_bound=2.0,
    blind_features=120,
    blind_hidden=24,
    state_dimension=1,
    pca_dimension=512,
    interpolation_knots=129,
    online_shrinkage_ablation=0.0,
    fifo_batches=10,
    covariance_shrinkage=0.5,
    covariance_ridge=0.001,
    template_samples=64,
    calibration_seconds=5,
    training_conditions=TRAIN_CONDITIONS,
    training_snrs=[30, 20],
    validation_snrs=[25],
    supervised_targets="clean EMG GT-triggered averages by pose; estimated, not generator truth",
    morph_loss="normalized waveform error + 0.1 normalized temporal derivative error + size prior",
    state_loss="angle MSE + 0.1 derivative MSE; random circular training sequence start",
    calibration_input_noise_fraction=0.03,
    state_input_noise_std=0.02,
    selection="validation waveform objective for morph; validation angle MSE for blind observer",
    deployment="mean templates of both seeds; same morph weights in K and B",
    primary_method="moca_b",
    kinematic_variant="moca_k",
    control="static_analytic",
    neural_parameters_use_subject_or_unit_ids=False,
    optional_peel_off=False,
    test_time_gradient_updates=False,
    success=dict(
        overall_mean_roa_pct=85,
        ramp5s_mean_roa_pct=80,
        units_roa_ge_90_pct=70,
        cpu_batch_ms=22,
        maximum_degradation_8_to_40_degrees_per_second_pp=10,
    ),
    incomplete_tests=[
        "independent new data",
        "1s and 2.5s true-generator ramps",
        "full raw-stream CPU latency including preprocessing/features",
    ],
)


def register():
    path = OUTPUT / "protocol.json"
    if path.exists():
        raise ValueError("Protocol already exists; preserve it")
    write_json(
        path,
        dict(
            registered_at_utc=datetime.now(UTC).isoformat(),
            spec=SPEC,
            source_sha256=source_hashes(),
            guide_sha256=(
                sha256(ROOT / "MUAP_dynamic_DL_research_report.pdf")
                if (ROOT / "MUAP_dynamic_DL_research_report.pdf").is_file()
                else None
            ),
            folds={s: split(s) for s in SUBJECTS[1:]},
            prior_exposure=(
                "All P1 outcomes were previously seen; this is exploratory grouped transfer"
            ),
            pilot="Analytic controls on sub-01 training data only, before registration",
            supervision_change="Uses training/validation true firings; unlike label-free Stage A",
            test="80 outer records, same 591 calibrated instances and original Pareto scorer",
            deviations_from_guide=[
                "PCA512 covariance approximation, with captured calibration variance reported",
                "129-knot frozen morph table precomputed from calibration; interpolated online",
                "Estimated MUAP supervision; no original generator MUAP arrays were distributed",
                "K state is directly observed angle, B is a trained causal scalar GRU observer",
                "Supervised waveform branch; no full EMG reconstruction/spike-BCE fine-tuning",
                "Faster true-generator conditions and raw-stream timing remain untested",
            ],
        ),
    )
    for relative in source_hashes():
        target = OUTPUT / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def checked():
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    if protocol["spec"] != SPEC or protocol["source_sha256"] != source_hashes():
        raise ValueError("Code/specification changed after registration; start a new experiment")
    return protocol


def training_records(test, phase):
    partition = split(test)
    subjects = partition["training"] if phase == "training" else partition["validation"]
    snrs = [30, 20] if phase == "training" else [25]
    return [
        supervision(s, c, n, outer=test, phase=phase)
        for s in subjects
        for c in TRAIN_CONDITIONS
        for n in snrs
    ]


def combine(records, device="cpu"):
    dataset = dict(
        cal=torch.cat([r["cal"][r["unit"]] for r in records]),
        q=torch.cat([r["q"] for r in records]),
        target=torch.cat([r["target"] for r in records]),
    )
    return {key: value.to(device) for key, value in dataset.items()}


def training_device(seed):
    """Require the registered accelerator; never silently fall back to CPU."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("Registered CUDA determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage B training requires CUDA; use .venv-stage-b-gpu")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device(SPEC["training_device"])


@torch.no_grad()
def validate_morph(model, dataset):
    values = []
    for start in range(0, len(dataset["q"]), 32):
        window = slice(start, start + 32)
        predicted = model(dataset["cal"][window], dataset["q"][window])
        values.append(
            (
                len(predicted),
                float(
                    template_objective(
                        predicted,
                        dataset["target"][window],
                        dataset["q"][window],
                        calibration=dataset["cal"][window],
                    )
                ),
            )
        )
    return sum(n * value for n, value in values) / sum(n for n, _ in values)


@torch.no_grad()
def validate_state(model, records):
    return float(
        np.mean(
            [
                float((model(r["features"][None])[0][0] - r["angle"]).square().mean())
                for r in records
            ]
        )
    )


def train(test, seed):
    checked()
    if seed not in SPEC["seeds"] or (OUTPUT / "locked.json").exists():
        raise ValueError("Undeclared seed or frozen experiment")
    folder = OUTPUT / "folds" / test / f"seed{seed}"
    if (folder / "complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    device = training_device(seed)
    print("training device:", torch.cuda.get_device_name(device), torch.__version__, flush=True)
    training = training_records(test, "training")
    validation = training_records(test, "validation")
    train_mu, valid_mu = combine(training, device), combine(validation, device)
    for record in training + validation:
        for key in ("features", "angle"):
            record[key] = record[key].to(device)
    morph = MorphNet(channels=SPEC["morph_channels"], bound=SPEC["morph_relative_bound"]).to(device)
    observer = BlindState(features=SPEC["blind_features"], hidden=SPEC["blind_hidden"]).to(device)
    history = dict(morph=[], state=[])
    best_morph, best_state = float("inf"), float("inf")
    optimizer = torch.optim.AdamW(
        morph.parameters(), lr=SPEC["morph_learning_rate"], weight_decay=SPEC["weight_decay"]
    )
    for epoch in range(SPEC["morph_epochs"] + 1):
        morph.train()
        if epoch:
            order = torch.randperm(len(train_mu["q"]), device=device)
            for batch in order.split(SPEC["morph_batch_size"]):
                template, q = train_mu["cal"][batch], train_mu["q"][batch]
                noise = torch.randn_like(template) * (
                    SPEC["calibration_input_noise_fraction"] / template[0].numel() ** 0.5
                )
                # Perturb only the conditioning observation; target stays fixed.
                prediction = morph(template + noise, q)
                loss = template_objective(
                    prediction, train_mu["target"][batch], q, calibration=template
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite morph objective")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(morph.parameters(), 1)
                optimizer.step()
        morph.eval()
        value = validate_morph(morph, valid_mu)
        history["morph"].append(dict(epoch=epoch, validation_objective=value))
        if value < best_morph:
            best_morph = value
            torch.save(
                dict(
                    spec=morph.spec,
                    model=morph.state_dict(),
                    epoch=epoch,
                    validation_objective=value,
                ),
                folder / "morph.pt",
            )
        write_json(folder / "history.json", history)
        print(test, seed, "morph", epoch, value, flush=True)
    optimizer = torch.optim.AdamW(
        observer.parameters(), lr=SPEC["state_learning_rate"], weight_decay=SPEC["weight_decay"]
    )
    for epoch in range(SPEC["state_epochs"] + 1):
        observer.train()
        if epoch:
            order = torch.randperm(len(training)).tolist()
            for index in order:
                record = training[index]
                shift = random.randrange(len(record["angle"]))
                features = record["features"].roll(shift, 0)
                target = record["angle"].roll(shift, 0)
                features = features + SPEC["state_input_noise_std"] * torch.randn_like(features)
                prediction, _ = observer(features[None])
                prediction = prediction[0]
                derivative = prediction[1:] - prediction[:-1] - target[1:] + target[:-1]
                loss = (prediction - target).square().mean() + 0.1 * derivative.square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite blind-state objective")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(observer.parameters(), 1)
                optimizer.step()
        observer.eval()
        value = validate_state(observer, validation)
        history["state"].append(dict(epoch=epoch, validation_angle_mse=value))
        if value < best_state:
            best_state = value
            torch.save(
                dict(
                    spec=observer.spec,
                    model=observer.state_dict(),
                    epoch=epoch,
                    validation_angle_mse=value,
                ),
                folder / "observer.pt",
            )
        write_json(folder / "history.json", history)
        print(test, seed, "state", epoch, value, flush=True)
    write_json(
        folder / "complete.json",
        dict(
            completed_at_utc=datetime.now(UTC).isoformat(),
            protocol_sha256=sha256(OUTPUT / "protocol.json"),
            morph_sha256=sha256(folder / "morph.pt"),
            observer_sha256=sha256(folder / "observer.pt"),
            training_pairs=len(train_mu["q"]),
            validation_pairs=len(valid_mu["q"]),
            device=str(device),
            gpu=torch.cuda.get_device_name(device),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
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
            for name in ("morph", "observer"):
                if entry[f"{name}_sha256"] != sha256(folder / f"{name}.pt"):
                    raise ValueError("Selected model changed")
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
    result = []
    for seed in SPEC["seeds"]:
        models = []
        for name, factory in (("morph", MorphNet), ("observer", BlindState)):
            path = OUTPUT / "folds" / test / f"seed{seed}" / f"{name}.pt"
            if sha256(path) != locked["models"][test][str(seed)][f"{name}_sha256"]:
                raise ValueError("Frozen model changed")
            saved = torch.load(path, weights_only=True, map_location="cpu")
            model = factory(**saved["spec"])
            model.load_state_dict(saved["model"])
            model.eval()
            models.append(model)
        result.append(tuple(models))
    return result
