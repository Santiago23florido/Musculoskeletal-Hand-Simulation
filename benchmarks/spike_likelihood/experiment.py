import json
import shutil
import time
from datetime import UTC, datetime

import numpy as np
import torch
from torch.nn import functional as F

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
from .data import decode, labels
from .model import SpikeLikelihood

SPEC = dict(
    seeds=[0, 1],
    epochs=10,
    learning_rate=0.001,
    weight_decay=0.01,
    batch_size=16384,
    features=19,
    hidden=8,
    logit_bound=4.0,
    training_device="cuda",
    precision="float32; no AMP/TF32",
    loss="unit-balanced natural-prior BCE on true event labels; no class reweighting",
    thresholds=[0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
    ensemble_strengths=[0.0, 0.5, 1.0],
    selection=(
        "validation mean unit Jaccard at calibration-only lags every two epochs; identity candidate"
    ),
    calibration_seconds=5,
    reserved_conditions=["triangular-ramp20s", "triangular-ramp5s"],
    training_conditions=TRAIN_CONDITIONS,
    training_snrs=[30, 20],
    validation_snrs=[25],
    angular_inputs=False,
    test_time_gradients=False,
    primary="posterior",
    control="rescue_union",
)


def register():
    if (OUTPUT / "protocol.json").exists():
        raise ValueError("Preserve the registered run")
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
            folds={s: split(s) for s in SUBJECTS[1:]},
            exposure=(
                "P1 already explored; hypothesis prompted by earlier failures; exploratory transfer"
            ),
            supervision="post-calibration true firing labels on training/validation subjects only",
            architecture=(
                "169 shared parameters; bounded residual event log odds; no unit/subject IDs"
            ),
            test="80 outer records; all 160 new predictions saved before scoring",
        ),
    )
    for relative in source_hashes():
        target = OUTPUT / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def checked():
    p = json.loads((OUTPUT / "protocol.json").read_text())
    if p["spec"] != SPEC or p["source_sha256"] != source_hashes():
        raise ValueError("Code/spec changed after registration")
    return p


def records(test, phase):
    return [
        labels(s, c, n, outer=test, phase=phase)
        for s in split(test)[phase]
        for c in TRAIN_CONDITIONS
        for n in ([30, 20] if phase == "training" else [25])
    ]


def jaccard(truth, prediction):
    values = []
    for j, ref in enumerate(truth):
        predicted = np.flatnonzero(prediction[:, j])
        i = k = tp = 0
        while i < len(ref) and k < len(predicted):
            difference = int(predicted[k]) - int(ref[i])
            if abs(difference) <= 4:
                tp += 1
                i += 1
                k += 1
            elif difference < 0:
                k += 1
            else:
                i += 1
        values.append(tp / (len(ref) + len(predicted) - tp) if len(ref) + len(predicted) else 0.0)
    return values


@torch.no_grad()
def logits(model, record, device):
    result = []
    for start in range(0, len(record["time"]), 65536):
        z = torch.from_numpy(record["features"][start : start + 65536]).to(device)
        p = torch.from_numpy(record["primary"][start : start + 65536]).to(device)
        result.append(model(z, p).cpu().numpy())
    return np.concatenate(result)


def choose(probabilities, validation, strengths):
    best = None
    for strength in strengths:
        p_all = []
        for output, record in zip(probabilities, validation, strict=True):
            prior = (record["primary"].astype(np.float32) * 2 - 1) * 2.1972245773362196
            z = prior + strength * (output - prior)
            p_all.append(1 / (1 + np.exp(-z)))
        for threshold in SPEC["thresholds"]:
            values = []
            for p, r in zip(p_all, validation, strict=True):
                values.extend(jaccard(r["truth"], decode(r, p, threshold)))
            entry = dict(
                strength=strength, threshold=threshold, validation_jaccard=float(np.mean(values))
            )
            priority = (entry["validation_jaccard"], -strength, -abs(threshold - 0.5))
            if best is None or priority > best[0]:
                best = (priority, entry)
    return best[1]


def train(test, seed):
    checked()
    if test not in SUBJECTS[1:] or seed not in SPEC["seeds"] or (OUTPUT / "locked.json").exists():
        raise ValueError("Undeclared or frozen training")
    folder = OUTPUT / "folds" / test / f"seed{seed}"
    if (folder / "complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    device = training_device(seed)
    training, validation = records(test, "training"), records(test, "validation")
    z, p, y, w = [], [], [], []
    unit_count = 0
    for r in training:
        use = r["time"] >= 10240
        z.append(r["features"][use])
        p.append(r["primary"][use])
        y.append(r["target"][use])
        weight = np.empty(use.sum(), dtype=np.float32)
        u = r["unit"][use]
        for j in range(r["m"]):
            mask = u == j
            weight[mask] = 1 / max(int(mask.sum()), 1)
            unit_count += 1
        w.append(weight)
    arrays = [np.concatenate(a) for a in (z, p, y, w)]
    arrays[3] *= len(arrays[0]) / unit_count
    features, primary, target, weight = [torch.from_numpy(a).to(device) for a in arrays]
    del training, z, p, y, w, arrays
    model = SpikeLikelihood().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SPEC["learning_rate"], weight_decay=SPEC["weight_decay"]
    )
    best = -1.0
    history = []
    for epoch in range(SPEC["epochs"] + 1):
        if epoch:
            model.train()
            order = torch.randperm(len(features), device=device)
            for batch in order.split(SPEC["batch_size"]):
                noise = torch.randn_like(features[batch]) * 0.02
                output = model(features[batch] + noise, primary[batch])
                loss = (
                    F.binary_cross_entropy_with_logits(output, target[batch], reduction="none")
                    * weight[batch]
                ).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite event likelihood")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
        if epoch % 2:
            continue
        model.eval()
        selection = choose([logits(model, r, device) for r in validation], validation, [1.0])
        history.append(dict(epoch=epoch, **selection))
        if selection["validation_jaccard"] > best:
            best = selection["validation_jaccard"]
            torch.save(
                dict(model=model.state_dict(), spec=model.spec, epoch=epoch, selection=selection),
                folder / "model.pt",
            )
        write_json(folder / "history.json", history)
        print(test, seed, epoch, selection, flush=True)
    write_json(
        folder / "complete.json",
        dict(
            completed_at_utc=datetime.now(UTC).isoformat(),
            protocol_sha256=sha256(OUTPUT / "protocol.json"),
            model_sha256=sha256(folder / "model.pt"),
            device=str(device),
            gpu=torch.cuda.get_device_name(device),
            torch_version=torch.__version__,
            training_events=len(features),
            training_units=unit_count,
            elapsed_seconds=time.perf_counter() - tick,
        ),
    )


def unfrozen_models(test):
    result = []
    for seed in SPEC["seeds"]:
        saved = torch.load(
            OUTPUT / "folds" / test / f"seed{seed}" / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        model = SpikeLikelihood(**saved["spec"])
        model.load_state_dict(saved["model"])
        result.append(model.eval())
    return result


def freeze():
    checked()
    if (OUTPUT / "locked.json").exists():
        raise ValueError("Already frozen")
    result = {}
    for test in SUBJECTS[1:]:
        entries = {}
        for seed in SPEC["seeds"]:
            folder = OUTPUT / "folds" / test / f"seed{seed}"
            entry = json.loads((folder / "complete.json").read_text())
            if entry["protocol_sha256"] != sha256(OUTPUT / "protocol.json") or entry[
                "model_sha256"
            ] != sha256(folder / "model.pt"):
                raise ValueError("Selected weights or protocol changed")
            entries[str(seed)] = entry
        models = unfrozen_models(test)
        validation = records(test, "validation")
        values = [np.mean([logits(m, r, "cpu") for m in models], axis=0) for r in validation]
        selection = choose(values, validation, SPEC["ensemble_strengths"])
        result[test] = dict(models=entries, selection=selection)
        print("ensemble", test, selection, flush=True)
    write_json(
        OUTPUT / "locked.json",
        dict(
            frozen_at_utc=datetime.now(UTC).isoformat(),
            protocol_sha256=sha256(OUTPUT / "protocol.json"),
            folds=result,
        ),
    )


def load_models(test):
    checked()
    lock = json.loads((OUTPUT / "locked.json").read_text())
    if lock["protocol_sha256"] != sha256(OUTPUT / "protocol.json"):
        raise ValueError("Frozen protocol changed")
    for seed in SPEC["seeds"]:
        if (
            sha256(OUTPUT / "folds" / test / f"seed{seed}" / "model.pt")
            != lock["folds"][test]["models"][str(seed)]["model_sha256"]
        ):
            raise ValueError("Frozen weights changed")
    return unfrozen_models(test), lock["folds"][test]["selection"]
