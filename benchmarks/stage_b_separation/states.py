"""Covariance/trajectory states from allowed training or validation subjects only."""

import json

import torch

from benchmarks.stage_b.common import OUTPUT as ORIGINAL_OUTPUT
from benchmarks.stage_b.data import prepare, supervision
from benchmarks.stage_b.experiment import training_device

from .common import OUTPUT, SUBJECTS, TRAIN_CONDITIONS, sha256, split, stub, write_json


def guarded(subject, condition, snr, outer, phase):
    if phase not in ("training", "validation"):
        raise ValueError("Unknown supervision phase")
    if subject not in split(outer)[phase]:
        raise ValueError("Outer subject labels cannot enter separation fitting")
    if condition not in TRAIN_CONDITIONS or snr not in ([30, 20] if phase == "training" else [25]):
        raise ValueError("Reserved movement/noise cannot enter separation fitting")


@torch.no_grad()
def prepare_state(subject, condition, snr, *, outer, phase):
    guarded(subject, condition, snr, outer, phase)
    folder = OUTPUT / "states" / stub(subject, condition, snr)
    if (folder / "metadata.json").exists():
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["source_sha256"] != sha256(__file__):
            raise ValueError("Preserve cached states from the previous implementation")
        if metadata["case_sha256"] != sha256(folder / "case.pt"):
            raise ValueError("Training states changed")
        return torch.load(folder / "case.pt", map_location="cpu", weights_only=True)
    labels = supervision(subject, condition, snr, outer=outer, phase=phase)
    record = prepare(subject, condition, snr)
    device = training_device(0)
    x = record["x"].to(device)
    batch = record["config"].batch_size
    centers, targets, covariances, counts = [], [], [], []
    for index in range(9):
        selected = (labels["q"] * 8).round().long() == index
        order = labels["unit"][selected].argsort()
        units = labels["unit"][selected][order]
        if not torch.equal(units, torch.arange(len(record["template"]))):
            raise ValueError("Each training state needs targets for every calibrated identity")
        targets.append(labels["target"][selected][order])
        centers.append(labels["q"][selected].mean())
        mask = (record["q"] - index / 8).abs() <= 1 / 16 + 1e-5
        sample_mask = mask.repeat_interleave(batch)[: len(x)].to(device)
        samples = x[sample_mask]
        if len(samples) < 1024:
            raise ValueError("Insufficient observations for a state covariance")
        samples = samples - samples.mean(0)
        covariances.append((samples.T @ samples / len(samples)).cpu())
        counts.append(len(samples))
    case = dict(
        cal=record["template"],
        target=torch.stack(targets),
        q=torch.stack(centers),
        covariance=torch.stack(covariances),
        projection=record["projection"],
        amplitude=record["amplitude"],
        rate=torch.from_numpy(record["calibration"].spikes).float().mean(0),
    )
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(case, folder / "case.pt")
    write_json(
        folder / "metadata.json",
        dict(
            subject=subject,
            condition=condition,
            snr_db=snr,
            source_sha256=sha256(__file__),
            case_sha256=sha256(folder / "case.pt"),
            samples_per_state=counts,
            original_feature_metadata_sha256=sha256(
                ORIGINAL_OUTPUT / "features" / stub(subject, condition, snr) / "metadata.json"
            ),
            outer_labels_used=False,
        ),
    )
    return case


def prepare_subject(subject):
    if subject not in SUBJECTS:
        raise ValueError("Unknown subject")
    training = next(s for s in SUBJECTS[1:] if subject in split(s)["training"])
    validation = next((s for s in SUBJECTS[1:] if subject in split(s)["validation"]), None)
    for condition in TRAIN_CONDITIONS:
        for snr in (30, 20):
            prepare_state(subject, condition, snr, outer=training, phase="training")
            print("state prepared", subject, condition, snr, flush=True)
        if validation:
            prepare_state(subject, condition, 25, outer=validation, phase="validation")
            print("state prepared", subject, condition, 25, flush=True)
