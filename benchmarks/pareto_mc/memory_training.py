"""C4 retraining on sealed memory-system trajectories and retained L1 data."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import OUTPUT, TRAIN, VALIDATION, now, records, sha256, stub, write_new
from .experiment import checked
from .memory_experiment import SEEDS, objective_value, read
from .models import SpikeLikelihood
from .train_likelihood import EPOCHS, labelled, load_model, predict_logits, quick_scores


def paths(root, index, phase):
    folders = [OUTPUT / "L1/open/features"]
    folders += [OUTPUT / f"L1/all_feedback/round{k}/rollout/events" for k in (1, 2, 3)]
    previous = read(root / "protocol.json")["spec"]["reference"]
    if "run_root" in previous:
        folders += [OUTPUT / previous["run_root"] / f"round{k}/rollout/events" for k in (1, 2, 3)]
    folders += [root / f"round{k}/rollout/events" for k in range(1, index + 1)]
    return [
        (folder / f"{stub(*case)}.npz", case)
        for folder in folders
        for case in records(TRAIN if phase == "training" else VALIDATION)
    ]


def verify_rollouts(root, index):
    folders = [root / f"round{k}/rollout" for k in range(1, index + 1)]
    folders += [OUTPUT / f"L1/all_feedback/round{k}/rollout" for k in (1, 2, 3)]
    previous = read(root / "protocol.json")["spec"]["reference"]
    if "run_root" in previous:
        folders += [OUTPUT / previous["run_root"] / f"round{k}/rollout" for k in (1, 2, 3)]
    for folder in folders:
        for name, digest in read(folder / "sealed.json")["files"].items():
            if sha256(folder / name) != digest:
                raise ValueError("DAgger input changed after its prediction seal")


def fit(root, index, seed):
    folder = root / f"round{index}/seed{seed}"
    if (folder / "complete.json").exists():
        if sha256(folder / "model.pt") != read(folder / "complete.json")["checkpoint_sha256"]:
            raise ValueError("Completed memory classifier changed")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Use the authorized CUDA training environment")
    torch.manual_seed(seed)
    model = SpikeLikelihood().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    training = paths(root, index, "training")
    validation = [
        labelled(path, case, "validation") for path, case in paths(root, index, "validation")[-20:]
    ]
    rng = np.random.default_rng(seed)
    history, best = [], float("inf")
    folder.mkdir(parents=True, exist_ok=True)
    for epoch in range(EPOCHS):
        losses = []
        model.train()
        for order in rng.permutation(len(training)):
            path, case = training[order]
            events = labelled(path, case, "training")
            keep = events["keep"]
            features = torch.from_numpy(events["features"][keep]).cuda()
            primary = torch.from_numpy(events["primary"][keep]).cuda()
            target = torch.from_numpy(events["target"][keep]).cuda()
            unit = events["unit"][keep]
            counts = np.bincount(unit, minlength=events["record"]["calibration"].spikes.shape[1])
            weights = torch.from_numpy(
                (len(unit) / (len(counts) * counts[unit].clip(1))).astype(np.float32)
            ).cuda()
            permutation = torch.randperm(len(target), device="cuda")
            for begin in range(0, len(target), 16384):
                ids = permutation[begin : begin + 16384]
                logits = model(features[ids] + torch.randn_like(features[ids]) * 0.02, primary[ids])
                loss = (
                    F.binary_cross_entropy_with_logits(logits, target[ids], reduction="none")
                    * weights[ids]
                ).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        values = []
        for events in validation:
            logits = predict_logits(model, events, "cuda")
            keep = events["keep"]
            loss = np.logaddexp(0, logits[keep]) - logits[keep] * events["target"][keep]
            values.extend(
                float(loss[events["unit"][keep] == j].mean())
                for j in range(events["record"]["calibration"].spikes.shape[1])
            )
        validation_loss = float(np.mean(values))
        history.append(
            dict(epoch=epoch, train_bce=float(np.mean(losses)), validation_bce=validation_loss)
        )
        if validation_loss < best:
            best = validation_loss
            torch.save(
                dict(model=model.cpu().state_dict(), spec=model.spec, epoch=epoch),
                folder / "model.pt",
            )
            model.cuda()
        print("memory C4", root.name, index, seed, history[-1], flush=True)
    write_new(folder / "history.json", history)
    write_new(
        folder / "complete.json",
        dict(
            completed_at_utc=now(),
            checkpoint_sha256=sha256(folder / "model.pt"),
            validation_bce=best,
            training_trajectories=len(training),
            epochs=EPOCHS,
            seed=seed,
            gpu=torch.cuda.get_device_name(),
        ),
    )


def select(root, index):
    import optuna

    folder = root / f"round{index}"
    if (folder / "selected.json").exists():
        return
    validation = [
        labelled(path, case, "validation") for path, case in paths(root, index, "validation")[-20:]
    ]
    models = [load_model(folder / f"seed{seed}/model.pt") for seed in SEEDS]
    logits = [[predict_logits(model, events) for events in validation] for model in models]
    baseline_precision = read(OUTPUT / "L0/validation/summary.json")["precision_pct"]

    def measure(seed, threshold, strength):
        metrics = quick_scores(validation, logits[seed], threshold, strength)
        return objective_value(metrics, baseline_precision), metrics

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2026),
        storage=f"sqlite:///{(folder / 'tpe.sqlite3').as_posix()}",
        study_name="memory_C4",
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
    checkpoint = folder / f"seed{winner['selected_seed']}/model.pt"
    write_new(
        folder / "selected.json",
        dict(
            selected_at_utc=now(),
            repeats=repeats,
            selection=winner,
            checkpoint=checkpoint.relative_to(OUTPUT).as_posix(),
            checkpoint_sha256=sha256(checkpoint),
            scope="post-calibration validation Jaccard; final gate uses original evaluator",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("round", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checked(args.root)
    verify_rollouts(args.root, args.round)
    for seed in SEEDS:
        fit(args.root, args.round, seed)
    select(args.root, args.round)
