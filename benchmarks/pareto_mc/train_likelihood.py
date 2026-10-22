"""Unit-balanced AdamW, validation-loss checkpoints, TPE and three-seed repeats."""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from benchmarks.spike_likelihood.data import batch_peaks
from benchmarks.spike_likelihood.experiment import jaccard

from .common import OUTPUT, TRAIN, VALIDATION, now, records, sha256, stub, write_new
from .events import descriptor, label_events
from .models import SpikeLikelihood

EPOCHS = 10
SEEDS = (0, 1, 2)


def dataset_paths(variant, round_index, phase):
    subjects = TRAIN if phase == "training" else VALIDATION
    folders = [OUTPUT / "L1/open/features"]
    folders += [
        OUTPUT / "L1" / variant / f"round{k}/rollout/events" for k in range(1, round_index + 1)
    ]
    return [
        (folder / f"{stub(*case)}.npz", case) for folder in folders for case in records(subjects)
    ]


def _labelled_uncached(path, case, phase):
    with np.load(path) as archive:
        events = {key: archive[key] for key in archive.files}
    record = descriptor(*case, phase=phase)
    label, truth, lags = label_events(events, record, phase=phase)
    keep = events["time"] >= 5 * record["config"].fs
    events.update(target=label, truth=truth, lags=lags, keep=keep, record=record)
    return events


def labelled(path, case, phase):
    from .training_cache import cached_labelled

    return cached_labelled(path, case, phase, _labelled_uncached)


def load_model(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = SpikeLikelihood(**saved["spec"])
    model.load_state_dict(saved["model"])
    return model.eval()


@torch.no_grad()
def predict_logits(model, events, device="cpu"):
    parts = []
    for start in range(0, len(events["time"]), 65536):
        features = torch.from_numpy(events["features"][start : start + 65536]).to(device)
        primary = torch.from_numpy(events["primary"][start : start + 65536]).to(device)
        parts.append(model(features, primary).cpu().numpy())
    return np.concatenate(parts)


def fit_seed(variant, round_index, seed):
    folder = OUTPUT / "L1" / variant / f"round{round_index}" / f"seed{seed}"
    if (folder / "complete.json").exists():
        completed = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
        if sha256(folder / "model.pt") != completed["checkpoint_sha256"]:
            raise ValueError("Completed classifier checkpoint changed")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("C4 training requires the authorized CUDA environment")
    torch.manual_seed(seed)
    model = SpikeLikelihood().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    paths = dataset_paths(variant, round_index, "training")
    validation = [
        labelled(path, case, "validation")
        for path, case in dataset_paths(variant, round_index, "validation")[-20:]
    ]
    rng = np.random.default_rng(seed)
    history, best = [], float("inf")
    folder.mkdir(parents=True, exist_ok=True)
    for epoch in range(EPOCHS):
        losses = []
        model.train()
        for order in rng.permutation(len(paths)):
            path, case = paths[order]
            events = labelled(path, case, "training")
            keep = events["keep"]
            features = torch.from_numpy(events["features"][keep]).cuda()
            primary = torch.from_numpy(events["primary"][keep]).cuda()
            target = torch.from_numpy(events["target"][keep]).cuda()
            unit = events["unit"][keep]
            counts = np.bincount(unit, minlength=events["record"]["calibration"].spikes.shape[1])
            weight = torch.from_numpy(
                (len(unit) / (len(counts) * counts[unit].clip(1))).astype(np.float32)
            ).cuda()
            permutation = torch.randperm(len(target), device="cuda")
            for begin in range(0, len(target), 16384):
                ids = permutation[begin : begin + 16384]
                z = features[ids] + torch.randn_like(features[ids]) * 0.02
                output = model(z, primary[ids])
                loss = (
                    F.binary_cross_entropy_with_logits(output, target[ids], reduction="none")
                    * weight[ids]
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
            # Unit balanced validation BCE; no threshold or test feedback.
            loss = np.logaddexp(0, logits[keep]) - logits[keep] * events["target"][keep]
            values.extend(
                [
                    float(loss[events["unit"][keep] == j].mean())
                    for j in range(events["record"]["calibration"].spikes.shape[1])
                ]
            )
        score = float(np.mean(values))
        history.append(dict(epoch=epoch, train_bce=float(np.mean(losses)), validation_bce=score))
        if score < best:
            best = score
            torch.save(
                dict(model=model.cpu().state_dict(), spec=model.spec, epoch=epoch),
                folder / "model.pt",
            )
            model.cuda()
        print("C4", variant, round_index, seed, history[-1], flush=True)
    write_new(folder / "history.json", history)
    write_new(
        folder / "complete.json",
        dict(
            completed_at_utc=now(),
            checkpoint_sha256=sha256(folder / "model.pt"),
            validation_bce=best,
            training_trajectories=len(paths),
            epochs=EPOCHS,
            seed=seed,
            gpu=torch.cuda.get_device_name(),
        ),
    )


def decode_record(events, logits, threshold, strength):
    record = events["record"]
    cfg = record["config"]
    prior = (events["primary"].astype(np.float32) * 2 - 1) * np.log(9)
    probability = 1 / (1 + np.exp(-(prior + strength * (logits - prior))))
    n, m = len(record["raw"]), record["calibration"].spikes.shape[1]
    binary = np.zeros((n, m), dtype=np.uint8)
    for j in range(m):
        keep = events["unit"] == j
        values = np.full(n, -np.inf, dtype=np.float32)
        values[events["time"][keep]] = probability[keep]
        positions = batch_peaks(values, cfg.batch_size, cfg.spike_min_dist, threshold)
        binary[positions, j] = 1
    binary[: len(events["initial"])] = events["initial"]
    return binary


def quick_scores(validation, logits, threshold, strength):
    scores, ramp, precision = [], [], []
    for events, values in zip(validation, logits, strict=True):
        binary = decode_record(events, values, threshold, strength)
        binary[: 5 * events["record"]["config"].fs] = 0
        roa = jaccard(events["truth"], binary)
        scores.extend(roa)
        for j, value in enumerate(roa):
            nr, npred = len(events["truth"][j]), int(binary[:, j].sum())
            tp = value * (nr + npred) / (1 + value)
            precision.append(tp / npred if npred else 0.0)
        if events["record"]["condition"] == "triangular-ramp5s":
            ramp.extend(roa)
    return dict(
        global_roa_pct=float(np.mean(scores) * 100),
        ramp5s_roa_pct=float(np.mean(ramp) * 100),
        precision_pct=float(np.mean(precision) * 100),
    )


def select(variant, round_index):
    import optuna

    folder = OUTPUT / "L1" / variant / f"round{round_index}"
    if (folder / "selected.json").exists():
        return
    # Select on the most recently visited validation states, never on training loss.
    validation = [
        labelled(path, case, "validation")
        for path, case in dataset_paths(variant, round_index, "validation")[-20:]
    ]
    models = [load_model(folder / f"seed{seed}/model.pt") for seed in SEEDS]
    values = [[predict_logits(model, events) for events in validation] for model in models]
    prior = [(events["primary"].astype(np.float32) * 2 - 1) * np.log(9) for events in validation]
    baseline = quick_scores(validation, prior, 0.5, 0)

    def measure(seed, threshold, strength):
        metrics = quick_scores(validation, values[seed], threshold, strength)
        penalty = max(0, baseline["precision_pct"] - metrics["precision_pct"] - 1)
        value = 0.5 * (metrics["global_roa_pct"] + metrics["ramp5s_roa_pct"]) - 5 * penalty
        return value, metrics

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2026),
        storage=f"sqlite:///{(folder / 'tpe.sqlite3').as_posix()}",
        study_name="C4",
        load_if_exists=True,
    )

    def objective(trial):
        threshold = trial.suggest_float("threshold", 0.25, 0.85)
        strength = trial.suggest_float("strength", 0.0, 1.0)
        value, metrics = measure(0, threshold, strength)
        trial.set_user_attr("metrics", metrics)
        return value

    study.enqueue_trial(dict(threshold=0.5, strength=0.0))
    study.optimize(objective, n_trials=max(0, 40 - len(study.trials)), catch=(FloatingPointError,))
    completed = [trial for trial in study.trials if trial.value is not None]
    top = sorted(completed, key=lambda trial: trial.value, reverse=True)[:3]
    repeated = []
    for trial in top:
        scores = [measure(seed, **trial.params) for seed in SEEDS]
        selected_seed = int(np.argsort([score[0] for score in scores])[1])
        repeated.append(
            dict(
                trial=trial.number,
                parameters=trial.params,
                seed_metrics=[score[1] for score in scores],
                median_objective=float(np.median([score[0] for score in scores])),
                selected_seed=selected_seed,
            )
        )
    winner = max(repeated, key=lambda row: row["median_objective"])
    checkpoint = folder / f"seed{winner['selected_seed']}/model.pt"
    write_new(
        folder / "selected.json",
        dict(
            selected_at_utc=now(),
            baseline=baseline,
            repeats=repeated,
            selection=winner,
            checkpoint=str(checkpoint.relative_to(OUTPUT)),
            checkpoint_sha256=sha256(checkpoint),
            trials=40,
            training_seeds=SEEDS,
            selection_scope=(
                "post-calibration event Jaccard at calibration-only lags; validation only"
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["all_feedback", "guarded_feedback"])
    parser.add_argument("round", type=int, choices=range(4))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for seed in SEEDS:
        fit_seed(args.variant, args.round, seed)
    select(args.variant, args.round)


if __name__ == "__main__":
    main()
