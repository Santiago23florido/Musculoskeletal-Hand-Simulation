"""P1: open-loop test of C1 posture inputs for C4, before any closed-loop level.

Data, calibration-lag labels, AdamW recipe, seeds, epoch selection and TPE of the
final L2 C4 round are reused unchanged. The only change is two causal inputs
from C1: the blind posture q at the decision batch and its 2 s displacement.
Every head then decides open loop on the sealed candidates of the L2 final run;
decisions are sealed before the original evaluator opens labels. Nothing here is
deployed or changes an accepted level.
"""

import argparse
import math

import numpy as np
import torch
from torch.nn import functional as F

from .common import OUTPUT, VALIDATION, now, records, register, sha256, stub, write_new
from .experiment import checked
from .memory_experiment import SEEDS, objective_value, read
from .models import SpikeLikelihood
from .train_likelihood import EPOCHS, decode_record, labelled, predict_logits, quick_scores

FOLDER = OUTPUT / "P1_posture_c4"
L2_ROOT = OUTPUT / "L2/standard"
FINAL = L2_ROOT / "final_validation"
OPEN = OUTPUT / "L1/open/features"
ARMS = ("base22", "posture24")
LAG_BATCHES = 20  # 2 s at 100 ms batches
BOOTSTRAP = 2000


class PostureLikelihood(SpikeLikelihood):
    """The 22 L2 inputs plus blind posture q and its causal 2 s displacement."""

    def __init__(self, features=24, hidden=8, bound=4.0):
        if features != 24:
            raise ValueError("The posture head has exactly 24 declared inputs")
        super().__init__(features, hidden, bound)

    def forward(self, features, primary):
        if features.shape[-1] != 24:
            raise ValueError("Only the 24 declared features are allowed")
        prior = (primary.float() * 2 - 1) * math.log(9)
        return prior + self.spec["bound"] * self.layers(features.clamp(-6, 6)).squeeze(-1).tanh()


def load_head(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cls = PostureLikelihood if saved["spec"]["features"] == 24 else SpikeLikelihood
    model = cls(**saved["spec"])
    model.load_state_dict(saved["model"])
    return model.eval()


def posture_columns(q, time, batch_size):
    """q and q - q(2 s earlier) at each candidate's own batch; past batches only."""
    q = np.asarray(q, dtype=np.float64)
    displacement = q - q[np.maximum(np.arange(len(q)) - LAG_BATCHES, 0)]
    batch = np.asarray(time) // batch_size
    return np.column_stack([q[batch], displacement[batch]]).astype(np.float32)


def speed_mismatch(features, time, speed, batch_size):
    """Largest deviation between the stored C4 speed input and the batch's C1 speed."""
    expected = np.clip(np.asarray(speed)[np.asarray(time) // batch_size], -6, 6)
    return float(np.max(np.abs(features[:, 21] - expected.astype(np.float32)), initial=0.0))


def l2_paths(phase):
    from .memory_training import paths

    return paths(L2_ROOT, 3, phase)


def source_state(path, case):
    """Per-batch C1 posture and speed exactly as used when the candidates were built."""
    if path.parent == OPEN:
        from .observer import load_observer

        with np.load(OUTPUT / "C1/features" / f"{stub(*case)}.npz") as archive:
            features = torch.from_numpy(archive["features"])
        with torch.no_grad():
            mean, _, _ = load_observer()(features[None])
        q = mean[0].numpy()
        return q, np.r_[0, np.abs(np.diff(q)) / (204 / 2048)]
    with np.load(path.parent.parent / "predictions" / path.name) as archive:
        state = archive["state"]
    return state[:, 0], state[:, 2]


def key(path):
    return path.relative_to(OUTPUT).as_posix().replace("/", "__").removesuffix(".npz")


def prepare():
    """Label-free: store per-batch q for every candidate source and verify alignment."""
    manifest_path = FOLDER / "posture_manifest.json"
    if manifest_path.exists():
        return read(manifest_path)
    from .memory_training import verify_rollouts

    torch.set_num_threads(2)
    verify_rollouts(L2_ROOT, 3)
    sealed = read(FINAL / "sealed.json")["files"]
    used = l2_paths("training") + l2_paths("validation")[-20:]
    sources, cases = [path for path, _ in used], [case for _, case in used]
    for case in records(VALIDATION):
        name = f"events/{stub(*case)}.npz"
        if sha256(FINAL / name) != sealed[name]:
            raise ValueError("Final validation candidates changed after their seal")
        sources.append(FINAL / name)
        cases.append(case)
    entries = {}
    for path, case in zip(sources, cases, strict=True):
        q, speed = source_state(path, case)
        with np.load(path) as archive:
            mismatch = speed_mismatch(archive["features"], archive["time"], speed, 204)
        tolerance = 1e-4 if path.parent == OPEN else 0.0
        if mismatch > tolerance:
            raise AssertionError(f"Candidate/batch misalignment in {path}: {mismatch}")
        destination = FOLDER / "posture" / f"{key(path)}.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            np.save(destination, np.asarray(q, dtype=np.float32))
        entries[path.relative_to(OUTPUT).as_posix()] = dict(
            q_sha256=sha256(destination), speed_max_abs_mismatch=mismatch
        )
    manifest = dict(
        prepared_at_utc=now(),
        labels_opened=False,
        sources=entries,
        alignment="stored C4 speed equals C1 speed at time // 204 (exact for loop rollouts; "
        "open-loop features recomputed with the same observer, tolerance 1e-4)",
    )
    write_new(manifest_path, manifest)
    return manifest


def posture_for(path, events):
    stored = FOLDER / "posture" / f"{key(path)}.npy"
    entry = read(FOLDER / "posture_manifest.json")["sources"][path.relative_to(OUTPUT).as_posix()]
    if sha256(stored) != entry["q_sha256"]:
        raise ValueError("Stored posture changed")
    return posture_columns(np.load(stored), events["time"], 204)


def inputs(arm, path, events):
    if arm == "base22":
        return events["features"]
    return np.concatenate([events["features"], posture_for(path, events)], axis=1)


@torch.no_grad()
def predict(model, arm, path, events, device="cpu"):
    if arm == "base22":
        return predict_logits(model, events, device)
    features = inputs(arm, path, events)
    parts = []
    for start in range(0, len(events["time"]), 65536):
        x = torch.from_numpy(np.ascontiguousarray(features[start : start + 65536])).to(device)
        primary = torch.from_numpy(events["primary"][start : start + 65536]).to(device)
        parts.append(model(x, primary).cpu().numpy())
    return np.concatenate(parts)


def head_path(arm, seed):
    """base22 seeds 1-2 reuse L2 only when seed 0 reproduced L2 bit for bit."""
    own = FOLDER / arm / f"seed{seed}/model.pt"
    if arm == "base22" and seed != 0 and not own.exists():
        if not read(FOLDER / "base22/seed0/reproduction.json")["identical"]:
            raise ValueError("Base seeds must be retrained: seed 0 did not reproduce L2")
        reference = L2_ROOT / f"round3/seed{seed}"
        if sha256(reference / "model.pt") != read(reference / "complete.json")["checkpoint_sha256"]:
            raise ValueError("L2 round-3 checkpoint changed")
        return reference / "model.pt"
    return own


def fit(arm, seed):
    """memory_training.fit, line for line; only the head class and inputs depend on arm."""
    folder = FOLDER / arm / f"seed{seed}"
    if (folder / "complete.json").exists():
        if sha256(folder / "model.pt") != read(folder / "complete.json")["checkpoint_sha256"]:
            raise ValueError("Completed posture classifier changed")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Use the authorized CUDA training environment")
    torch.manual_seed(seed)
    model = (SpikeLikelihood() if arm == "base22" else PostureLikelihood()).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    training = l2_paths("training")
    validation_paths = l2_paths("validation")[-20:]
    validation = [labelled(path, case, "validation") for path, case in validation_paths]
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
            features = torch.from_numpy(inputs(arm, path, events)[keep]).cuda()
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
        for (path, _), events in zip(validation_paths, validation, strict=True):
            logits = predict(model, arm, path, events, "cuda")
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
        print("posture C4", arm, seed, history[-1], flush=True)
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
    if arm == "base22":
        reference = L2_ROOT / f"round3/seed{seed}"
        mine = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        theirs = torch.load(reference / "model.pt", map_location="cpu", weights_only=True)
        identical = mine["epoch"] == theirs["epoch"] and all(
            torch.equal(mine["model"][name], theirs["model"][name]) for name in theirs["model"]
        )
        write_new(
            folder / "reproduction.json",
            dict(
                reference=reference.relative_to(OUTPUT).as_posix(),
                identical=identical,
                history_identical=history == read(reference / "history.json"),
                checkpoint_sha256_equal=sha256(folder / "model.pt")
                == read(reference / "complete.json")["checkpoint_sha256"],
            ),
        )


def select(arm):
    """memory_training.select with the same TPE seed, budget, objective and repeats."""
    import optuna

    folder = FOLDER / arm
    if (folder / "selected.json").exists():
        return read(folder / "selected.json")
    validation_paths = l2_paths("validation")[-20:]
    validation = [labelled(path, case, "validation") for path, case in validation_paths]
    models = [load_head(head_path(arm, seed)) for seed in SEEDS]
    logits = [
        [
            predict(model, arm, path, events)
            for (path, _), events in zip(validation_paths, validation, strict=True)
        ]
        for model in models
    ]
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
    result = dict(
        selected_at_utc=now(),
        repeats=repeats,
        selection=winner,
        heads={seed: head_path(arm, seed).relative_to(OUTPUT).as_posix() for seed in SEEDS},
        scope="L2 round-3 validation candidates, as in L2; final comparison uses other candidates",
    )
    write_new(folder / "selected.json", result)
    return result


def hold_calibration(binary, events, batch_size, fs):
    """Batches starting before 5 s keep Pareto's own detections, as in the loop."""
    stop = math.ceil(5 * fs / batch_size) * batch_size
    window = (events["time"] >= len(events["initial"])) & (events["time"] < stop)
    binary[len(events["initial"]) : stop] = 0
    primary = window & events["primary"].astype(bool)
    binary[events["time"][primary], events["unit"][primary]] = 1
    return binary


def final_events(case):
    from .events import descriptor

    name = f"events/{stub(*case)}.npz"
    if sha256(FINAL / name) != read(FINAL / "sealed.json")["files"][name]:
        raise ValueError("Final validation candidates changed after their seal")
    with np.load(FINAL / name) as archive:
        events = {k: archive[k] for k in archive.files}
    events["record"] = descriptor(*case, phase="validation")
    return FINAL / name, events


def decide():
    """Label-free open-loop decisions of every head on the sealed L2 final candidates."""
    seal_path = FOLDER / "decisions/sealed.json"
    if seal_path.exists():
        return read(seal_path)
    files, reliance, reproduction = {}, {}, {}
    # The accepted L2 head, decided open loop on its own candidates, must give back
    # its sealed closed-loop spikes; otherwise open-loop decoding is not the loop's.
    accepted = read(L2_ROOT / "round3/selected.json")
    if sha256(OUTPUT / accepted["checkpoint"]) != accepted["checkpoint_sha256"]:
        raise ValueError("Accepted L2 head changed")
    l2_head = load_head(OUTPUT / accepted["checkpoint"])
    for case in records(VALIDATION):
        _, events = final_events(case)
        logits = predict_logits(l2_head, events)
        own = hold_calibration(
            decode_record(events, logits, **accepted["selection"]["parameters"]),
            events,
            204,
            2048,
        )
        with np.load(FINAL / f"predictions/{stub(*case)}.npz") as archive:
            reproduction[stub(*case)] = int(np.sum(own != archive["spikes"]))
    for arm in ARMS:
        parameters = read(FOLDER / arm / "selected.json")["selection"]["parameters"]
        for seed in SEEDS:
            model = load_head(head_path(arm, seed))
            flips = candidates = 0
            for case in records(VALIDATION):
                path, events = final_events(case)
                logits = predict(model, arm, path, events)
                binary = hold_calibration(
                    decode_record(events, logits, **parameters), events, 204, 2048
                )
                destination = FOLDER / f"decisions/{arm}/seed{seed}/{stub(*case)}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ValueError("Orphaned decision: preserve it and use a fresh folder")
                np.savez_compressed(destination, spikes=binary)
                files[destination.relative_to(FOLDER).as_posix()] = sha256(destination)
                if arm == "posture24":
                    # Reliance: posture frozen at the record median, displacement zero.
                    threshold = parameters["threshold"]
                    original = decision_probability(logits, events, parameters) >= threshold
                    frozen = predict_frozen(model, events, posture_for(path, events))
                    changed = decision_probability(frozen, events, parameters) >= threshold
                    flips += int(np.sum(original != changed))
                    candidates += len(logits)
            if arm == "posture24":
                reliance[f"seed{seed}"] = dict(
                    flipped_pct=flips / candidates * 100, candidates=candidates
                )
    result = dict(
        sealed_at_utc=now(),
        labels_opened=False,
        files=files,
        accepted_l2_head_open_loop_sample_differences=reproduction,
        posture_reliance=reliance,
    )
    write_new(seal_path, result)
    return result


def decision_probability(logits, events, parameters):
    prior = (events["primary"].astype(np.float32) * 2 - 1) * np.log(9)
    strength = parameters["strength"]
    return 1 / (1 + np.exp(-(prior + strength * (logits - prior))))


@torch.no_grad()
def predict_frozen(model, events, columns):
    columns = columns.copy()
    columns[:, 0] = np.median(columns[:, 0])
    columns[:, 1] = 0
    features = np.concatenate([events["features"], columns], axis=1)
    parts = []
    for start in range(0, len(features), 65536):
        x = torch.from_numpy(np.ascontiguousarray(features[start : start + 65536]))
        primary = torch.from_numpy(events["primary"][start : start + 65536])
        parts.append(model(x, primary).numpy())
    return np.concatenate(parts)


def posture_strata(record, binary):
    from adapt_decomp.spikes import rate_of_agreement_paired

    from benchmarks.robust_mvdr.phases import masked

    from .data import load_angle, load_truth

    truth = load_truth(record, phase="validation")
    angle = load_angle(record, phase="validation")
    after = np.arange(len(angle)) >= 5 * 2048
    result = {}
    for label, mask in (
        ("near_reference", (angle < 5) & after),
        ("far_20deg", (angle >= 20) & after),
    ):
        if mask.sum() < 2048:
            result[label] = [None] * truth.shape[1]
            continue
        roa = rate_of_agreement_paired(
            masked(truth, mask), masked(binary, mask), fs=2048, tol_spike_ms=2
        )[0]
        result[label] = [float(v) for v in roa]
    return result


def score():
    """Original evaluator on sealed open-loop decisions; paired cluster bootstrap."""
    from collections import defaultdict

    from .evaluate import score_record
    from .events import descriptor
    from .l2_ablation import combine, metrics

    path = FOLDER / "analysis.json"
    if path.exists():
        return read(path)
    seal = read(FOLDER / "decisions/sealed.json")
    for name, digest in seal["files"].items():
        if sha256(FOLDER / name) != digest:
            raise ValueError("Sealed open-loop decision changed")
    rows, clusters = {}, defaultdict(list)
    for case in records(VALIDATION):
        calibration = descriptor(*case, phase="validation")["calibration"]
        for unit, index in enumerate(calibration.gt_matched_indices):
            clusters[(case[1], int(index))].append((case[1], case[2], unit))
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_seed{seed}"
            rows[name] = {}
            for case in records(VALIDATION):
                record = descriptor(*case, phase="validation")
                score_path = FOLDER / f"scores/{arm}/seed{seed}/{stub(*case)}.json"
                if not score_path.exists():
                    with np.load(FOLDER / f"decisions/{arm}/seed{seed}/{stub(*case)}.npz") as a:
                        binary = a["spikes"]
                    result = score_record(record, dict(spikes=binary), phase="validation")
                    strata = posture_strata(record, binary)
                    for row in result["rows"]:
                        row["strata"] = {k: v[row["unit"]] for k, v in strata.items()}
                    write_new(score_path, result)
                for row in read(score_path)["rows"]:
                    rows[name][(row["condition"], row["snr_db"], row["unit"])] = row
    keys = sorted(rows["base22_seed0"])
    names = sorted(clusters)

    def summary(selected):
        values = {}
        for arm_seed, table in rows.items():
            chosen = [table[k] for k in selected]
            value = metrics(chosen)
            for label in ("near_reference", "far_20deg"):
                found = [r["strata"][label] for r in chosen if r["strata"][label] is not None]
                value[label] = float(np.mean(found) * 100) if found else None
            values[arm_seed] = value
        weights = {f"posture24_seed{s}": 1 / 3 for s in SEEDS}
        weights.update({f"base22_seed{s}": -1 / 3 for s in SEEDS})
        return values, combine(weights, values)

    arms, point = summary(keys)
    rng = np.random.default_rng(2026)
    draws = defaultdict(list)
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, len(names), len(names))
        _, contrast = summary([k for i in pick for k in clusters[names[i]]])
        for label, value in contrast.items():
            draws[label].append(np.nan if value is None else value)
    effect = {
        label: dict(
            estimate=value,
            ci95=[float(v) for v in np.nanpercentile(draws[label], [2.5, 97.5])],
        )
        for label, value in point.items()
    }
    bce = {
        arm: [read(head_path(arm, s).with_name("complete.json"))["validation_bce"] for s in SEEDS]
        for arm in ARMS
    }
    decisions = dict(
        base_seed0_reproduced_l2=read(FOLDER / "base22/seed0/reproduction.json")["identical"],
        open_loop_reproduces_sealed_l2=not any(
            seal["accepted_l2_head_open_loop_sample_differences"].values()
        ),
        posture_helps_open_loop=effect["global_roa_pct"]["ci95"][0] > 0,
        bce_lower_all_seeds=all(
            p < b for p, b in zip(bce["posture24"], bce["base22"], strict=True)
        ),
        far_gain_exceeds_near=effect["far_20deg"]["estimate"]
        > effect["near_reference"]["estimate"],
    )
    decisions["propose_closed_loop_level"] = (
        decisions["posture_helps_open_loop"] and decisions["bce_lower_all_seeds"]
    )
    result = dict(
        analysed_at_utc=now(),
        instances=len(keys),
        clusters=len(names),
        arms=arms,
        posture_minus_base=effect,
        validation_bce=bce,
        selections={arm: read(FOLDER / arm / "selected.json")["selection"] for arm in ARMS},
        reliance=seal["posture_reliance"],
        decisions=decisions,
    )
    write_new(path, result)
    return result


def register_protocol():
    if (FOLDER / "protocol.json").exists():
        return checked(FOLDER)
    register(
        FOLDER,
        dict(
            purpose="Open-loop test: does C1's blind posture improve C4 discharge decisions?",
            arms=dict(
                base22="22 L2 inputs; seed 0 retrained to verify the pipeline reproduces L2",
                posture24="22 L2 inputs + q and q - q(2 s earlier) from C1, same recipe",
            ),
            data="exactly memory_training.paths(L2/standard, 3): L1 open + L1 rounds 1-3 + "
            "L2 rounds 1-3 (training sub-01..03); epoch/TPE selection on L2 round-3 sub-04",
            recipe="memory_training.fit/select unchanged: AdamW 1e-3, wd 0.01, 10 epochs, "
            "unit-balanced BCE, feature noise 0.02, seeds 0/1/2, TPE seed 2026, 40 trials, "
            "top-3 three-seed repeats, median seed",
            base_reuse="if base22 seed 0 equals L2 round-3 seed 0 bit for bit, base seeds 1-2 "
            "reuse the L2 round-3 checkpoints; otherwise they are retrained",
            evaluation="open-loop decisions on the sealed L2 final_validation candidates (not "
            "used for training or selection), calibration batches held at Pareto, sealed "
            "before the original evaluator; the accepted L2 head must reproduce its sealed "
            "spikes there",
            primary="seed-mean global RoA, posture24 - base22, paired cluster bootstrap "
            f"({BOOTSTRAP} draws, seed 2026, condition x ground-truth unit)",
            decisions=[
                "posture helps in open loop only if the primary 95% CI lies above zero",
                "supporting: posture24 validation BCE below base22 for all three seeds",
                "mechanism: far (>=20 deg) gain above near-reference (<5 deg) gain",
                "propose a registered closed-loop level only if the first two hold",
                "nothing is deployed; no accepted level changes; sub-05 never opened",
            ],
            known_bias="C1 is in-sample on training subjects; sub-04 q is out of sample",
        ),
    )
    return checked(FOLDER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "select", "decide", "score"])
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "prepare":
        register_protocol()
        print(len(prepare()["sources"]), flush=True)
    else:
        checked(FOLDER)
        if args.command == "train":
            for seed in args.seeds:
                fit(args.arm, seed)
        elif args.command == "select":
            print(select(args.arm)["selection"], flush=True)
        elif args.command == "decide":
            print(decide()["accepted_l2_head_open_loop_sample_differences"], flush=True)
        else:
            print(score()["decisions"], flush=True)
