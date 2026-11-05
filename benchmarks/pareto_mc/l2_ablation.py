"""L2 attribution ablation and label-free memory screening, before resuming L3.

Earlier levels remain read-only. Every arm is sealed, and checked against the
sealed L1/L2 predictions, before the original evaluator opens sub-04 labels.
Screening reads only EMG-derived states and memory traces. No registered L2/L3
source changes; this module only adds new runs in fresh folders.
"""

import argparse
import copy
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import numpy as np
import torch

from .backbone import build
from .common import (
    CONDITIONS,
    OUTPUT,
    TRAIN,
    VALIDATION,
    now,
    records,
    register,
    sha256,
    stub,
    write_new,
)
from .data import load_record
from .evaluate import score_record, summarize
from .events import descriptor
from .experiment import checked
from .memory import PosturalMemory
from .memory_experiment import SEEDS, STRUCTURES, read, seed_selection
from .memory_loop import MemoryConfig, MemoryLoop, make_memory_adapter
from .observer import EMGFeatures, load_observer
from .shared_rollout import shared_replay
from .train_likelihood import load_model

MODES = ("off", "feature", "mixing", "full")
BOOTSTRAP = 2000
SCREEN_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
SCREEN_BETA_MAX = 0.45  # Midpoint of the registered TPE range [0.1, 0.8].
FULL_SCREEN = dict(thresholds=(0.6, 0.85), kappa=0.16, beta_max=0.45, momentum=0.895)
FULL_SCREEN_SNRS = (30, 20)
REFERENCES = {
    "l2_seed2_full": "L2/standard/final_validation",
    "l1_off": "L1/all_feedback/final_validation",
}


class NoMixing(MemoryLoop):
    """C4 still receives retrieval confidence; the post-QR filter blend is skipped."""

    def _update_sep_vectors(self, z, sources, trusted, index):
        # beta = beta_max * mass * reliability: zero reliability removes only the
        # blend, keeping retrieval, traces and C4 inputs of the accepted loop.
        self.query_reliability = 0.0
        super()._update_sep_vectors(z, sources, trusted, index)


class HiddenMass:
    """C4 receives zero retrieval confidence, as in L1 data; the blend is unchanged."""

    def __init__(self, builder):
        self.builder = builder

    def build(self, pareto, memory, primary, *, speed, mass=None):
        return self.builder.build(pareto, memory, primary, speed=speed)


def mode_settings(settings, mode):
    if mode not in MODES:
        raise ValueError(f"Undeclared ablation mode: {mode}")
    return dict(settings, beta_max=0.0) if mode == "off" else dict(settings)


def apply_mode(adapter, mode):
    if mode == "feature":
        adapter.__class__ = NoMixing
    elif mode == "mixing":
        adapter.builder.builder = HiddenMass(adapter.builder.builder)
    adapter.collector = False
    return adapter


def head(selection):
    return dict(
        checkpoint=(OUTPUT / selection["checkpoint"]).relative_to(OUTPUT).as_posix(),
        checkpoint_sha256=selection["checkpoint_sha256"],
        parameters=selection["selection"]["parameters"],
    )


def mode_adapter(record, spec):
    path = OUTPUT / spec["head"]["checkpoint"]
    if sha256(path) != spec["head"]["checkpoint_sha256"]:
        raise ValueError("Ablation classifier changed")
    adapter = make_memory_adapter(
        record,
        load_model(path),
        memory_config=MemoryConfig(**spec["settings"]),
        **spec["head"]["parameters"],
    )
    return apply_mode(adapter, spec["mode"])


def ablation_arms():
    """Factorial memory modes for the three L2 seeds; L1's head with and without blend."""
    retained = read(OUTPUT / "L2/retained.json")
    if not retained["passed"]:
        raise ValueError("L2 must be the retained level")
    settings = retained["settings"]
    plan = {}
    for seed in SEEDS:
        selection = head(seed_selection(retained["selection"], seed))
        for mode in MODES:
            plan[f"l2_seed{seed}_{mode}"] = dict(
                head=selection, mode=mode, settings=mode_settings(settings, mode)
            )
    l1 = head(read(OUTPUT / "L1/all_feedback/round3/selected.json"))
    for mode in ("off", "mixing"):
        plan[f"l1_{mode}"] = dict(head=l1, mode=mode, settings=mode_settings(settings, mode))
    return plan


def full_screen_arms():
    """L3's search head (L2 seed 0) over every structure, at two insertion thresholds."""
    retained = read(OUTPUT / "L2/retained.json")
    selection = head(seed_selection(retained["selection"], 0))
    plan = {}
    for bins, h in STRUCTURES:
        for threshold in FULL_SCREEN["thresholds"]:
            settings = MemoryConfig(
                bins=bins,
                h=h,
                threshold=threshold,
                kappa=FULL_SCREEN["kappa"],
                beta_max=FULL_SCREEN["beta_max"],
                momentum=FULL_SCREEN["momentum"],
                calibration_only=False,
            )
            plan[f"b{bins}_h{h}_t{threshold}"] = dict(
                head=selection, mode="full", settings=asdict(settings)
            )
    return plan


def shared_task(arguments):
    root, case, plan, phase = arguments
    torch.set_num_threads(4)
    name = stub(*case)
    pending = {}
    for arm, spec in plan.items():
        folder = root / "arms" / arm
        path = folder / "metadata" / f"{name}.json"
        if path.exists():
            metadata = read(path)
            if metadata["spec"] != spec:
                raise ValueError("Cannot resume an arm with a different specification")
            for key, digest in metadata["files"].items():
                if sha256(folder / key) != digest:
                    raise ValueError("Completed ablation prediction changed")
        elif (folder / "predictions" / f"{name}.npz").exists():
            raise ValueError("Orphaned prediction: preserve it and use a fresh output folder")
        else:
            pending[arm] = spec
    if not pending:
        return name
    record = load_record(*case, phase=phase)
    inputs = read(OUTPUT / "C1/features" / f"{name}.json")
    if any(record[key] != inputs[key] for key in ("raw_sha256", "calibration_sha256")):
        raise ValueError("Raw EMG or calibration changed since C1")
    config = copy.deepcopy(record["config"])
    config.max_sigma_batches = 1
    master = build(record["calibration"], config)
    adapters = [mode_adapter(record, spec) for spec in pending.values()]
    outputs = shared_replay(
        record["raw"],
        master,
        adapters,
        observer=load_observer(),
        extractor=EMGFeatures(record["calibration"], config),
    )
    for (arm, spec), adapter, output in zip(pending.items(), adapters, outputs, strict=True):
        folder = root / "arms" / arm
        destination = folder / "predictions" / f"{name}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **output)
        write_new(
            folder / "metadata" / f"{name}.json",
            dict(
                predicted_at_utc=now(),
                record=list(case),
                spec=spec,
                labels_opened=False,
                shared="raw preprocessing, whitening and blind C1 only; per-batch causal reuse",
                memory_hit_rate=float(output["memory"][:, 3].mean()),
                mean_retrieval_mass=float(output["memory"][:, 0].mean()),
                populated_bins=int((adapter.memory.quality[1:] > 0).any(1).sum()),
                files={destination.relative_to(folder).as_posix(): sha256(destination)},
            ),
        )
    return name


def roll_out(root, plan, cases, phase, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(shared_task, (root, case, plan, phase)) for case in cases]
        for future in as_completed(futures):
            print("ablation group", len(plan), future.result(), flush=True)
    for arm in plan:
        folder = root / "arms" / arm
        files = {}
        for case in cases:
            metadata = read(folder / "metadata" / f"{stub(*case)}.json")
            for key, digest in metadata["files"].items():
                if sha256(folder / key) != digest:
                    raise ValueError("Ablation prediction changed before seal")
                files[key] = digest
        path = folder / "sealed.json"
        if not path.exists():
            write_new(path, dict(sealed_at_utc=now(), files=files, no_labels_during_rollout=True))
        elif read(path)["files"] != files:
            raise ValueError("Existing ablation seal disagrees with completed predictions")


def reproduction(root):
    """Label-free bitwise comparison with the sealed L2 and L1 final predictions."""
    path = root / "reproduction.json"
    if path.exists():
        return read(path)
    result = dict(checked_at_utc=now(), labels_opened=False, arms={})
    for arm, reference in REFERENCES.items():
        source = OUTPUT / reference
        sealed = read(source / "sealed.json")["files"]
        rows = {}
        for case in records(VALIDATION):
            key = f"predictions/{stub(*case)}.npz"
            if sha256(source / key) != sealed[key]:
                raise ValueError("Reference prediction changed after its seal")
            with np.load(source / key) as a, np.load(root / "arms" / arm / key) as b:
                shared = [
                    k
                    for k in ("spikes", "state", "memory", "centroid", "trusted_count")
                    if k in a.files
                ]
                rows[stub(*case)] = dict(
                    equal={k: bool(np.array_equal(a[k], b[k])) for k in shared},
                    spike_samples_equal_pct=float(np.mean(a["spikes"] == b["spikes"]) * 100),
                )
        result["arms"][arm] = dict(
            reference=reference,
            bitwise=all(all(row["equal"].values()) for row in rows.values()),
            records=rows,
        )
    result["bitwise"] = all(value["bitwise"] for value in result["arms"].values())
    write_new(path, result)
    return result


def score_arm(folder):
    for key, digest in read(folder / "sealed.json")["files"].items():
        if sha256(folder / key) != digest:
            raise ValueError("Sealed ablation prediction changed")
    results = []
    for case in records(VALIDATION):
        name = stub(*case)
        path = folder / "scores" / f"{name}.json"
        if not path.exists():
            with np.load(folder / "predictions" / f"{name}.npz") as archive:
                prediction = dict(spikes=archive["spikes"])
            record = descriptor(*case, phase="validation")
            write_new(path, score_record(record, prediction, phase="validation"))
        results.append(read(path))
    return results


def metrics(rows):
    """evaluate.summarize definitions, flattened, plus ramp5s cycles 6-8."""
    summary = summarize([dict(rows=rows)])
    values = {
        key: summary[key]
        for key in (
            "global_roa_pct",
            "ramp5s_roa_pct",
            "ge90_pct",
            "precision_pct",
            "recall_pct",
            "ramp5s_cycle1_roa_pct",
            "ramp5s_cycles2_8_roa_pct",
        )
    }
    values.update(summary["phases"])
    late = [v for r in rows if r["condition"] == "triangular-ramp5s" for v in r["cycles"][5:8]]
    values["ramp5s_cycles6_8_roa_pct"] = float(np.mean(late) * 100) if late else None
    values["lost"] = float(sum(summary["lost_by_condition"].values()))
    return values


def combine(weights, values):
    """Weighted sum of arm metrics; None when any contributing metric is undefined."""
    result = {}
    for key in next(iter(values.values())):
        parts = [(w, values[arm][key]) for arm, w in weights.items()]
        result[key] = None if any(v is None for _, v in parts) else sum(w * v for w, v in parts)
    return result


def contrast_weights():
    third = 1 / len(SEEDS)

    def l2(mode, sign=1.0):
        return {f"l2_seed{s}_{mode}": sign * third for s in SEEDS}

    def merge(*parts, scale=1.0):
        total = defaultdict(float)
        for part in parts:
            for arm, w in part.items():
                total[arm] += w * scale
        return dict(total)

    return dict(
        memory_total=merge(l2("full"), l2("off", -1)),
        blend_main=merge(l2("full"), l2("feature", -1), l2("mixing"), l2("off", -1), scale=0.5),
        confidence_input_main=merge(
            l2("full"), l2("mixing", -1), l2("feature"), l2("off", -1), scale=0.5
        ),
        interaction=merge(l2("full"), l2("feature", -1), l2("mixing", -1), l2("off")),
        c4_retraining=merge(l2("off"), {"l1_off": -1.0}),
        l1_head_blend=dict(l1_mixing=1.0, l1_off=-1.0),
        accepted_l2_vs_l1=dict(l2_seed2_full=1.0, l1_off=-1.0),
    )


def analyse(root, plan):
    path = root / "analysis.json"
    if path.exists():
        return read(path)
    if not (root / "reproduction.json").exists():
        raise ValueError("Reproduction must be recorded before labels open")
    rows = {}
    for arm in plan:
        rows[arm] = {
            (r["condition"], r["snr_db"], r["unit"]): r
            for item in score_arm(root / "arms" / arm)
            for r in item["rows"]
        }
    keys = sorted(rows["l1_off"])
    if any(sorted(value) != keys for value in rows.values()):
        raise ValueError("Arms do not cover the same instances")
    # The same ground-truth unit appears at four SNRs; resample it as one cluster.
    clusters = defaultdict(list)
    for case in records(VALIDATION):
        calibration = descriptor(*case, phase="validation")["calibration"]
        for unit, index in enumerate(calibration.gt_matched_indices):
            clusters[(case[1], int(index))].append((case[1], case[2], unit))
    if sorted(k for group in clusters.values() for k in group) != keys:
        raise ValueError("Cluster map does not match the scored instances")
    names = sorted(clusters)
    weights = contrast_weights()

    def evaluate(selected):
        values = {arm: metrics([rows[arm][k] for k in selected]) for arm in plan}
        return values, {name: combine(w, values) for name, w in weights.items()}

    arms, point = evaluate(keys)
    rng = np.random.default_rng(2026)
    draws = defaultdict(lambda: defaultdict(list))
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, len(names), len(names))
        _, contrasts = evaluate([k for i in pick for k in clusters[names[i]]])
        for name, values in contrasts.items():
            for key, value in values.items():
                draws[name][key].append(np.nan if value is None else value)
    effects = {
        name: {
            key: dict(
                estimate=value,
                ci95=[float(v) for v in np.nanpercentile(draws[name][key], [2.5, 97.5])],
            )
            for key, value in values.items()
        }
        for name, values in point.items()
    }
    seeds = {
        mode: {
            key: [arms[f"l2_seed{s}_{mode}"][key] for s in SEEDS]
            for key in ("global_roa_pct", "ramp5s_roa_pct", "reference_rest")
        }
        for mode in MODES
    }
    memory = effects["memory_total"]["global_roa_pct"]
    retraining = effects["c4_retraining"]["global_roa_pct"]["estimate"]
    rest_off = float(np.mean(seeds["off"]["reference_rest"]))
    decisions = dict(
        memory_credited=memory["ci95"][0] > 0,
        gain_mainly_c4_retraining=memory["ci95"][0] <= 0 and retraining >= memory["estimate"],
        l2_heads_without_memory_mean_rest_pct=rest_off,
        rest_criterion_needs_memory=rest_off < 97,
        reproduction_bitwise=read(root / "reproduction.json")["bitwise"],
    )
    candidate, previous = arms["l2_seed2_full"], arms["l1_off"]
    delta = effects["accepted_l2_vs_l1"]
    gates = dict(
        note="Retrospective: L2 judged by L3-style criteria against L1. Not applied anywhere.",
        registered_cycles=candidate["ramp5s_cycles2_8_roa_pct"]
        >= candidate["ramp5s_cycle1_roa_pct"] + 5,
        proposed_global=delta["global_roa_pct"]["estimate"] >= 1
        and delta["global_roa_pct"]["ci95"][0] > 0,
        proposed_reuse=delta["ramp5s_cycles2_8_roa_pct"]["estimate"]
        - delta["ramp5s_cycle1_roa_pct"]["estimate"]
        >= 0,
        proposed_late_cycles=candidate["ramp5s_cycles6_8_roa_pct"]
        >= previous["ramp5s_cycles6_8_roa_pct"],
    )
    result = dict(
        analysed_at_utc=now(),
        instances=len(keys),
        clusters=len(names),
        bootstrap=dict(draws=BOOTSTRAP, seed=2026, unit="condition x ground-truth motor unit"),
        arms=arms,
        effects=effects,
        seed_spread=seeds,
        decisions=decisions,
        gate_proposal=gates,
    )
    write_new(path, result)
    return result


def calibration_terms(state, bins, h, beta_max, *, batch_size, fs):
    """Per-batch retrieval mass and blend positivity of the calibration-only loop.

    Mirrors MemoryEvents.build and MemoryLoop._update_sep_vectors exactly, so the
    hit rate needs no rollout: only the immutable calibration bin is populated.
    """
    memory = PosturalMemory(torch.zeros(1, 1), torch.zeros(2, 1), bins=bins, h=h)
    mass = torch.zeros(len(state))
    positive = np.zeros(len(state), dtype=bool)
    for index, (q, logvar, _) in enumerate(np.asarray(state).tolist()):
        if index * batch_size < 5 * fs:
            continue
        _, _, value = memory.retrieve(q, batch=index)
        reliability = float(np.exp(-0.5 * np.exp(logvar) / h**2))
        mass[index] = value[0]
        positive[index] = bool((beta_max * value * reliability)[0] > 0)
    return mass, positive


def calibration_hits(mass, positive, threshold):
    return positive & (mass >= threshold).numpy()


def screen_calibration_only(root):
    """Exact, label-free hit rates for every L2 structure and threshold."""
    path = root / "screen_calibration_only.json"
    if path.exists():
        return read(path)
    checked(root)
    # Validate the rollout-free formula against every recorded L2 search trace.
    search = OUTPUT / "L2/standard/search"
    trials = read(OUTPUT / "L2/standard/trials.json")
    mismatched, batches = 0, 0
    for trial in trials:
        settings = trial["attrs"]["settings"]
        folder = search / f"trial{trial['number']}/seed0"
        sealed = read(folder / "sealed.json")["files"]
        for case in records(VALIDATION):
            key = f"predictions/{stub(*case)}.npz"
            if sha256(folder / key) != sealed[key]:
                raise ValueError("Search prediction changed after its seal")
            with np.load(folder / key) as archive:
                state, trace = archive["state"], archive["memory"]
            mass, positive = calibration_terms(
                state,
                settings["bins"],
                settings["h"],
                settings["beta_max"],
                batch_size=204,
                fs=2048,
            )
            expected = calibration_hits(mass, positive, settings["threshold"])
            mismatched += int(np.sum(trace[:, 3, :].astype(bool) != expected[:, None]))
            batches += trace.shape[0] * trace.shape[2]
    if mismatched:
        raise AssertionError(f"Rollout-free hit rate disagrees on {mismatched} unit-batches")
    # C1 states depend only on EMG; any sealed L2 rollout of the training records holds them.
    source = OUTPUT / "L2/standard/round1/rollout"
    sealed = read(source / "sealed.json")["files"]
    states = {}
    for case in records(TRAIN):
        key = f"predictions/{stub(*case)}.npz"
        if sha256(source / key) != sealed[key]:
            raise ValueError("Training rollout changed after its seal")
        with np.load(source / key) as archive:
            states[stub(*case)] = archive["state"]
    table = []
    for bins, h in STRUCTURES:
        hits = {t: [] for t in SCREEN_THRESHOLDS}
        for state in states.values():
            mass, positive = calibration_terms(
                state, bins, h, SCREEN_BETA_MAX, batch_size=204, fs=2048
            )
            for threshold in SCREEN_THRESHOLDS:
                hits[threshold].append(calibration_hits(mass, positive, threshold))
        rest = PosturalMemory(torch.zeros(1, 1), torch.zeros(2, 1), bins=bins, h=h)
        table.append(
            dict(
                bins=bins,
                h=h,
                max_mass_at_reference=float(rest.retrieve(0.0, batch=0)[2][0]),
                hit_rate_pct={
                    str(t): float(np.mean(np.concatenate(v)) * 100) for t, v in hits.items()
                },
            )
        )
    result = dict(
        screened_at_utc=now(),
        labels_opened=False,
        validation=dict(
            source="all 45 sealed L2 search trials x 20 sub-04 records",
            unit_batches=batches,
            mismatched=mismatched,
        ),
        training_records=len(states),
        beta_max=SCREEN_BETA_MAX,
        hit_definition="fraction of all batches (calibration included, as in memory_trace) "
        "where the calibration-only loop blends filters",
        structures=table,
    )
    write_new(path, result)
    return result


def full_screen_summary(root, plan, cases):
    path = root / "summary.json"
    if path.exists():
        return read(path)
    arms = {}
    for arm, spec in plan.items():
        folder = root / "arms" / arm
        metadata = [read(folder / "metadata" / f"{stub(*case)}.json") for case in cases]
        arms[arm] = dict(
            settings=spec["settings"],
            hit_rate_pct=float(np.mean([m["memory_hit_rate"] for m in metadata]) * 100),
            mean_retrieval_mass=float(np.mean([m["mean_retrieval_mass"] for m in metadata])),
            mean_populated_bins=float(np.mean([m["populated_bins"] for m in metadata])),
            inactive_records=int(sum(m["memory_hit_rate"] == 0 for m in metadata)),
        )
    result = dict(summarized_at_utc=now(), labels_opened=False, records=len(cases), arms=arms)
    write_new(path, result)
    return result


def openblas_guard():
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise RuntimeError("Set OPENBLAS_NUM_THREADS=1, as in the registered L2 run")


def run_ablation(workers):
    root = OUTPUT / "L2_ablation"
    plan = ablation_arms()
    if not (root / "protocol.json").exists():
        openblas_guard()
        register(
            root,
            dict(
                purpose="Attribute the accepted L1->L2 gain to C4 retraining, the memory "
                "confidence input, and the filter blend before resuming L3",
                references=dict(
                    l1_gate_sha256=sha256(OUTPUT / "L1/all_feedback/gate.json"),
                    l2_retained_sha256=sha256(OUTPUT / "L2/retained.json"),
                    observer_sha256=sha256(OUTPUT / "C1/seed0/observer.pt"),
                ),
                arms=plan,
                modes=dict(
                    off="beta_max=0: no confidence input and no blend (L1 loop)",
                    feature="confidence input to C4, blend disabled",
                    mixing="blend as accepted, C4 sees zero confidence",
                    full="accepted L2 loop",
                ),
                cases=records(VALIDATION),
                execution="shared raw preprocessing, whitening and C1 per batch "
                "(bitwise-verified in C2_shared_identity_v1); four record workers, "
                "four Torch threads each, OpenBLAS one thread",
                reproduction="before labels open: l2_seed2_full must equal the sealed L2 "
                "final spikes/state/memory/centroids, and l1_off the sealed L1 final "
                "spikes/state/centroids, bit for bit",
                scoring="original evaluator after per-arm seals; evaluate.summarize "
                "metrics plus ramp5s cycles 6-8",
                contrasts=contrast_weights(),
                uncertainty=f"paired cluster bootstrap, {BOOTSTRAP} draws, seed 2026, "
                "clusters = condition x ground-truth motor unit (one unit across SNRs)",
                decisions=[
                    "memory is credited only if the 95% CI of memory_total (global RoA) "
                    "lies above zero",
                    "if not credited and c4_retraining >= memory_total (global point "
                    "estimates), the L1->L2 gain is attributed mainly to C4 retraining "
                    "and L3's premise is flagged as unsupported by L2",
                    "report whether L2 heads without memory meet reference rest >= 97%",
                    "without bitwise reproduction, contrasts are descriptive only",
                    "nothing is selected from this ablation; no accepted level changes",
                ],
                gate_proposal="retrospective only: L2 vs L1 under global >= +1 pp with "
                "CI above zero, delta(cycles 2-8) >= delta(cycle 1), and cycles 6-8 not "
                "below the previous level; compared with the registered cycle gate",
                sub05="never opened",
                openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
            ),
        )
    protocol = checked(root)
    if protocol["spec"]["arms"] != plan:
        raise ValueError("Ablation arms differ from the registered protocol")
    openblas_guard()
    roll_out(root, plan, records(VALIDATION), "validation", workers)
    return reproduction(root)


def run_full_screen(workers):
    root = OUTPUT / "L3_screen"
    plan = full_screen_arms()
    cases = [(s, c, n) for s in TRAIN for c in CONDITIONS for n in FULL_SCREEN_SNRS]
    if not (root / "protocol.json").exists():
        openblas_guard()
        register(
            root,
            dict(
                purpose="Label-free hit rates of every L3 memory structure on training EMG",
                arms=plan,
                cases=cases,
                fixed=FULL_SCREEN,
                head="L2 round3 seed 0, the head of L3's registered search",
                inactive_rule="a structure is flagged inactive when its hit rate is below "
                "1% at both screened thresholds",
                labels="never opened; training subjects only",
                sub05="never opened",
            ),
        )
    protocol = checked(root)
    if protocol["spec"]["arms"] != plan:
        raise ValueError("Screening arms differ from the registered protocol")
    openblas_guard()
    roll_out(root, plan, cases, "training", workers)
    return full_screen_summary(root, plan, cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "score", "screen", "screen-full"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "run":
        print(run_ablation(args.workers)["bitwise"], flush=True)
    elif args.command == "score":
        root = OUTPUT / "L2_ablation"
        checked(root)
        print(analyse(root, ablation_arms())["decisions"], flush=True)
    elif args.command == "screen":
        print(screen_calibration_only(OUTPUT / "L2_ablation")["structures"], flush=True)
    else:
        print(run_full_screen(args.workers)["arms"], flush=True)
