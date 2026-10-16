"""Aggregate only disjoint outer subjects and report each benchmark failure mode."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.stage_a.evaluate import paired_statistics, summarize, write_csv

from .common import CONDITIONS, OUTPUT, SUBJECTS, sha256, write_json


def report():
    rows = []
    for subject in SUBJECTS[1:]:
        folder = OUTPUT / "evaluations" / subject
        paths = list((folder / "records").glob("*.json"))
        if len(paths) != 20:
            raise ValueError("All 80 held-out records are required")
        plan = json.loads((folder / "plan.json").read_text())
        if plan["locked_sha256"] != sha256(OUTPUT / "locked.json"):
            raise ValueError("Folds did not use the same frozen candidates")
        for path in paths:
            rows.extend(json.loads(path.read_text())["rows"])
    summaries = summarize(rows)
    keys = {}
    for method in summaries:
        values = [r for r in rows if r["method"] == method]
        keys[method] = {(r["subject"], r["condition"], r["snr_db"], r["unit"]) for r in values}
        if len(keys[method]) != len(values):
            raise ValueError("Repeated unit rows")
        summaries[method].update(
            mean_first_hold_roa=float(np.mean([r["first_hold_roa"] for r in values])),
            mean_ramp_roa=float(np.mean([r["ramp_roa"] for r in values])),
            mean_last_hold_roa=float(np.mean([r["last_hold_roa"] for r in values])),
            end_units_roa_ge_90=sum(r["last_hold_roa"] >= 0.9 for r in values),
            mean_precision=float(np.mean([r["precision"] for r in values])),
            mean_recall=float(np.mean([r["recall"] for r in values])),
            mean_f1=float(np.mean([r["f1"] for r in values])),
            by_subject={
                s: float(np.mean([r["roa"] for r in values if r["subject"] == s]) * 100)
                for s in SUBJECTS[1:]
            },
        )
    if any(k != keys["pareto"] for k in keys.values()):
        raise ValueError("Methods evaluated different calibrated units")
    primary, baseline = summaries["hybrid"], summaries["pareto"]
    gates = {}
    for condition, minimum in [
        ("triangular-ramp5s", 5),
        ("triangular-ramp10s", 5),
        ("staircase", 0),
    ]:
        delta = primary["by_condition"][condition] - baseline["by_condition"][condition]
        gates[condition] = dict(delta_pp=delta, minimum_pp=minimum, passed=delta >= minimum)
    end_delta = (primary["mean_last_hold_roa"] - baseline["mean_last_hold_roa"]) * 100
    tracked_delta = primary["end_units_roa_ge_90"] - baseline["end_units_roa_ge_90"]
    gates["final_phase"] = dict(delta_pp=end_delta, minimum_pp=0, passed=end_delta >= 0)
    gates["final_tracked_units"] = dict(delta=tracked_delta, minimum=0, passed=tracked_delta >= 0)
    source_diagnostics = [
        json.loads(p.read_text()) for p in (OUTPUT / "features").glob("*/metadata.json")
    ]
    if len(source_diagnostics) != 100 or any(
        d["identity_spike_disagreements"] for d in source_diagnostics
    ):
        raise ValueError("The 100-record zero-residual audit is incomplete")
    result = dict(
        status="go" if all(g["passed"] for g in gates.values()) else "no-go",
        protocol_sha256=sha256(OUTPUT / "protocol.json"),
        locked_sha256=sha256(OUTPUT / "locked.json"),
        scope="80 records; outer subjects sub-02..05; same units for every method",
        interpretation=(
            "Exploratory grouped transfer; original P1 outcomes had previously been seen"
        ),
        summaries=summaries,
        gates=gates,
        statistics=paired_statistics(rows, method="hybrid", seed=2027),
        zero_residual_audit=dict(recordings=100, spike_disagreements=0),
        unseen_training_selection_conditions=["15dB", "ramp5s", "ramp20s", "outer subject"],
        timing_scope="head on cached adaptive sources, not live end-to-end latency",
    )
    destination = OUTPUT / "report"
    write_json(destination / "summary.json", result)
    write_csv(destination / "per_unit.csv", rows)
    positions = np.arange(len(CONDITIONS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method, offset, color in [("pareto", -0.18, "#217a60"), ("hybrid", 0.18, "#386cb0")]:
        axes[0].bar(
            positions + offset,
            [summaries[method]["by_condition"][c] for c in CONDITIONS],
            0.36,
            label=method,
            color=color,
        )
    axes[0].set_xticks(positions, ["Staircase", "40 s", "20 s", "10 s", "5 s"])
    axes[0].set(ylabel="Mean RoA (%)", ylim=(0, 100), title="Held-out subject: 80 records")
    for method, color in [("pareto", "#217a60"), ("hybrid", "#386cb0")]:
        axes[1].plot(
            ["First", "Middle", "Last"],
            [
                summaries[method]["mean_" + p + "_roa"] * 100
                for p in ["first_hold", "ramp", "last_hold"]
            ],
            "o-",
            color=color,
            label=method,
        )
    axes[1].set(ylabel="Mean RoA (%)", ylim=(0, 100), title="Tracking across phases")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle(f"Calibration-conditioned Pareto residual — {result['status']}")
    fig.tight_layout()
    fig.savefig(destination / "comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"status": result["status"], "gates": gates}, indent=2))
    return result
