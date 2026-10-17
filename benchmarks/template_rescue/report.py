"""Paired outcomes, fixed acceptance gates and component ablations."""

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.stage_a.evaluate import paired_statistics, summarize, write_csv

from .common import CONDITIONS, OUTPUT, ROOT, SUBJECTS, sha256, write_json
from .experiment import SPEC, frozen


def acceptance(primary, baseline):
    gates = {}
    for condition, minimum in [
        ("triangular-ramp5s", 5),
        ("triangular-ramp10s", 5),
        ("staircase", 0),
    ]:
        delta = primary["by_condition"][condition] - baseline["by_condition"][condition]
        gates[condition] = dict(delta_pp=delta, minimum_pp=minimum, passed=delta >= minimum)
    for key, metric in [
        ("final_phase", "mean_last_hold_roa"),
        ("final_tracked_units", "end_units_roa_ge_90"),
    ]:
        delta = primary[metric] - baseline[metric]
        if key == "final_phase":
            delta *= 100
        gates[key] = dict(delta=delta, minimum=0, passed=delta >= 0)
    delta = primary["mean_roa_pct"] - baseline["mean_roa_pct"]
    gates["global_mean"] = dict(delta_pp=delta, minimum_exclusive=0, passed=delta > 0)
    delta = primary["by_snr"]["15"] - baseline["by_snr"]["15"]
    gates["snr15"] = dict(delta_pp=delta, minimum_pp=0, passed=delta >= 0)
    return gates


def report():
    frozen()
    paths = sorted((OUTPUT / "scored").glob("*.json"))
    if len(paths) != 80:
        raise ValueError("All 80 scored records required")
    rows = [r for p in paths for r in json.loads(p.read_text())["rows"]]
    summaries = summarize(rows)
    identities = {}
    for method, result in summaries.items():
        selected = [r for r in rows if r["method"] == method]
        keys = {(r["subject"], r["condition"], r["snr_db"], r["unit"]) for r in selected}
        if len(keys) != 591 or len(selected) != 591:
            raise ValueError("Missing or duplicate unit instances")
        identities[method] = keys
        result.update(
            mean_first_hold_roa=float(np.mean([r["first_hold_roa"] for r in selected])),
            mean_ramp_roa=float(np.mean([r["ramp_roa"] for r in selected])),
            mean_last_hold_roa=float(np.mean([r["last_hold_roa"] for r in selected])),
            end_units_roa_ge_90=sum(r["last_hold_roa"] >= 0.9 for r in selected),
            mean_precision=float(np.mean([r["precision"] for r in selected])),
            mean_recall=float(np.mean([r["recall"] for r in selected])),
            mean_f1=float(np.mean([r["f1"] for r in selected])),
            by_subject={
                s: float(np.mean([r["roa"] for r in selected if r["subject"] == s]) * 100)
                for s in SUBJECTS[1:]
            },
        )
    if set(summaries) != set(SPEC["methods"]) or any(
        keys != identities["pareto"] for keys in identities.values()
    ):
        raise ValueError("Unequal method comparisons")
    all_gates = {m: acceptance(summaries[m], summaries["pareto"]) for m in SPEC["methods"][1:]}
    metadata = [
        json.loads(p.read_text()) for p in (OUTPUT / "predictions" / "metadata").glob("*.json")
    ]
    if len(metadata) != 80 or any(m["physical"]["observer_spike_disagreements"] for m in metadata):
        raise ValueError("Independent Pareto observer audit failed")
    if any(m["methods"]["rescue"]["fallback_spike_disagreements"] for m in metadata):
        raise ValueError("Fallback audit failed")
    baseline_rows = {
        (r["subject"], r["condition"], r["snr_db"], r["unit"]): r
        for r in rows
        if r["method"] == "pareto"
    }
    principal = [r for r in rows if r["method"] == "rescue"]
    gains = sum(
        r["last_hold_roa"]
        >= 0.9
        > baseline_rows[(r["subject"], r["condition"], r["snr_db"], r["unit"])]["last_hold_roa"]
        for r in principal
    )
    losses = sum(
        r["last_hold_roa"]
        < 0.9
        <= baseline_rows[(r["subject"], r["condition"], r["snr_db"], r["unit"])]["last_hold_roa"]
        for r in principal
    )
    result = dict(
        status="go" if all(g["passed"] for g in all_gates["rescue"].values()) else "no-go",
        primary_method="rescue",
        summaries=summaries,
        gates=all_gates,
        protocol_sha256=sha256(OUTPUT / "protocol.json"),
        locked_sha256=sha256(OUTPUT / "locked.json"),
        predictions_manifest_sha256=sha256(OUTPUT / "predictions_complete.json"),
        interpretation="Exploratory reuse of known benchmark; no new independent data",
        statistics=paired_statistics(rows, method="rescue", seed=2028),
        final_threshold_changes=dict(gained=gains, lost=losses),
        audits=dict(
            independent_observer_records=80,
            observer_spike_disagreements=0,
            fallback_spike_disagreements=0,
        ),
        diagnostics=dict(
            recovered_candidate_events=sum(
                m["methods"]["rescue"]["recovered_candidate_events"] for m in metadata
            ),
            median_physical_batch_ms=float(
                np.median([m["physical"]["median_physical_batch_ms"] for m in metadata])
            ),
            median_head_batch_ms=float(
                np.median([m["methods"]["rescue"]["median_head_batch_ms"] for m in metadata])
            ),
            timing_scope="cached Pareto; excludes base adaptation, preprocessing and I/O",
        ),
    )
    destination = OUTPUT / "report"
    write_json(destination / "summary.json", result)
    write_csv(destination / "per_unit.csv", rows)
    write_json(destination / "prediction_diagnostics.json", metadata)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for method in ("pareto", "previous", "constant", "protected", "physical", "rescue"):
        axes[0].plot(
            range(5), [summaries[method]["by_condition"][c] for c in CONDITIONS], "o-", label=method
        )
        axes[1].plot(
            ["First", "Middle", "Last"],
            [
                summaries[method]["mean_" + phase + "_roa"] * 100
                for phase in ("first_hold", "ramp", "last_hold")
            ],
            "o-",
            label=method,
        )
    axes[0].set_xticks(range(5), ["Staircase", "40 s", "20 s", "10 s", "5 s"])
    axes[0].set_title("Same 80 records: conditions")
    axes[1].set_title("Tracking phases")
    for axis in axes:
        axis.set(ylabel="Mean RoA (%)", ylim=(0, 100))
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Calibration-anchored multichannel rescue: {result['status']}")
    fig.tight_layout()
    fig.savefig(destination / "comparison.png", dpi=180)
    plt.close(fig)
    for source, target in [
        (OUTPUT / "protocol.json", "template_rescue_protocol.json"),
        (OUTPUT / "locked.json", "template_rescue_locked.json"),
        (destination / "summary.json", "template_rescue_summary.json"),
        (destination / "per_unit.csv", "template_rescue_per_unit.csv"),
        (destination / "prediction_diagnostics.json", "template_rescue_diagnostics.json"),
    ]:
        shutil.copy2(source, ROOT / "docs/benchmarks" / target)
    shutil.copy2(destination / "comparison.png", ROOT / "docs/images/template-rescue.png")
    print(
        json.dumps(
            {
                "status": result["status"],
                "gates": all_gates["rescue"],
                "means": {m: s["mean_roa_pct"] for m, s in summaries.items()},
            },
            indent=2,
        )
    )
    return result
