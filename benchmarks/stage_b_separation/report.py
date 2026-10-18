"""Stage B thresholds, state gap, unit retention and explicit untested conditions."""

import importlib.metadata
import json
import platform
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.stage_a.evaluate import paired_statistics, summarize, write_csv
from benchmarks.stage_b.common import OUTPUT as ORIGINAL_OUTPUT

from .common import CONDITIONS, OUTPUT, ROOT, SUBJECTS, sha256, write_json
from .experiment import SPEC, checked


def report():
    checked()
    paths = sorted((OUTPUT / "scored").glob("*.json"))
    if len(paths) != 80:
        raise ValueError("All 80 outer records are required")
    documents = [json.loads(p.read_text()) for p in paths]
    rows = [r for document in documents for r in document["rows"]]
    summaries = summarize(rows)
    for method, result in summaries.items():
        selected = [r for r in rows if r["method"] == method]
        keys = {(r["subject"], r["condition"], r["snr_db"], r["unit"]) for r in selected}
        if len(selected) != 591 or len(keys) != 591:
            raise ValueError("Missing or duplicate unit instances")
        result.update(
            mean_precision=float(np.mean([r["precision"] for r in selected])),
            mean_recall=float(np.mean([r["recall"] for r in selected])),
            mean_f1=float(np.mean([r["f1"] for r in selected])),
            mean_sil=float(np.mean([r["sil"] for r in selected if r["sil"] is not None])),
            mean_first_hold_roa=float(np.mean([r["first_hold_roa"] for r in selected])),
            mean_ramp_roa=float(np.mean([r["ramp_roa"] for r in selected])),
            mean_last_hold_roa=float(np.mean([r["last_hold_roa"] for r in selected])),
            end_units_roa_ge_90=sum(r["last_hold_roa"] >= 0.9 for r in selected),
            by_subject={
                s: float(np.mean([r["roa"] for r in selected if r["subject"] == s]) * 100)
                for s in SUBJECTS[1:]
            },
        )
    if set(summaries) != {
        "pareto",
        "aligned_k",
        "aligned_b",
        "gain_only_b",
        "static_mvdr",
        "original_b",
    }:
        raise ValueError("Unexpected methods")

    def key(row):
        return (row["subject"], row["condition"], row["snr_db"], row["unit"])

    baseline = {key(r): r for r in rows if r["method"] == "pareto"}
    for method, result in summaries.items():
        selected = [r for r in rows if r["method"] == method]
        result["final_tracking_vs_pareto"] = dict(
            gained=sum(
                r["last_hold_roa"] >= 0.9 and baseline[key(r)]["last_hold_roa"] < 0.9
                for r in selected
            ),
            lost=sum(
                r["last_hold_roa"] < 0.9 and baseline[key(r)]["last_hold_roa"] >= 0.9
                for r in selected
            ),
        )
    primary = summaries["aligned_b"]
    accuracy = {}
    for name, value, minimum in [
        ("global_mean", primary["mean_roa_pct"], 85),
        ("ramp5s", primary["by_condition"]["triangular-ramp5s"], 80),
        ("units_ge90", primary["pct_units_roa_ge_90"], 70),
    ]:
        accuracy[name] = dict(observed=value, minimum=minimum, passed=value >= minimum)
    metadata = [json.loads(p.read_text()) for p in (OUTPUT / "predictions/metadata").glob("*.json")]
    if len(metadata) != 80:
        raise ValueError("Missing timing")
    timing = {}
    for method in ("aligned_k", "aligned_b", "gain_only_b", "static_mvdr"):
        timing[method] = dict(
            median_record_median_core_ms=float(
                np.median([r["methods"][method]["median_core_batch_ms"] for r in metadata])
            ),
            median_record_p95_core_ms=float(
                np.median([r["methods"][method]["p95_core_batch_ms"] for r in metadata])
            ),
        )
    features = [
        json.loads(p.read_text()) for p in (ORIGINAL_OUTPUT / "features").glob("*/metadata.json")
    ]
    environment = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        inference_device="cpu",
        versions={
            k: importlib.metadata.version(k)
            for k in ("torch", "numpy", "scipy", "pandas", "matplotlib", "pytest")
        },
        training_runs={
            str(p.parent.relative_to(OUTPUT)): json.loads(p.read_text())
            for p in (OUTPUT / "folds").glob("*/*/complete.json")
        },
    )
    result = dict(
        status="no-go" if not all(g["passed"] for g in accuracy.values()) else "incomplete",
        accuracy_go=all(g["passed"] for g in accuracy.values()),
        primary_method="aligned_b",
        accuracy_gates=accuracy,
        summaries=summaries,
        scope="same 80 outer records, 591 calibrated instances; exploratory supervised transfer",
        k_minus_b_mean_pp=summaries["aligned_k"]["mean_roa_pct"] - primary["mean_roa_pct"],
        state_rmse_degrees=float(
            np.mean([d["state"]["aligned_b"]["angle_rmse_degrees"] for d in documents])
        ),
        timing=timing,
        delta_vs_original_b_pp=primary["mean_roa_pct"] - summaries["original_b"]["mean_roa_pct"],
        delta_vs_gain_only_b_pp=primary["mean_roa_pct"] - summaries["gain_only_b"]["mean_roa_pct"],
        unverified_gates=dict(
            raw_stream_cpu_22ms=dict(passed=None, reason="timing excludes projection and features"),
            speed_8_to_40_deg_per_second=dict(
                passed=None, reason="true-generator 1s and 2.5s conditions not distributed in FDSI"
            ),
        ),
        variance_captured_range=[
            min(r["calibration_variance_captured"] for r in features),
            max(r["calibration_variance_captured"] for r in features),
        ],
        statistics=paired_statistics(rows, method="aligned_b", seed=2029),
        protocol_sha256=sha256(OUTPUT / "protocol.json"),
        locked_sha256=sha256(OUTPUT / "locked.json"),
        prediction_manifest_sha256=sha256(OUTPUT / "predictions_complete.json"),
        interpretation="Estimated MUAP supervision; same generator; previously explored data",
        untested=SPEC["incomplete_tests"],
    )
    folder = OUTPUT / "report"
    write_json(folder / "summary.json", result)
    write_csv(folder / "per_unit.csv", rows)
    write_json(folder / "environment.json", environment)
    write_json(folder / "timing.json", metadata)
    histories = {
        str(p.relative_to(OUTPUT)): json.loads(p.read_text())
        for p in (OUTPUT / "folds").glob("*/*/history.json")
    }
    write_json(folder / "training.json", histories)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method in ("pareto", "original_b", "static_mvdr", "aligned_k", "aligned_b", "gain_only_b"):
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
    axes[0].set_title("Same held-out records")
    axes[1].set_title("Tracking phases")
    for axis in axes:
        axis.set(ylabel="Mean RoA (%)", ylim=(0, 100))
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Stage B — separation-aligned MoCA: {result['status']}")
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)
    for source, name in [
        (OUTPUT / "protocol.json", "stage_b_separation_protocol.json"),
        (OUTPUT / "locked.json", "stage_b_separation_locked.json"),
        (folder / "summary.json", "stage_b_separation_summary.json"),
        (folder / "per_unit.csv", "stage_b_separation_per_unit.csv"),
        (folder / "environment.json", "stage_b_separation_environment.json"),
        (folder / "training.json", "stage_b_separation_training.json"),
        (folder / "timing.json", "stage_b_separation_timing.json"),
    ]:
        shutil.copy2(source, ROOT / "docs/benchmarks" / name)
    shutil.copy2(folder / "comparison.png", ROOT / "docs/images/stage-b-separation.png")
    print(
        json.dumps(
            {
                "status": result["status"],
                "accuracy_gates": accuracy,
                "means": {m: s["mean_roa_pct"] for m, s in summaries.items()},
                "state_rmse_degrees": result["state_rmse_degrees"],
            },
            indent=2,
        )
    )
    return result
