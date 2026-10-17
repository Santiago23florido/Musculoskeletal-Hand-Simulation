"""Post-hoc diagnostics of frozen predictions; never selects or changes a model."""

import csv
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.template_rescue.common import (
    OUTPUT,
    PREVIOUS,
    ROOT,
    SUBJECTS,
    reference,
    sha256,
    stub,
    write_json,
)


def boundary_diagnostic(rows):
    from benchmarks.stage_a.common import pareto_path
    from benchmarks.template_rescue.physics import reference_envelope

    reference()
    from adapt_decomp import AdaptationResult

    baseline = {
        (r["subject"], r["condition"], r["snr_db"], r["unit"]): r
        for r in rows
        if r["method"] == "pareto"
    }
    changed = [
        r
        for r in rows
        if r["method"] == "rescue"
        and r["last_hold_roa"]
        != baseline[(r["subject"], r["condition"], r["snr_db"], r["unit"])]["last_hold_roa"]
    ]
    records = []
    identities = sorted({(r["subject"], r["condition"], int(r["snr_db"])) for r in changed})
    for subject, condition, snr in identities:
        name = stub(subject, condition, snr)
        original = AdaptationResult.load(pareto_path(subject, condition, snr)).spikes.numpy()
        with np.load(OUTPUT / "predictions/rescue" / f"{name}.npz") as archive:
            binary = archive["spikes"]
        offsets = np.flatnonzero(np.any(binary[-10240:] != original[-10240:], axis=1))
        features = np.load(PREVIOUS / "features" / name / "features.npy")
        contexts = []
        for index in sorted(set(((offsets + len(binary) - 10240) // 204).tolist())):
            q, speed = (float(x) for x in features[index, 0, :2])
            envelope, fallback = reference_envelope(q, speed)
            contexts.append(
                dict(batch=index, q=q, speed=speed, envelope=envelope, fallback=fallback)
            )
        records.append(
            dict(
                record=name,
                changed_samples_in_final_window=len(offsets),
                max_offset_from_phase_start_samples=int(offsets.max()),
                contexts=contexts,
            )
        )
    result = dict(
        analysis_code_sha256=sha256(__file__),
        changed_final_phase_instances=len(changed),
        records=records,
        selection_used=False,
        interpretation="Post-hoc boundary diagnosis; no threshold or model changes",
    )
    write_json(OUTPUT / "report/boundary_diagnostic.json", result)
    write_json(ROOT / "docs/benchmarks/template_rescue_boundary.json", result)


def analyze():
    with (OUTPUT / "report/per_unit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {
        (r["subject"], r["condition"], int(r["snr_db"]), int(r["unit"]), r["method"]): r
        for r in rows
    }
    if len(indexed) != 3546:
        raise ValueError("Expected six methods over the same 591 instances")
    rng = np.random.default_rng(2028)
    comparisons = {}
    for method, comparator in [
        ("rescue", "pareto"),
        ("rescue", "previous"),
        ("protected", "previous"),
        ("physical", "protected"),
        ("rescue", "physical"),
        ("previous", "constant"),
    ]:
        differences = {s: [] for s in SUBJECTS[1:]}
        for key, row in indexed.items():
            if row["method"] == method:
                other = indexed[(*key[:4], comparator)]
                differences[row["subject"]].append((float(row["roa"]) - float(other["roa"])) * 100)
        clusters = [np.asarray(differences[s]) for s in SUBJECTS[1:]]
        bootstrap = [
            np.concatenate([clusters[i] for i in rng.integers(0, 4, 4)]).mean() for _ in range(2000)
        ]
        comparisons[f"{method}_minus_{comparator}"] = dict(
            mean_delta_pp=float(np.concatenate(clusters).mean()),
            by_subject_delta_pp={s: float(np.mean(differences[s])) for s in SUBJECTS[1:]},
            subject_cluster_bootstrap_95_ci_pp=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        )
    totals = dict(added_events=0, net_true_positives=0, net_false_positives=0)
    by_condition = {}
    for path in sorted((OUTPUT / "predictions/metadata").glob("*.json")):
        name = path.stem
        metadata = json.loads(path.read_text())
        with np.load(OUTPUT / "predictions/physical" / f"{name}.npz") as archive:
            physical = archive["spikes"].astype(bool)
        with np.load(OUTPUT / "predictions/rescue" / f"{name}.npz") as archive:
            rescued = archive["spikes"].astype(bool)
        if np.any(physical & ~rescued):
            raise ValueError("The recovery branch removed primary discharges")
        added = int(np.count_nonzero(rescued & ~physical))
        if added != metadata["methods"]["rescue"]["recovered_candidate_events"]:
            raise ValueError("Recovery event count differs from the prediction archive")
        preparation = metadata["inputs"]["previous_preparation"]
        subject, condition, snr = (preparation[k] for k in ("subject", "condition", "snr_db"))
        if stub(subject, condition, snr) != name:
            raise ValueError("Record identity changed")
        tp_delta = 0
        for unit, (n_physical, n_rescue) in enumerate(
            zip(physical.sum(0), rescued.sum(0), strict=True)
        ):
            p = indexed[(subject, condition, snr, unit, "physical")]
            r = indexed[(subject, condition, snr, unit, "rescue")]
            tp_delta += round(float(r["precision"]) * n_rescue) - round(
                float(p["precision"]) * n_physical
            )
        values = dict(
            added_events=added, net_true_positives=tp_delta, net_false_positives=added - tp_delta
        )
        condition_total = by_condition.setdefault(condition, dict.fromkeys(totals, 0))
        for key, value in values.items():
            totals[key] += int(value)
            condition_total[key] += int(value)
    result = dict(
        analysis_code_sha256=sha256(__file__),
        per_unit_sha256=sha256(OUTPUT / "report/per_unit.csv"),
        comparisons=comparisons,
        recovery_increment=totals,
        recovery_increment_by_condition=by_condition,
        alignment_caveat=(
            "Net TP/FP increments use each method's official whole-record alignment; "
            "they do not independently label individual added events."
        ),
        statistical_caveat=(
            "Only four subject clusters, overlapping training folds and previously seen data; "
            "bootstrap intervals are descriptive, not independent population confirmation."
        ),
        selection_used=False,
    )
    write_json(OUTPUT / "report/diagnostic_analysis.json", result)
    write_json(ROOT / "docs/benchmarks/template_rescue_analysis.json", result)
    experiment = json.loads((OUTPUT / "report/summary.json").read_text())
    summary = experiment["summaries"]
    methods = ["previous", "constant", "protected", "physical", "rescue"]
    conditions = ["staircase", "triangular-ramp10s", "triangular-ramp5s"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    positions = np.arange(3)
    for index, method in enumerate(methods):
        delta = [
            summary[method]["by_condition"][c] - summary["pareto"]["by_condition"][c]
            for c in conditions
        ]
        axes[0].bar(positions + (index - 2) * 0.15, delta, width=0.14, label=method)
    axes[0].scatter(
        positions, [0, 5, 5], marker="D", color="black", s=35, label="required gain", zorder=5
    )
    axes[0].set_xticks(positions, ["Staircase", "10 s", "5 s"])
    axes[0].set(
        ylabel="RoA difference vs Pareto (points)",
        ylim=(-1, 6),
        title="Predeclared movement targets",
    )
    counts = [summary[m]["end_units_roa_ge_90"] for m in methods]
    axes[1].bar(methods, np.asarray(counts) - 511, color=[f"C{i}" for i in range(len(methods))])
    for index, count in enumerate(counts):
        axes[1].annotate(
            str(count), (index, count - 511), xytext=(0, 5), textcoords="offset points", ha="center"
        )
    axes[1].set(
        ylabel="Final retained instances minus Pareto's 511",
        ylim=(-12, 2),
        title="Final RoA >=90% (591 calibrated instances)",
    )
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].tick_params(axis="x", rotation=20)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.suptitle(f"Frozen template-rescue experiment: {experiment['status']}")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUTPUT / "report/deltas.png", dpi=180)
    fig.savefig(ROOT / "docs/images/template-rescue-deltas.png", dpi=180)
    plt.close(fig)
    boundary_diagnostic(rows)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    analyze()
