"""Descriptive analysis after frozen event-likelihood scoring; no fitting."""

import csv
import json

import numpy as np
import torch

from benchmarks.spike_likelihood.common import OUTPUT, ROOT, SUBJECTS, sha256, write_json
from benchmarks.spike_likelihood.experiment import checked


def main():
    checked()
    rows = list(csv.DictReader((OUTPUT / "report/per_unit.csv").open(encoding="utf-8")))

    def key(r):
        return r["subject"], r["condition"], r["snr_db"], r["unit"]

    baseline = {key(r): r for r in rows if r["method"] == "pareto"}
    primary = {key(r): r for r in rows if r["method"] == "posterior"}
    assert baseline.keys() == primary.keys() and len(primary) == 591
    delta = {k: (float(primary[k]["roa"]) - float(baseline[k]["roa"])) * 100 for k in primary}
    clusters = [np.asarray([v for k, v in delta.items() if k[0] == s]) for s in SUBJECTS[1:]]
    rng = np.random.default_rng(2033)
    boot = [
        float(np.concatenate([clusters[j] for j in rng.integers(0, 4, 4)]).mean())
        for _ in range(2000)
    ]
    selections = {}
    for p in (OUTPUT / "folds").glob("*/*/model.pt"):
        saved = torch.load(p, weights_only=True, map_location="cpu")
        selections[str(p.parent.relative_to(OUTPUT))] = dict(
            epoch=saved["epoch"], selection=saved["selection"], model_sha256=sha256(p)
        )
    event_counts = {
        method: dict(tp=0, fp=0, fn=0) for method in ("pareto", "posterior", "rescue_union")
    }
    changed = dict(added_sample_events=0, removed_sample_events=0)
    for metadata in (OUTPUT / "predictions/metadata").glob("*.json"):
        name = metadata.stem
        subject = name.split("_", 1)[0]
        candidates = OUTPUT / "features" / name / "events.npz"
        with np.load(candidates) as a:
            units = a["unit"]
            primary_flags = a["primary"]
            times = a["time"]
        unit_rows = {
            int(r["unit"]): r
            for r in rows
            if r["subject"] == subject
            and r["method"] == "pareto"
            and name.endswith(f"_snr{r['snr_db']}dB")
            and f"_FDSI_{r['condition']}_" in name
        }
        with np.load(OUTPUT / "predictions/posterior" / f"{name}.npz") as a:
            posterior = a["spikes"]
        reference = np.zeros_like(posterior)
        reference[times[primary_flags], units[primary_flags]] = 1
        changed["added_sample_events"] += int(np.count_nonzero(posterior & ~reference))
        changed["removed_sample_events"] += int(np.count_nonzero(reference & ~posterior))
        for method in event_counts:
            if method == "pareto":
                counts = reference.sum(0)
            else:
                with np.load(OUTPUT / "predictions" / method / f"{name}.npz") as a:
                    counts = a["spikes"].sum(0)
            for j, base in unit_rows.items():
                row = next(r for r in rows if key(r) == key(base) and r["method"] == method)
                base_tp = round(float(base["precision"]) * int(reference[:, j].sum()))
                assert float(base["recall"]) > 0
                nr = round(base_tp / float(base["recall"]))
                tp = round(float(row["precision"]) * int(counts[j]))
                event_counts[method]["tp"] += tp
                event_counts[method]["fp"] += int(counts[j]) - tp
                event_counts[method]["fn"] += nr - tp
    result = dict(
        scope="descriptive analysis; frozen predictions; no model or threshold selection",
        mean_delta_pp=float(np.mean(list(delta.values()))),
        improved_instances=sum(v > 1e-10 for v in delta.values()),
        worsened_instances=sum(v < -1e-10 for v in delta.values()),
        unchanged_instances=sum(abs(v) <= 1e-10 for v in delta.values()),
        subject_cluster_bootstrap_95_ci_pp=np.quantile(boot, [0.025, 0.975]).tolist(),
        selected_checkpoints=selections,
        counts=event_counts,
        sample_changes=changed,
        delta_counts_vs_pareto={
            k: event_counts["posterior"][k] - event_counts["pareto"][k] for k in ("tp", "fp", "fn")
        },
        count_interpretation=(
            "counts use each method's official alignment; sample additions include retiming, "
            "not independently classified extra events"
        ),
        limitations=(
            "four subjects, overlapping fitting folds, previously explored generator; "
            "descriptive confidence interval"
        ),
        summary_sha256=sha256(OUTPUT / "report/summary.json"),
    )
    write_json(OUTPUT / "report/analysis.json", result)
    write_json(ROOT / "docs/benchmarks/spike_likelihood_analysis.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("selected_checkpoints",)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
