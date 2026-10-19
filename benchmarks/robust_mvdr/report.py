"""Aggregate scored records into per-instance and summary tables."""

import csv
import json
from collections import defaultdict

import numpy as np

from . import evaluate
from .common import METHODS, OUTPUT, PRIMARY, stub, write_json

PHASES = (
    "first_hold_roa",
    "ramp_roa",
    "last_hold_roa",
    "angle_reference_rest_roa",
    "angle_moving_roa",
    "angle_hold_away_roa",
)


def _mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) * 100 if values else None


def report():
    evaluate.checked()
    rows = []
    for subject, condition, snr in evaluate.RECORDS:
        path = OUTPUT / "scored" / f"{stub(subject, condition, snr)}.json"
        rows += json.loads(path.read_text())["rows"]
    methods = sorted({r["method"] for r in rows})
    keys = {
        m: {
            (r["subject"], r["condition"], r["snr_db"], r["unit"]) for r in rows if r["method"] == m
        }
        for m in methods
    }
    if any(len(k) != len(keys["pareto"]) or k != keys["pareto"] for k in keys.values()):
        raise ValueError("Every method must cover the same calibrated instances")
    by = defaultdict(dict)
    for r in rows:
        by[r["method"]][(r["subject"], r["condition"], r["snr_db"], r["unit"])] = r
    summary = dict(primary=PRIMARY, instances=len(keys["pareto"]), methods={})
    for method in methods:
        values = list(by[method].values())
        entry = dict(
            mean_roa_pct=_mean(v["roa"] for v in values),
            median_roa_pct=float(np.median([v["roa"] for v in values])) * 100,
            pct_instances_roa_ge_90=float(np.mean([v["roa"] >= 0.9 for v in values])) * 100,
            mean_precision_pct=_mean(v["precision"] for v in values),
            mean_recall_pct=_mean(v["recall"] for v in values),
            by_condition={
                c: _mean(v["roa"] for v in values if v["condition"] == c)
                for c in sorted({v["condition"] for v in values})
            },
            by_snr={
                str(s): _mean(v["roa"] for v in values if v["snr_db"] == s)
                for s in sorted({v["snr_db"] for v in values})
            },
            by_phase={p: _mean(v.get(p) for v in values) for p in PHASES},
        )
        if method != "pareto":
            deltas = [by[method][k]["roa"] - by["pareto"][k]["roa"] for k in by["pareto"]]
            entry["delta_vs_pareto_pp"] = float(np.mean(deltas)) * 100
            entry["improved_vs_pareto"] = int(sum(d > 0 for d in deltas))
            entry["worsened_vs_pareto"] = int(sum(d < 0 for d in deltas))
        if "stage_b_static_analytic" in by and method in METHODS:
            reference = by["stage_b_static_analytic"]
            deltas = [by[method][k]["roa"] - reference[k]["roa"] for k in reference]
            entry["delta_vs_stage_b_static_pp"] = float(np.mean(deltas)) * 100
        summary["methods"][method] = entry
    OUTPUT.joinpath("report").mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "report/per_unit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(OUTPUT / "report/summary.json", summary)
    for method in methods:
        e = summary["methods"][method]
        print(
            f"{method:28s} RoA {e['mean_roa_pct']:6.2f}  ramp5s "
            f"{e['by_condition'].get('triangular-ramp5s', float('nan')):6.2f}  "
            f"moving {e['by_phase']['angle_moving_roa'] or float('nan'):6.2f}  "
            f"recall {e['mean_recall_pct']:6.2f}  precision {e['mean_precision_pct']:6.2f}"
        )
    return summary
