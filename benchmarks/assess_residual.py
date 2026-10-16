"""Additional user requirements, declared before any outer-test scoring."""

import argparse
import json
from datetime import UTC, datetime

import pandas as pd

from benchmarks.pareto_residual.common import OUTPUT, sha256, write_json


def declare():
    if list((OUTPUT / "evaluations").glob("*/records/*.json")):
        raise ValueError("Declare acceptance requirements before scoring the outer tests")
    path = OUTPUT / "acceptance.json"
    if path.exists():
        raise ValueError("Additional criteria already declared")
    write_json(
        path,
        dict(
            declared_at_utc=datetime.now(UTC).isoformat(),
            protocol_sha256=sha256(OUTPUT / "protocol.json"),
            requirements={
                "global_mean_delta_pp_strictly_greater_than": 0,
                "unseen_15dB_delta_pp_at_least": 0,
            },
            note=(
                "Additional user success conditions; no change to training, "
                "validation or model selection."
            ),
            code_sha256=sha256(__file__),
        ),
    )


def assess():
    acceptance = json.loads((OUTPUT / "acceptance.json").read_text())
    if acceptance["protocol_sha256"] != sha256(OUTPUT / "protocol.json"):
        raise ValueError("Protocol changed")
    if acceptance["code_sha256"] != sha256(__file__):
        raise ValueError("Assessment implementation changed after declaring requirements")
    summary = json.loads((OUTPUT / "report/summary.json").read_text())
    rows = pd.read_csv(OUTPUT / "report/per_unit.csv")
    actual, baseline = summary["summaries"]["hybrid"], summary["summaries"]["pareto"]
    delta = actual["mean_roa_pct"] - baseline["mean_roa_pct"]
    noise_delta = actual["by_snr"]["15"] - baseline["by_snr"]["15"]
    checks = dict(summary["gates"])
    checks["global_mean"] = dict(delta_pp=delta, passed=delta > 0, requirement="strictly >0")
    checks["unseen_15dB"] = dict(delta_pp=noise_delta, passed=noise_delta >= 0, requirement=">=0")
    index = ["subject", "condition", "snr_db", "unit"]
    p = rows[rows.method == "pareto"].set_index(index)
    h = rows[rows.method == "hybrid"].set_index(index).loc[p.index]
    changes = dict(
        improved_instances=int((h.roa > p.roa).sum()),
        worsened_instances=int((h.roa < p.roa).sum()),
        unchanged_instances=int((h.roa == p.roa).sum()),
        gained_final_90=int(((p.last_hold_roa < 0.9) & (h.last_hold_roa >= 0.9)).sum()),
        lost_final_90=int(((p.last_hold_roa >= 0.9) & (h.last_hold_roa < 0.9)).sum()),
    )
    by_condition = {}
    for condition in p.index.get_level_values("condition").unique():
        a, b = h.xs(condition, level="condition"), p.xs(condition, level="condition")
        by_condition[condition] = dict(
            instances=len(a),
            pareto_mean_pct=float(b.roa.mean() * 100),
            hybrid_mean_pct=float(a.roa.mean() * 100),
            pareto_ge_90_pct=float((b.roa >= 0.9).mean() * 100),
            hybrid_ge_90_pct=float((a.roa >= 0.9).mean() * 100),
        )
    result = dict(
        status="go" if all(c["passed"] for c in checks.values()) else "no-go",
        acceptance_sha256=sha256(OUTPUT / "acceptance.json"),
        summary_sha256=sha256(OUTPUT / "report/summary.json"),
        per_unit_sha256=sha256(OUTPUT / "report/per_unit.csv"),
        checks=checks,
        unit_changes=changes,
        by_condition=by_condition,
        by_snr_instance_counts={
            str(s): int((p.index.get_level_values("snr_db") == s).sum()) for s in (15, 20, 25, 30)
        },
        interpretation="Exploratory transfer to held-out subjects; not a fresh blind data set.",
    )
    write_json(OUTPUT / "report/assessment.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true")
    args = parser.parse_args()
    declare() if args.declare else assess()
