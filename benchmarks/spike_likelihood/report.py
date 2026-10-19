import importlib.metadata
import json
import platform
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.stage_a.evaluate import paired_statistics, summarize, write_csv

from .common import CONDITIONS, OUTPUT, ROOT, SUBJECTS, sha256, write_json
from .experiment import checked


def report():
    checked()
    paths = list((OUTPUT / "scored").glob("*.json"))
    if len(paths) != 80:
        raise ValueError("All 80 records are required")
    rows = [r for p in paths for r in json.loads(p.read_text())["rows"]]
    summaries = summarize(rows)
    if set(summaries) != {"posterior", "rescue_union", "pareto", "previous_b"}:
        raise ValueError("Unexpected methods")

    def key(r):
        return r["subject"], r["condition"], r["snr_db"], r["unit"]

    baseline = {key(r): r for r in rows if r["method"] == "pareto"}
    for method, summary in summaries.items():
        selected = [r for r in rows if r["method"] == method]
        if len(selected) != 591 or len({key(r) for r in selected}) != 591:
            raise ValueError("Missing or duplicate instances")
        summary.update(
            mean_precision=float(np.mean([r["precision"] for r in selected])),
            mean_recall=float(np.mean([r["recall"] for r in selected])),
            mean_f1=float(np.mean([r["f1"] for r in selected])),
            by_subject={
                s: float(np.mean([r["roa"] for r in selected if r["subject"] == s]) * 100)
                for s in SUBJECTS[1:]
            },
            end_units_roa_ge_90=sum(r["last_hold_roa"] >= 0.9 for r in selected),
            final_gained=sum(
                r["last_hold_roa"] >= 0.9 and baseline[key(r)]["last_hold_roa"] < 0.9
                for r in selected
            ),
            final_lost=sum(
                r["last_hold_roa"] < 0.9 and baseline[key(r)]["last_hold_roa"] >= 0.9
                for r in selected
            ),
        )
        for phase in ("first_hold_roa", "ramp_roa", "last_hold_roa"):
            summary["mean_" + phase] = float(np.mean([r[phase] for r in selected]))
    p = summaries["posterior"]
    gates = {}
    for name, observed, minimum in [
        ("overall", p["mean_roa_pct"], 85),
        ("ramp5s", p["by_condition"]["triangular-ramp5s"], 80),
        ("units_ge90", p["pct_units_roa_ge_90"], 70),
    ]:
        gates[name] = dict(observed=observed, minimum=minimum, passed=observed >= minimum)
    result = dict(
        status="incomplete" if all(g["passed"] for g in gates.values()) else "no-go",
        summaries=summaries,
        accuracy_gates=gates,
        primary="posterior",
        delta_vs_pareto_pp=p["mean_roa_pct"] - summaries["pareto"]["mean_roa_pct"],
        delta_vs_previous_b_pp=p["mean_roa_pct"] - summaries["previous_b"]["mean_roa_pct"],
        statistics=paired_statistics(rows, method="posterior", seed=2031),
        protocol_sha256=sha256(OUTPUT / "protocol.json"),
        locked_sha256=sha256(OUTPUT / "locked.json"),
        manifest_sha256=sha256(OUTPUT / "predictions_complete.json"),
        scope=(
            "supervised event likelihood over cached EMG source views; "
            "exploratory same-generator transfer"
        ),
        unverified=[
            "independent generator/data",
            "1s/2.5s true-generator ramps",
            "full raw-stream CPU latency",
        ],
        sil_scope="combined event decisions have no single synthesized source; SIL not assigned",
    )
    folder = OUTPUT / "report"
    write_json(folder / "summary.json", result)
    write_csv(folder / "per_unit.csv", rows)
    environment = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        inference_device="cpu",
        versions={k: importlib.metadata.version(k) for k in ("torch", "numpy", "scipy")},
        training={
            str(p.parent.relative_to(OUTPUT)): json.loads(p.read_text())
            for p in (OUTPUT / "folds").glob("*/*/complete.json")
        },
    )
    write_json(folder / "environment.json", environment)
    history = {
        str(p.parent.relative_to(OUTPUT)): json.loads(p.read_text())
        for p in (OUTPUT / "folds").glob("*/*/history.json")
    }
    write_json(folder / "training.json", history)
    timing = [json.loads(p.read_text()) for p in (OUTPUT / "predictions/metadata").glob("*.json")]
    write_json(folder / "timing.json", timing)
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for method in ("pareto", "previous_b", "posterior", "rescue_union"):
        ax.plot(
            range(5), [summaries[method]["by_condition"][c] for c in CONDITIONS], "o-", label=method
        )
    ax.set_xticks(range(5), ["Staircase", "40 s", "20 s", "10 s", "5 s"])
    ax.set(ylabel="Mean RoA (%)", ylim=(0, 100), title="True-event likelihood repair")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)
    for source, name in [
        (OUTPUT / "protocol.json", "protocol.json"),
        (OUTPUT / "locked.json", "locked.json"),
    ] + [
        (folder / name, name)
        for name in (
            "summary.json",
            "per_unit.csv",
            "environment.json",
            "training.json",
            "timing.json",
        )
    ]:
        shutil.copy2(source, ROOT / "docs/benchmarks" / f"spike_likelihood_{name}")
    shutil.copy2(folder / "comparison.png", ROOT / "docs/images/spike-likelihood.png")
    print(
        json.dumps(
            {
                "status": result["status"],
                "means": {m: r["mean_roa_pct"] for m, r in summaries.items()},
                "gates": gates,
            },
            indent=2,
        )
    )
    return result
