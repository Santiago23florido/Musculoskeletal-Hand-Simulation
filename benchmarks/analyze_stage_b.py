"""Descriptive analysis after locked Stage B scoring; never selects model weights."""

import json

import numpy as np

from benchmarks.stage_b.common import CONDITIONS, OUTPUT, ROOT, SUBJECTS, sha256, write_json
from benchmarks.stage_b.experiment import checked


def instance(row):
    return row["subject"], row["condition"], row["snr_db"], row["unit"]


def compare(rows, method, reference):
    a = {instance(r): r for r in rows if r["method"] == method}
    b = {instance(r): r for r in rows if r["method"] == reference}
    if a.keys() != b.keys() or len(a) != 591:
        raise ValueError("Paired analysis needs the same 591 instances")
    differences = np.asarray([a[k]["roa"] - b[k]["roa"] for k in a]) * 100
    return dict(
        method=method,
        reference=reference,
        mean_delta_pp=float(differences.mean()),
        median_delta_pp=float(np.median(differences)),
        improved_instances=int((differences > 1e-10).sum()),
        worsened_instances=int((differences < -1e-10).sum()),
        unchanged_instances=int((np.abs(differences) <= 1e-10).sum()),
        by_subject={
            s: float(np.mean([a[k]["roa"] - b[k]["roa"] for k in a if k[0] == s]) * 100)
            for s in SUBJECTS[1:]
        },
        phase_delta_pp={
            p: float(np.mean([a[k][p] - b[k][p] for k in a]) * 100)
            for p in ("first_hold_roa", "ramp_roa", "last_hold_roa")
        },
    )


def main():
    checked()
    if not (OUTPUT / "report/summary.json").exists():
        raise ValueError("Complete and report the frozen experiment first")
    paths = sorted((OUTPUT / "scored").glob("*.json"))
    if len(paths) != 80:
        raise ValueError("Incomplete scoring")
    documents = [json.loads(p.read_text()) for p in paths]
    rows = [r for document in documents for r in document["rows"]]
    methods = ("pareto", "moca_k", "moca_b", "moca_b_fifo", "static_analytic")
    cells = []
    for condition in CONDITIONS:
        for snr in (30, 25, 20, 15):
            values = [r for r in rows if r["condition"] == condition and r["snr_db"] == snr]
            cells.append(
                dict(
                    condition=condition,
                    snr_db=snr,
                    mean_roa_pct={
                        m: float(np.mean([r["roa"] for r in values if r["method"] == m]) * 100)
                        for m in methods
                    },
                )
            )
    state = []
    for path, document in zip(paths, documents, strict=True):
        exemplar = document["rows"][0]
        predictions = OUTPUT / "predictions"
        with np.load(predictions / "moca_b" / f"{path.stem}.npz") as a:
            with np.load(predictions / "moca_b_fifo" / f"{path.stem}.npz") as b:
                if not np.array_equal(a["q"], b["q"]):
                    raise ValueError("The covariance ablation changed the independent observer")
        state.append(
            dict(
                record=path.stem,
                subject=exemplar["subject"],
                condition=exemplar["condition"],
                snr_db=exemplar["snr_db"],
                **document["state"]["moca_b"],
            )
        )
    result = dict(
        scope="descriptive post-scoring analysis of frozen predictions; no parameter selection",
        summary_sha256=sha256(OUTPUT / "report/summary.json"),
        comparisons=[
            compare(rows, "moca_b", r)
            for r in ("pareto", "moca_k", "moca_b_fifo", "static_analytic")
        ],
        condition_noise_cells=cells,
        state=state,
        state_rmse_by_condition={
            c: float(np.mean([r["angle_rmse_degrees"] for r in state if r["condition"] == c]))
            for c in CONDITIONS
        },
        state_rmse_by_snr={
            str(s): float(np.mean([r["angle_rmse_degrees"] for r in state if r["snr_db"] == s]))
            for s in (30, 25, 20, 15)
        },
        b_and_fifo_state_sequences_identical=True,
        limitations=(
            "same previously explored generator; four outer subjects with shared fitting data"
        ),
    )
    write_json(OUTPUT / "report/analysis.json", result)
    write_json(ROOT / "docs/benchmarks/stage_b_analysis.json", result)
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
