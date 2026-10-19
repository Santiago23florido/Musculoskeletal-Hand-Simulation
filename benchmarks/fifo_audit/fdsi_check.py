"""Read existing sub-01 feature caches only; never prepare or write Stage B data."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from benchmarks.robust_mvdr.common import METHODS
from benchmarks.robust_mvdr.inference import predict as robust_predict
from benchmarks.stage_a.common import DATA, ROOT, calibration_path, sha256, stub
from benchmarks.stage_a.data import load_calibration
from benchmarks.stage_b import data as stage_b_data
from benchmarks.stage_b.common import CONDITIONS, DIMENSION
from benchmarks.stage_b.inference import predict as stage_b_predict

from .harness import (
    VARIANTS,
    amplitude_metrics,
    compare_predictions,
    decode,
    provenance,
    score_window,
    write_new_json,
)


def require_sub01(subject):
    if subject != "sub-01":
        raise ValueError(
            f"FIFO audit permits only sub-01; refused {subject!r}. Outer subjects are forbidden."
        )


def cached_record(subject, condition):
    """Equivalent to prepare's cache-read branch, with no cache-creation path."""
    require_sub01(subject)
    folder = ROOT / "outputs/stage_b_main/features" / stub(subject, condition, 30)
    path = folder / "metadata.json"
    if not path.is_file():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if (
        metadata["code_sha256"] != sha256(stage_b_data.__file__)
        or metadata["dimension"] != DIMENSION
    ):
        raise ValueError(f"Existing cache provenance mismatch: {path}")
    if metadata["calibration_sha256"] != sha256(calibration_path(subject, condition, 30)):
        raise ValueError(f"Existing cache calibration mismatch: {path}")
    calibration, config = load_calibration(subject, condition, 30)
    arrays = {
        key: torch.from_numpy(np.load(folder / f"{key}.npy", mmap_mode="c"))
        for key in ("x", "features", "q", "template", "amplitude", "projection", "x_cal")
    }
    return dict(**arrays, calibration=calibration, config=config, metadata=metadata)


def metrics(record, truth, prediction):
    fs = record["config"].fs
    duration = len(record["x"]) / fs
    return dict(
        from_5s=score_window(truth, prediction["spikes"], fs, 5),
        from_10s=score_window(truth, prediction["spikes"], fs, 10),
        amplitude=amplitude_metrics(
            prediction["sources"],
            truth,
            fs,
            dict(calibration=(1, 5), first_second=(5, 6), rest=(6, duration)),
        ),
    )


def classify_controls(records):
    """Declare the sub-01 case from instance-weighted real-predict post-5 s RoA."""
    if not records:
        raise ValueError("No existing sub-01 cache; no diagnostic case can be evaluated")
    summary = {}
    for method in ("stage_b_static", "static_fifo_mvdr"):
        values = [row["real_predict_controls"][method] for row in records.values()]
        summary[method] = dict(
            roa_from_5s=float(
                np.mean([v for row in values for v in row["from_5s"]["roa_per_unit"]])
            ),
            roa_from_10s=float(
                np.mean([v for row in values for v in row["from_10s"]["roa_per_unit"]])
            ),
            first_second_amplitude_ratio=float(
                np.median(
                    [
                        v
                        for row in values
                        for v in row["amplitude"]["first_second"]["per_unit_ratio_vs_calibration"]
                        if v is not None
                    ]
                )
            ),
            rest_amplitude_ratio=float(
                np.median(
                    [
                        v
                        for row in values
                        for v in row["amplitude"]["rest"]["per_unit_ratio_vs_calibration"]
                        if v is not None
                    ]
                )
            ),
        )
    fifo = summary["static_fifo_mvdr"]["roa_from_5s"]
    stage_b = summary["stage_b_static"]["roa_from_5s"]
    case = "C" if fifo >= 0.2 else "B" if stage_b < 0.2 else "A"
    return dict(
        case=case,
        summary=summary,
        collapse_roa_below=0.2,
        scope="mean across calibrated instances in all existing sub-01 30 dB caches, from 5 s",
        lower_fifo_amplitude=summary["static_fifo_mvdr"]["rest_amplitude_ratio"]
        < summary["stage_b_static"]["rest_amplitude_ratio"],
    )


def run_check(output, subject="sub-01"):
    require_sub01(subject)
    from adapt_decomp.utils import load_gt

    tick = time.perf_counter()
    report = dict(
        started_at_utc=datetime.now(UTC).isoformat(),
        provenance=provenance(),
        subject=subject,
        snr_db=30,
        records={},
        skipped=[],
        interpretation="sub-01 diagnostic only; no outer subjects, training, or new Stage B caches",
        truth_selection='load_gt(...)[:, record["calibration"].gt_matched_indices]',
        declared_decision="A: FIFO RoA <0.20, Stage B >=0.20; B: both <0.20; C: FIFO >=0.20, stop",
    )
    loaded = {}
    for condition in CONDITIONS:
        record = cached_record(subject, condition)
        if record is None:
            report["skipped"].append(
                dict(condition=condition, reason="Stage B feature cache absent")
            )
            print("skip", subject, condition, "existing Stage B cache absent", flush=True)
            continue
        name = stub(subject, condition, 30)
        print("diagnose", name, flush=True)
        truth = load_gt(
            DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_spikes.npz",
            n_samples=len(record["x"]),
        )[:, record["calibration"].gt_matched_indices]
        controls = dict(
            stage_b_static=stage_b_predict(record, [], mode="static"),
            static_fifo_mvdr=robust_predict(record, [], METHODS["static_fifo_mvdr"]),
        )
        result = dict(
            cache_metadata=record["metadata"],
            units=truth.shape[1],
            methods={},
            real_predict_controls={
                key: metrics(record, truth, value) for key, value in controls.items()
            },
            replay_equivalence={},
        )
        report["records"][name] = result
        loaded[name] = (record, truth, controls)
        for method, entry in result["real_predict_controls"].items():
            print(
                "real control",
                name,
                method,
                "RoA 5s/10s",
                entry["from_5s"]["roa_mean"],
                entry["from_10s"]["roa_mean"],
                "amplitude first/rest",
                entry["amplitude"]["first_second"]["median_ratio_vs_calibration"],
                entry["amplitude"]["rest"]["median_ratio_vs_calibration"],
                flush=True,
            )
    report["decision"] = classify_controls(report["records"])
    print("sub-01 decision", report["decision"], flush=True)
    if report["decision"]["case"] == "C":
        report["stopped_case_c"] = True
        report["elapsed_seconds"] = time.perf_counter() - tick
        write_new_json(Path(output) / "fifo_audit_sub01.json", report)
        return report
    for name, (record, truth, controls) in loaded.items():
        result = report["records"][name]
        for method, variant in VARIANTS.items():
            prediction = decode(record, variant)
            if method in controls:
                result["replay_equivalence"][method] = compare_predictions(
                    prediction, controls[method]
                )
            entry = metrics(record, truth, prediction)
            entry["detector"] = prediction["detector"]
            result["methods"][method] = entry
            print(
                name,
                method,
                "RoA 5s/10s",
                round(entry["from_5s"]["roa_mean"], 6),
                round(entry["from_10s"]["roa_mean"], 6),
                "amplitude first/rest",
                round(entry["amplitude"]["first_second"]["median_ratio_vs_calibration"], 6),
                round(entry["amplitude"]["rest"]["median_ratio_vs_calibration"], 6),
                flush=True,
            )
    report["elapsed_seconds"] = time.perf_counter() - tick
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    destination = Path(output) / "fifo_audit_sub01.json"
    write_new_json(destination, report)
    print("saved", destination, "seconds", report["elapsed_seconds"], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/benchmarks")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    require_sub01(args.subject)
    torch.set_num_threads(args.threads)
    report = run_check(args.output, args.subject)
    if report["decision"]["case"] == "C":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
