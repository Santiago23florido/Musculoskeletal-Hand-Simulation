"""Read sealed predictions to publish compact level tables and preservation checks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.pareto_residual.detector import Detector

from .common import (
    OUTPUT,
    ROOT,
    VALIDATION,
    now,
    pareto_path,
    records,
    register,
    sha256,
    stub,
    write_new,
)
from .evaluate import score_record, summarize
from .events import descriptor


def baseline_validation():
    """Published predictions verified in L0; no retraining or outer-subject labels."""
    from adapt_decomp import AdaptationResult

    folder = OUTPUT / "L0/validation"
    if (folder / "summary.json").exists():
        return json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if not (folder / "protocol.json").exists():
        register(
            folder,
            dict(
                reference="Pareto-rerun",
                cases=records(VALIDATION),
                basis="exact agreement with published predictions on all three L0 controls",
                original_scorer="rate_of_agreement_paired fs=2048 tol_spike_ms=2",
                timing="L0 raw-CPU timing on the three sub-01 identity controls; not cached replay",
            ),
        )
    files = {}
    for case in records(VALIDATION):
        name = stub(*case)
        path = folder / "predictions" / f"{name}.npz"
        if not path.exists():
            record = descriptor(*case, phase="validation")
            baseline = AdaptationResult.load(pareto_path(*case))
            sources = baseline.sources.numpy()
            detector = Detector(record["calibration"], record["config"])
            centroids, trusted = [], []
            batch = record["config"].batch_size
            for i, start in enumerate(range(0, len(sources), batch)):
                trim = record["config"].ext_fact if i == 0 else 0
                _, mask = detector.step(torch.from_numpy(sources[start + trim : start + batch]), i)
                centroids.append(
                    torch.stack([detector.decomp.base_centr, detector.decomp.spikes_centr]).numpy()
                )
                trusted.append(mask.sum(0).numpy())
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                spikes=baseline.spikes.numpy().astype(np.uint8),
                sources=sources,
                centroid=np.asarray(centroids),
                trusted_count=np.asarray(trusted),
            )
        files[path.relative_to(folder).as_posix()] = sha256(path)
    if not (folder / "sealed.json").exists():
        write_new(folder / "sealed.json", dict(sealed_at_utc=now(), files=files))
    seal = json.loads((folder / "sealed.json").read_text())
    results = []
    for case in records(VALIDATION):
        name = stub(*case)
        path = folder / "predictions" / f"{name}.npz"
        if sha256(path) != seal["files"][path.relative_to(folder).as_posix()]:
            raise ValueError("Baseline prediction seal changed")
        record = descriptor(*case, phase="validation")
        with np.load(path) as archive:
            prediction = {key: archive[key] for key in archive.files}
        score_path = folder / "scores" / f"{name}.json"
        if score_path.exists():
            result = json.loads(score_path.read_text())
        else:
            result = score_record(record, prediction, phase="validation")
            write_new(score_path, result)
        results.append(result)
    summary = summarize(results)
    timings = json.loads((OUTPUT / "L0/result.json").read_text())["records"]
    summary["median_batch_ms"] = float(np.median([r["median_batch_ms"] for r in timings]))
    summary["p95_batch_ms"] = max(r["p95_batch_ms"] for r in timings)
    summary["timing_scope"] = (
        "median of 3 sub-01 medians; maximum p95 of those controls; CPU 4 threads"
    )
    summary["memory_hit_rate"] = None
    summary["controller"] = "identity: wh=1, sv=1, beta=0; no dependence on blind speed"
    write_new(folder / "summary.json", summary)
    return summary


def export(destination):
    destination = Path(destination)
    l0 = baseline_validation()
    levels = {"L0": l0}
    for variant in ("all_feedback", "guarded_feedback"):
        path = OUTPUT / "L1" / variant / "gate.json"
        if path.exists():
            gate = json.loads(path.read_text())
            levels[f"L1/{variant}"] = dict(**gate["actual"], gate_passed=gate["passed"])
    original = json.loads((OUTPUT / "L0/protocol.json").read_text())["protected"]
    changed = [name for name, digest in original.items() if sha256(ROOT / name) != digest]
    if changed:
        raise ValueError(f"Historical tracked files changed: {changed}")
    write_new(
        destination / "pareto_mc_summary.json",
        dict(
            generated_at_utc=now(),
            levels=levels,
            validation_subject="sub-04",
            scope="exploratory development only; no external 80-record result is implied",
            historical_files_verified=len(original),
            historical_files_changed=changed,
            confirmation=None,
        ),
    )
    for relative in ("L0/protocol.json", "L0/result.json", "C1/seed0/complete.json", "phase0.json"):
        value = json.loads((OUTPUT / relative).read_text())
        write_new(
            destination
            / ("pareto_mc_" + relative.replace("/", "_").replace(".json", "") + ".json"),
            value,
        )
    print(json.dumps(levels, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.output is None:
        print(json.dumps(baseline_validation(), indent=2), flush=True)
    else:
        export(args.output)


if __name__ == "__main__":
    main()
