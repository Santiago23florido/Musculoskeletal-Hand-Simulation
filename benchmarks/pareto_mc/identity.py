"""L0 equivalence to a fresh, genuinely causal pinned Pareto execution."""

import copy
import shutil
import time

import numpy as np

from .backbone import AdaptDecomp, build, replay
from .common import (
    OUTPUT,
    ROOT,
    now,
    pareto_path,
    protected_hashes,
    sha256,
    source_hashes,
    stub,
    write_new,
)
from .data import load_record, load_truth

CASES = [
    ("sub-01", "staircase", 30),
    ("sub-01", "triangular-ramp10s", 30),
    ("sub-01", "triangular-ramp5s", 30),
]


def run():
    from adapt_decomp import AdaptationResult
    from adapt_decomp.spikes import rate_of_agreement_paired

    folder = OUTPUT / "L0"
    write_new(
        folder / "protocol.json",
        dict(
            registered_at_utc=now(),
            cases=CASES,
            sources=source_hashes(),
            protected=protected_hashes(),
            reference="Pareto-rerun",
            mode="raw EMG; pinned preprocess_emg_stateful; CPU; no offline Data",
            required_sample_equality=0.999,
            required_roa_difference_pp=0.01,
            development=dict(
                training=["sub-01", "sub-02", "sub-03"],
                validation=["sub-04"],
                training_conditions="all five",
                training_snrs=[30, 25, 20, 15],
                sub05="unopened",
            ),
        ),
    )
    for relative in source_hashes():
        destination = folder / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    results = []
    for subject, condition, snr in CASES:
        name = stub(subject, condition, snr)
        print("identity", name, flush=True)
        tick = time.perf_counter()
        record = load_record(subject, condition, snr, phase="identity")
        adapter = build(record["calibration"], record["config"])
        reference = copy.deepcopy(adapter)
        reference.__class__ = AdaptDecomp
        actual = replay(record["raw"], adapter, progress=True)
        expected = replay(record["raw"], reference, collect_sources=False)
        prediction = folder / f"{name}.npz"
        np.savez_compressed(
            prediction,
            **{k: v for k, v in actual.items() if v is not None},
            reference_spikes=expected["spikes"],
        )
        # Seal both predictions before opening the post-calibration labels.
        write_new(
            folder / f"{name}_sealed.json",
            dict(
                sealed_at_utc=now(),
                prediction_sha256=sha256(prediction),
                raw_sha256=record["raw_sha256"],
                calibration_sha256=record["calibration_sha256"],
            ),
        )
        truth = load_truth(record, phase="identity")
        fs = record["config"].fs
        a, _, _ = rate_of_agreement_paired(truth, actual["spikes"], fs=fs, tol_spike_ms=2)
        b, _, _ = rate_of_agreement_paired(truth, expected["spikes"], fs=fs, tol_spike_ms=2)
        published = AdaptationResult.load(pareto_path(subject, condition, snr))
        item = dict(
            record=name,
            samples=len(truth),
            units=truth.shape[1],
            disagreements=int(np.count_nonzero(actual["spikes"] != expected["spikes"])),
            equality=float(np.mean(actual["spikes"] == expected["spikes"])),
            roa_mean_pct=float(np.mean(a) * 100),
            max_roa_difference_pp=float(np.max(np.abs(a - b)) * 100),
            published_equality=float(np.mean(actual["spikes"] == published.spikes.numpy())),
            published_roa_pct=float(np.mean(published.roa) * 100),
            median_batch_ms=actual["median_batch_ms"],
            p95_batch_ms=actual["p95_batch_ms"],
            elapsed_seconds=time.perf_counter() - tick,
        )
        item["passed"] = item["equality"] >= 0.999 and item["max_roa_difference_pp"] <= 0.01
        write_new(folder / f"{name}_result.json", item)
        print(item, flush=True)
        results.append(item)
        del adapter, reference, record, actual, expected, truth, published
    report = dict(
        completed_at_utc=now(),
        reference="Pareto-rerun",
        records=results,
        passed=len(results) >= 3 and all(r["passed"] for r in results),
    )
    write_new(folder / "result.json", report)
    return report
