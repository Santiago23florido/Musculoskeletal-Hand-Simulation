"""P3 post hoc latency with ONE Torch thread (not registered; reported as post hoc).

Why (11 November 2026): on this laptop (Core Ultra 7 155H, torch's oneMKL 2025.1) the
four-thread LU used by torch.linalg.slogdet on Pareto's 1000-dimensional whitening is
pathological: 1000x1000 slogdet takes ~12 ms with one thread and ~500 ms with four, and
during Pareto's calibration MKL prints 'Parameter 6 was incorrect on entry to SLASWP'
(0 messages with one thread, ~816 with four). The registered latency (four threads,
closed_context.latency) is run unchanged; this script repeats the same records and
systems with one thread so the cost of the method can be separated from that defect.
Note: the calibration statistic differs with the thread count, so these decisions are
not bit-identical to the four-thread runs; only the timing is of interest here.
"""

import os
from collections import defaultdict

import numpy as np
import torch

from benchmarks.pareto_mc import closed_context as cc
from benchmarks.pareto_mc.data import load_record

PATH = cc.OUTPUT / "P3_posthoc" / "latency_threads1.json"


def main():
    if PATH.exists():
        return cc.read(PATH)
    torch.set_num_threads(1)
    selected = cc.read(cc.FOLDER / "round3/selected.json")["head"]
    retained = cc.read(cc.OUTPUT / "L2/retained.json")["selection"]
    l2 = dict(
        kind="l2",
        checkpoint=retained["checkpoint"],
        checkpoint_sha256=retained["checkpoint_sha256"],
        parameters=retained["selection"]["parameters"],
    )
    systems = dict(pareto=dict(kind="pareto"), l2=l2, context=selected)
    cases = [("sub-04", condition, 20) for condition in cc.CONDITIONS]
    timings = defaultdict(list)
    for case in cases:
        record = load_record(*case, phase="validation")
        for name, spec in systems.items():
            prediction, _ = cc.run_system(record, spec)
            timings[name].append(prediction["batch_ms"])
            print("latency1", name, cc.stub(*case), round(float(np.median(timings[name][-1])), 2),
                  flush=True)
    summary = {}
    for name, parts in timings.items():
        values = np.concatenate(parts)
        summary[name] = dict(
            batches=len(values),
            median_ms=float(np.median(values)),
            p95_ms=float(np.percentile(values, 95)),
            mean_ms=float(np.mean(values)),
            max_ms=float(np.max(values)),
        )
    result = dict(
        measured_at_utc=cc.now(),
        post_hoc=True,
        scope="as closed_context.latency but ONE Torch thread; native Windows CPU, one "
        "record at a time; raw filtering through closed-loop update; 204-sample batches",
        reason="four-thread oneMKL LU pathology on this laptop (see module docstring)",
        records=[cc.stub(*case) for case in cases],
        target_ms=22,
        systems=summary,
        cpu=os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
    )
    PATH.parent.mkdir(parents=True, exist_ok=True)
    cc.write_new(PATH, result)
    return result


if __name__ == "__main__":
    print("P3 latency threads1", main()["systems"], cc.now(), flush=True)
