"""Three real C2 identity controls and an isolated CPU profile of retained L1."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .backbone import replay
from .closed_loop import make_adapter
from .common import OUTPUT, now, register, sha256, stub, write_new
from .data import load_record
from .experiment import checked
from .identity import CASES
from .memory_loop import MemoryConfig, make_memory_adapter
from .models import SpikeLikelihood
from .observer import load_observer
from .train_likelihood import load_model


def run(folder):
    if not (folder / "protocol.json").exists():
        register(
            folder,
            dict(
                cases=CASES,
                gate="all three zero-memory controls exactly equal sealed L0 spikes",
                memory="calibration-only, beta=0; trained blind observer; zero C4 head",
                profile="retained L1 selected head, sub-01 ramp5s 30dB; no labels opened",
                timing="one record at a time, native Windows CPU, four Torch threads",
            ),
        )
    checked(folder)
    observer = load_observer()
    rows = []
    for case in CASES:
        name = stub(*case)
        path = folder / f"{name}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
            continue
        record = load_record(*case, phase="identity")
        adapter = make_memory_adapter(
            record, SpikeLikelihood(), observer, memory_config=MemoryConfig()
        )
        print("identity", name, "filter dimensions", adapter.decomp.whitening.shape, flush=True)
        result = replay(record["raw"], adapter, collect_sources=False)
        with np.load(OUTPUT / "L0" / f"{name}.npz") as archive:
            reference = archive["spikes"]
        disagreed = int(np.count_nonzero(result["spikes"] != reference))
        output = folder / f"{name}.npz"
        np.savez_compressed(output, spikes=result["spikes"], batch_ms=result["batch_ms"])
        row = dict(
            record=name,
            disagreements=disagreed,
            passed=disagreed == 0,
            median_batch_ms=result["median_batch_ms"],
            p95_batch_ms=result["p95_batch_ms"],
            sha256=sha256(output),
        )
        write_new(path, row)
        rows.append(row)
        print(row, flush=True)
    if not (folder / "l1_isolated_cpu.json").exists():
        selection = json.loads((OUTPUT / "L1/all_feedback/round3/selected.json").read_text())
        checkpoint = OUTPUT / selection["checkpoint"]
        if sha256(checkpoint) != selection["checkpoint_sha256"]:
            raise ValueError("L1 checkpoint changed")
        record = load_record("sub-01", "triangular-ramp5s", 30, phase="identity")
        adapter = make_adapter(
            record, load_model(checkpoint), observer, **selection["selection"]["parameters"]
        )
        prediction = replay(record["raw"], adapter, collect_sources=False)
        output = folder / "l1_isolated_cpu.npz"
        np.savez_compressed(output, spikes=prediction["spikes"], batch_ms=prediction["batch_ms"])
        write_new(
            folder / "l1_isolated_cpu.json",
            dict(
                completed_at_utc=now(),
                record=stub("sub-01", "triangular-ramp5s", 30),
                checkpoint_sha256=sha256(checkpoint),
                prediction_sha256=sha256(output),
                median_batch_ms=prediction["median_batch_ms"],
                p95_batch_ms=prediction["p95_batch_ms"],
                timing_scope="single record; CPU four Torch threads; full raw-EMG L1 loop",
                labels_opened=False,
                passed_22_ms=prediction["median_batch_ms"] <= 22,
            ),
        )
    if not (folder / "result.json").exists():
        write_new(
            folder / "result.json",
            dict(completed_at_utc=now(), records=rows, passed=all(row["passed"] for row in rows)),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.folder)
