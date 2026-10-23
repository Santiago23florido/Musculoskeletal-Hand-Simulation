"""Real active L3 equivalence before searching full-memory hyperparameters."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .backbone import replay
from .common import OUTPUT, now, register, write_new
from .data import load_record
from .experiment import checked
from .memory_experiment import read, seed_selection
from .memory_loop import MemoryConfig, make_memory_adapter
from .observer import load_observer
from .shared_rollout import prepare, shared_replay
from .train_likelihood import load_model


def configurations():
    return [
        MemoryConfig(
            bins=bins,
            h=h,
            threshold=threshold,
            beta_max=beta,
            momentum=momentum,
            calibration_only=False,
        )
        for bins, h, threshold, beta, momentum in (
            (8, 0.03, 0.5, 0.8, 0.8),
            (16, 0.06, 0.65, 0.4, 0.95),
            (32, 0.12, 0.8, 0.2, 0.99),
        )
    ]


def run(folder):
    previous = read(OUTPUT / "L2/retained.json")
    if not previous["passed"]:
        raise ValueError("L2 must pass before any L3 input is opened")
    runs = [
        (seed_selection(previous["selection"], seed), asdict(settings))
        for seed, settings in enumerate(configurations())
    ]
    if not (folder / "protocol.json").exists():
        register(
            folder,
            dict(
                cases=[["sub-01", "triangular-ramp5s", 30]],
                prefix_seconds=30,
                runs=runs,
                labels="never opened",
                observer="frozen C1",
                classifiers="retained L2 round3 training seeds 0/1/2",
                gate="bitwise equality of ordinary and shared trajectories, including memory bins",
                active_check="at least one non-calibration bin inserted, and memory used",
            ),
        )
    checked(folder)
    if (folder / "result.json").exists():
        return read(folder / "result.json")
    record = load_record("sub-01", "triangular-ramp5s", 30, phase="identity")
    record["raw"] = record["raw"][: 30 * record["config"].fs]
    ordinary = [
        make_memory_adapter(
            record,
            load_model(OUTPUT / selection["checkpoint"]),
            load_observer(),
            memory_config=MemoryConfig(**settings),
            **selection["selection"]["parameters"],
        )
        for selection, settings in runs
    ]
    expected = [replay(record["raw"], adapter) for adapter in ordinary]
    master, adapters, observer, extractor = prepare(record, runs)
    actual = shared_replay(
        record["raw"],
        master,
        adapters,
        observer=observer,
        extractor=extractor,
        collect_sources=True,
    )
    rows = []
    for index, (left, right, a, b) in enumerate(
        zip(expected, actual, ordinary, adapters, strict=True)
    ):
        equality = {
            key: bool(np.array_equal(left[key], right[key]))
            for key in ("spikes", "sources", "centroid", "trusted_count")
        }
        equality["state"] = bool(np.array_equal(a.state_trace, b.state_trace))
        equality["memory_trace"] = bool(np.array_equal(a.memory_trace, b.memory_trace))
        for key in ("sep_vectors", "whitening", "spikes_centr", "base_centr", "source_fifo"):
            equality[key] = torch.equal(getattr(a.decomp, key), getattr(b.decomp, key))
        for key in ("filters", "centroids", "quality"):
            equality["memory_" + key] = torch.equal(getattr(a.memory, key), getattr(b.memory, key))
        row = dict(
            run=index,
            exact=equality,
            populated_bins=int((b.memory.quality[1:] > 0).any(dim=1).sum()),
            memory_hit_rate=float(np.asarray(b.memory_trace)[:, 3].mean()),
            passed=all(equality.values()),
        )
        rows.append(row)
        print("L3 identity", row, flush=True)
    active = any(row["populated_bins"] > 0 and row["memory_hit_rate"] > 0 for row in rows)
    result = dict(completed_at_utc=now(), runs=rows, active=active)
    result["passed"] = active and all(row["passed"] for row in rows)
    write_new(folder / "result.json", result)
    if not result["passed"]:
        raise RuntimeError("Full-memory search did not pass its real active identity control")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.folder)
