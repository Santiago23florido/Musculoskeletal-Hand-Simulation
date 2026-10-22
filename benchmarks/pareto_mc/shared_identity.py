"""Real EMG equivalence of ordinary and computation-sharing search trajectories."""

import numpy as np
import torch

from .backbone import replay
from .common import OUTPUT, now, register, write_new
from .data import load_record
from .experiment import checked
from .memory_experiment import l1_selection
from .memory_loop import MemoryConfig, make_memory_adapter
from .observer import load_observer
from .shared_rollout import prepare, shared_replay
from .train_likelihood import load_model


def run():
    folder = OUTPUT / "C2_shared_identity_v1"
    runs = [
        (l1_selection(0), MemoryConfig(bins=8, h=0.03, beta_max=0.8).__dict__),
        (l1_selection(1), MemoryConfig(bins=16, h=0.06, beta_max=0.2).__dict__),
        (l1_selection(2), MemoryConfig(bins=32, h=0.12, beta_max=0.4).__dict__),
    ]
    if not (folder / "protocol.json").exists():
        register(
            folder,
            dict(
                record=["sub-01", "triangular-ramp5s", 30],
                prefix_seconds=30,
                runs=runs,
                labels="never opened",
                gate="bitwise equality of spikes, sources, C1 states, memory, centroids and SV",
            ),
        )
    checked(folder)
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
    for index, (a, b, left, right) in enumerate(
        zip(expected, actual, ordinary, adapters, strict=True)
    ):
        keys = ("spikes", "sources", "centroid", "trusted_count")
        equality = {key: bool(np.array_equal(a[key], b[key])) for key in keys}
        equality["state"] = bool(np.array_equal(left.state_trace, right.state_trace))
        equality["memory"] = bool(np.array_equal(left.memory_trace, right.memory_trace))
        for key in ("sep_vectors", "whitening", "spikes_centr", "base_centr", "source_fifo"):
            equality[key] = torch.equal(getattr(left.decomp, key), getattr(right.decomp, key))
        row = dict(run=index, exact=equality, passed=all(equality.values()))
        rows.append(row)
        print(row, flush=True)
    hits = float(np.asarray(adapters[0].memory_trace)[:, 3].mean())
    write_new(
        folder / "result.json",
        dict(
            completed_at_utc=now(),
            runs=rows,
            memory_hit_rate=hits,
            passed=all(r["passed"] for r in rows),
        ),
    )
    if not all(row["passed"] for row in rows) or hits == 0:
        raise RuntimeError("Shared search failed its active-memory equivalence check")


if __name__ == "__main__":
    torch.set_num_threads(4)
    run()
