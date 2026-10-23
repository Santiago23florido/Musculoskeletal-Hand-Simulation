"""Real-record identity of C4 plus the unused fixed-mode KL-stat initialization shortcut."""

import numpy as np
import torch

from .backbone import replay
from .closed_loop import make_adapter
from .common import OUTPUT, now, register, sha256, stub, write_new
from .data import load_record
from .identity import CASES
from .models import SpikeLikelihood
from .observer import load_observer


def run():
    folder = OUTPUT / "C4_identity"
    register(
        folder,
        dict(
            cases=CASES,
            classifier="zero-initialized 22->8->1",
            observer="frozen C1 seed0",
            whitening_statistics="one calibration window, irrelevant to fixed learning rates",
            reference="sealed L0 Pareto-rerun, also exactly published Pareto",
            gate="zero spike disagreements in all three records; no new GT scoring needed",
        ),
    )
    rows = []
    for case in CASES:
        record = load_record(*case, phase="identity")
        name = stub(*case)
        observer = load_observer()
        adapter = make_adapter(record, SpikeLikelihood(), observer)
        result = replay(record["raw"], adapter, collect_sources=False)
        with np.load(OUTPUT / "L0" / f"{name}.npz") as archive:
            reference = archive["spikes"]
        disagreements = int(np.count_nonzero(result["spikes"] != reference))
        prediction = folder / f"{name}.npz"
        np.savez_compressed(prediction, spikes=result["spikes"], batch_ms=result["batch_ms"])
        row = dict(
            record=name,
            disagreements=disagreements,
            equality=float(np.mean(result["spikes"] == reference)),
            median_batch_ms=result["median_batch_ms"],
            p95_batch_ms=result["p95_batch_ms"],
            prediction_sha256=sha256(prediction),
            passed=disagreements == 0,
        )
        write_new(folder / f"{name}.json", row)
        print(row, flush=True)
        rows.append(row)
    write_new(
        folder / "result.json",
        dict(completed_at_utc=now(), records=rows, passed=all(row["passed"] for row in rows)),
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    run()
