import os
from pathlib import Path

from benchmarks.stage_a.common import CONDITIONS, DATA, ROOT, reference, sha256, stub, write_json

OUTPUT = Path(os.environ.get("PARETO_RESIDUAL_OUTPUT", ROOT / "outputs/pareto_residual")).resolve()
TRAIN_CONDITIONS = ["staircase", "triangular-ramp40s", "triangular-ramp10s"]
SUBJECTS = [f"sub-0{i}" for i in range(1, 6)]


def split(test):
    """Keep the original Pareto tuning subject in training, never in the primary test."""
    if test not in SUBJECTS[1:]:
        raise ValueError("Primary outer tests are sub-02 through sub-05")
    index = SUBJECTS[1:].index(test)
    validation = SUBJECTS[1:][(index + 1) % 4]
    training = [s for s in SUBJECTS if s not in (test, validation)]
    return dict(training=training, validation=[validation], test=[test])


def source_hashes():
    return {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}


__all__ = [
    "CONDITIONS",
    "DATA",
    "ROOT",
    "reference",
    "sha256",
    "stub",
    "write_json",
    "OUTPUT",
    "TRAIN_CONDITIONS",
    "SUBJECTS",
    "split",
    "source_hashes",
]
