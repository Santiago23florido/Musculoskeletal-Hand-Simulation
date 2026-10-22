"""Separate, append-only experiment outputs; historical experiments are read-only."""

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.stage_a.common import (
    COMMIT,
    CONDITIONS,
    DATA,
    REFERENCE,
    ROOT,
    calibration_path,
    pareto_path,
    reference,
    sha256,
    stub,
)

OUTPUT = Path(os.environ.get("PARETO_MC_OUTPUT", ROOT / "outputs/pareto_mc")).resolve()
TRAIN = ("sub-01", "sub-02", "sub-03")
VALIDATION = ("sub-04",)
SNRS = (30, 25, 20, 15)


def now():
    return datetime.now(UTC).isoformat()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes():
    paths = list((ROOT / "benchmarks/pareto_mc").glob("*.py"))
    paths += list((REFERENCE / "src/adapt_decomp").rglob("*.py"))
    paths += [ROOT / "benchmarks/robust_mvdr/phases.py", ROOT / "benchmarks/fifo_audit/harness.py"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def protected_hashes():
    tracked = subprocess.check_output(["git", "ls-files", "benchmarks", "docs"], text=True)
    paths = [ROOT / p for p in tracked.splitlines() if "pareto_mc" not in p]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths if p.is_file()}


def records(subjects):
    return [(s, c, n) for s in subjects for c in CONDITIONS for n in SNRS]


def register(folder, specification):
    folder = Path(folder)
    sources = source_hashes()
    write_new(
        folder / "protocol.json",
        dict(
            registered_at_utc=now(),
            spec=specification,
            sources=sources,
            protected=protected_hashes(),
        ),
    )
    for relative in sources:
        destination = folder / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)


def guard(subject, phase):
    allowed = {"training": TRAIN, "validation": VALIDATION, "identity": TRAIN}
    if phase not in allowed or subject not in allowed[phase]:
        raise ValueError(f"Refused {phase} access to {subject}; sub-05 remains sealed")


__all__ = [
    "COMMIT",
    "CONDITIONS",
    "DATA",
    "REFERENCE",
    "ROOT",
    "OUTPUT",
    "TRAIN",
    "VALIDATION",
    "SNRS",
    "calibration_path",
    "pareto_path",
    "reference",
    "sha256",
    "stub",
    "now",
    "write_new",
    "source_hashes",
    "protected_hashes",
    "records",
    "register",
    "guard",
]
