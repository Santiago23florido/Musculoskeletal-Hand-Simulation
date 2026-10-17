"""Isolated Stage B artifacts and subject splits."""

import os
from pathlib import Path

from benchmarks.pareto_residual.common import SUBJECTS, TRAIN_CONDITIONS, split
from benchmarks.stage_a.common import (
    CONDITIONS,
    DATA,
    ROOT,
    calibration_path,
    pareto_path,
    reference,
    sha256,
    stub,
    write_json,
)

OUTPUT = Path(os.environ.get("STAGE_B_OUTPUT", ROOT / "outputs/stage_b_main")).resolve()
RADIUS = 32
DIMENSION = 512


def source_hashes():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [
        ROOT / "benchmarks/stage_a/common.py",
        ROOT / "benchmarks/stage_a/data.py",
        ROOT / "benchmarks/stage_a/evaluate.py",
        ROOT / "benchmarks/pareto_residual/detector.py",
        ROOT / "benchmarks/pareto_residual/common.py",
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


__all__ = [
    "OUTPUT",
    "RADIUS",
    "DIMENSION",
    "SUBJECTS",
    "TRAIN_CONDITIONS",
    "split",
    "CONDITIONS",
    "DATA",
    "ROOT",
    "reference",
    "sha256",
    "stub",
    "write_json",
    "source_hashes",
    "calibration_path",
    "pareto_path",
]
