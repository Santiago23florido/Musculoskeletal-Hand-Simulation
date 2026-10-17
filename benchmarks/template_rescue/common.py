"""Separate artifacts and immutable dependencies for the rescue experiment."""

import os
from pathlib import Path

from benchmarks.pareto_residual.common import SUBJECTS, TRAIN_CONDITIONS, split
from benchmarks.stage_a.common import (
    CONDITIONS,
    DATA,
    ROOT,
    reference,
    sha256,
    stub,
    write_json,
)

OUTPUT = Path(os.environ.get("TEMPLATE_RESCUE_OUTPUT", ROOT / "outputs/template_rescue")).resolve()
PREVIOUS = ROOT / "outputs/pareto_residual"


def source_hashes():
    files = list(Path(__file__).parent.glob("*.py"))
    for folder in ("pareto_residual", "stage_a"):
        files.extend((ROOT / "benchmarks" / folder).glob("*.py"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


__all__ = [
    "OUTPUT",
    "PREVIOUS",
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
]
