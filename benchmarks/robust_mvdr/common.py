"""Isolated outputs and a pre-registered specification for the robust decoding test."""

import os
from pathlib import Path

from benchmarks.stage_b.common import (
    CONDITIONS,
    DATA,
    ROOT,
    SUBJECTS,
    pareto_path,
    reference,
    sha256,
    stub,
    write_json,
)

OUTPUT = Path(os.environ.get("ROBUST_MVDR_OUTPUT", ROOT / "outputs/robust_mvdr")).resolve()

# Each method: (state mode, covariance strategy). Nothing below is fitted to FDSI
# labels; values were fixed from the synthetic study documented in README.md.
#   rho: fraction of the unit's own modelled lag covariance removed from R
#   loading: diagonal loading in units of the mean eigenvalue of R
#   alpha: weight of the original Stage B model covariance (0 disables it)
METHODS = {
    "static_subtract": dict(mode="static", rho=0.5, loading=0.03, alpha=0.0),
    "static_subtract_strong": dict(mode="static", rho=1.0, loading=0.1, alpha=0.0),
    "static_loaded": dict(mode="static", rho=0.0, loading=0.1, alpha=0.0),
    "static_fifo_mvdr": dict(mode="static", rho=0.0, loading=0.001, alpha=0.0),
    "moca_k_subtract": dict(mode="k", rho=0.5, loading=0.03, alpha=0.0),
    "moca_b_subtract": dict(mode="b", rho=0.5, loading=0.03, alpha=0.0),
}
# Without Stage B's frozen weights, run the static methods only. The choice is part of
# the registered specification, so keep the same value for every command of a run.
if os.environ.get("ROBUST_MVDR_STATIC_ONLY") == "1":
    METHODS = {k: v for k, v in METHODS.items() if v["mode"] == "static"}
PRIMARY = "static_subtract"
LAG_KNOTS = 33
FIFO_BATCHES = 10
SUBTRACTION_FLOOR = 0.05
ANGLE_REST_DEG = 1.0
SPEED_REST_DEG_S = 0.5  # FDSI's slowest ramps move at exactly 1 deg/s
REFERENCE_METHODS = ("moca_k", "moca_b", "static_analytic")


def source_hashes():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [
        ROOT / "benchmarks/stage_b/inference.py",
        ROOT / "benchmarks/stage_b/data.py",
        ROOT / "benchmarks/stage_b/experiment.py",
        ROOT / "benchmarks/pareto_residual/detector.py",
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


__all__ = [
    "ANGLE_REST_DEG",
    "CONDITIONS",
    "DATA",
    "FIFO_BATCHES",
    "LAG_KNOTS",
    "METHODS",
    "OUTPUT",
    "PRIMARY",
    "REFERENCE_METHODS",
    "ROOT",
    "SPEED_REST_DEG_S",
    "SUBJECTS",
    "SUBTRACTION_FLOOR",
    "pareto_path",
    "reference",
    "sha256",
    "source_hashes",
    "stub",
    "write_json",
]
