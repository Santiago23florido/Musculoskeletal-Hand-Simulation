"""Pinned inputs and provenance shared by Stage A commands."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "external/farina/adapt_decomp"
DATA = ROOT / "data/benchmarks/farina/fdsi_benchmark"
OUTPUT = (
    Path(os.environ.get("STAGE_A_OUTPUT", str(ROOT / "outputs/stage_a"))).expanduser().resolve()
)
COMMIT = "a8f53736e297962670276bc123c137050b7056b6"
CONDITIONS = [
    "staircase",
    "triangular-ramp40s",
    "triangular-ramp20s",
    "triangular-ramp10s",
    "triangular-ramp5s",
]
POOL = ["triangular-ramp40s", "triangular-ramp10s", "triangular-ramp5s"]


def reference():
    revision = subprocess.check_output(
        ["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(REFERENCE), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if revision != COMMIT or dirty:
        raise ValueError("The clean pinned reference checkout is required")
    if str(REFERENCE / "src") not in sys.path:
        sys.path.insert(0, str(REFERENCE / "src"))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stub(subject, condition, snr):
    return f"{subject}_FDSI_{condition}_snr{snr}dB"


def calibration_path(subject, condition, snr):
    return DATA / "outputs/calibration" / subject / f"{stub(subject, condition, snr)}_cbss.pkl"


def pareto_path(subject, condition, snr):
    return (
        DATA
        / "outputs/adaptation"
        / subject
        / "lr_fixed/tpe_pareto"
        / f"{stub(subject, condition, snr)}_adapt_lr_fixed_tpe_pareto.pkl"
    )
