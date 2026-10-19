import os
from pathlib import Path

from benchmarks.stage_b.common import (
    CONDITIONS,
    DATA,
    ROOT,
    SUBJECTS,
    TRAIN_CONDITIONS,
    pareto_path,
    reference,
    sha256,
    split,
    stub,
    write_json,
)
from benchmarks.stage_b.common import (
    source_hashes as baseline_sources,
)

OUTPUT = Path(
    os.environ.get("SPIKE_LIKELIHOOD_OUTPUT", ROOT / "outputs/spike_likelihood")
).resolve()


def source_hashes():
    result = baseline_sources()
    paths = list(Path(__file__).parent.glob("*.py"))
    paths.append(ROOT / "benchmarks/pareto_residual/data.py")
    paths.append(ROOT / "benchmarks/stage_b_separation/states.py")
    paths.append(ROOT / "benchmarks/stage_b_separation/common.py")
    result.update({str(p.relative_to(ROOT)): sha256(p) for p in paths})
    return result


__all__ = [
    "OUTPUT",
    "ROOT",
    "DATA",
    "SUBJECTS",
    "CONDITIONS",
    "TRAIN_CONDITIONS",
    "split",
    "stub",
    "write_json",
    "sha256",
    "source_hashes",
    "reference",
    "pareto_path",
]
