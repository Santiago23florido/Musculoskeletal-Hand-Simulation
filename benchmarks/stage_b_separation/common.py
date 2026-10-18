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
    source_hashes as original_sources,
)

OUTPUT = Path(
    os.environ.get("STAGE_B_SEPARATION_OUTPUT", ROOT / "outputs/stage_b_separation")
).resolve()


def source_hashes():
    result = original_sources()
    result.update({str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob("*.py")})
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
