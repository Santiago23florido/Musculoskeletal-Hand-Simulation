"""Read frozen Stage B checkpoints without changing their historical registration."""

import ast
import json
from pathlib import Path

import torch

from .common import ROOT, sha256

EXPERIMENT = "benchmarks/stage_b/experiment.py"
GUIDE_HASH = 'sha256(ROOT / "MUAP_dynamic_DL_research_report.pdf")'
OPTIONAL_GUIDE_HASH = (
    f'{GUIDE_HASH} if (ROOT / "MUAP_dynamic_DL_research_report.pdf").is_file() else None'
)


def _metadata_only_change(frozen, current):
    """Allow exactly the published optional-guide expression, nowhere else."""
    trees = [ast.parse(frozen), ast.parse(current)]
    expressions = [GUIDE_HASH, OPTIONAL_GUIDE_HASH]
    for tree, expression in zip(trees, expressions, strict=True):
        functions = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "register"
        ]
        if len(functions) != 1:
            return False
        keywords = [
            n
            for n in ast.walk(functions[0])
            if isinstance(n, ast.keyword) and n.arg == "guide_sha256"
        ]
        expected = ast.parse(expression, mode="eval").body
        if len(keywords) != 1 or ast.dump(keywords[0].value) != ast.dump(expected):
            return False
        keywords[0].value = ast.Constant(value=None)
    return ast.dump(trees[0]) == ast.dump(trees[1])


def _relative(name):
    path = Path(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or path.parts[0] != "benchmarks":
        raise ValueError(f"Invalid frozen source path: {name}")
    return path


def validate(folder, *, root=ROOT):
    """Verify the original protocol, archived sources, current code and every weight."""
    folder, root = Path(folder), Path(root)
    protocol_path = folder / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    locked = json.loads((folder / "locked.json").read_text(encoding="utf-8"))
    protocol_hash = sha256(protocol_path)
    if locked["protocol_sha256"] != protocol_hash:
        raise ValueError("Frozen Stage B protocol changed")
    original, current, compatibility = {}, {}, {}
    for name, expected in protocol["source_sha256"].items():
        relative = _relative(name)
        key = relative.as_posix()
        archive = folder / "source" / relative
        actual = root / relative
        if not archive.is_file() or sha256(archive) != expected:
            raise ValueError(f"Frozen Stage B source snapshot changed: {key}")
        original[key] = expected
        current[key] = sha256(actual)
        if current[key] != expected:
            if key != EXPERIMENT or not _metadata_only_change(
                archive.read_text(encoding="utf-8"), actual.read_text(encoding="utf-8")
            ):
                raise ValueError(f"Stage B numerical source changed: {key}")
            compatibility[key] = "Optional guide metadata only; all other AST nodes identical"
    if "benchmarks/stage_b/model.py" not in original or EXPERIMENT not in original:
        raise ValueError("Stage B model and experiment sources must be registered")
    if set(locked["models"]) != set(protocol["folds"]):
        raise ValueError("Frozen Stage B subject coverage changed")
    seeds = protocol["spec"]["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid frozen Stage B seeds")
    checkpoints = {}
    for subject, models in locked["models"].items():
        if set(models) != {str(seed) for seed in seeds}:
            raise ValueError("Frozen Stage B seed coverage changed")
        for seed in seeds:
            entry = models[str(seed)]
            if entry["protocol_sha256"] != protocol_hash:
                raise ValueError("Stage B checkpoint refers to another protocol")
            for kind in ("morph", "observer"):
                relative = Path("folds") / subject / f"seed{seed}" / f"{kind}.pt"
                path = folder / relative
                if not path.is_file() or sha256(path) != entry[f"{kind}_sha256"]:
                    raise ValueError(f"Frozen Stage B checkpoint changed: {relative.as_posix()}")
                checkpoints[relative.as_posix()] = entry[f"{kind}_sha256"]
    return dict(
        protocol_sha256=protocol_hash,
        locked_sha256=sha256(folder / "locked.json"),
        registered_source_sha256=original,
        current_source_sha256=current,
        compatibility_changes=compatibility,
        checkpoint_sha256=checkpoints,
    )


def load_models(subject, folder, *, expected, root=ROOT):
    """Load the verified, unchanged original models strictly on CPU, without fitting."""
    folder = Path(folder)
    if validate(folder, root=root) != expected:
        raise ValueError("Stage B frozen models differ from the registered ones")
    from benchmarks.stage_b.model import BlindState, MorphNet

    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    if subject not in protocol["folds"]:
        raise ValueError("Subject has no frozen Stage B models")
    result = []
    for seed in protocol["spec"]["seeds"]:
        models = []
        for kind, factory in (("morph", MorphNet), ("observer", BlindState)):
            path = folder / "folds" / subject / f"seed{seed}" / f"{kind}.pt"
            saved = torch.load(path, weights_only=True, map_location="cpu")
            model = factory(**saved["spec"])
            model.load_state_dict(saved["model"], strict=True)
            model.eval()
            models.append(model)
        result.append(tuple(models))
    return result
