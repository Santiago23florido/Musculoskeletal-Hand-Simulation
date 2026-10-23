"""Frozen weights stay verifiable across the publication-only guide metadata change."""

import json

import pytest
import torch

from benchmarks.stage_b.model import BlindState, MorphNet

from . import frozen
from .common import sha256, write_json


@pytest.fixture
def archive(tmp_path):
    root, folder = tmp_path / "repo", tmp_path / "stage_b"
    experiment = f"def register():\n    return dict(guide_sha256={frozen.GUIDE_HASH})\n"
    sources = {
        frozen.EXPERIMENT: experiment,
        "benchmarks/stage_b/model.py": "# Immutable model source fixture.\n",
    }
    hashes = {}
    for name, content in sources.items():
        for base in (root, folder / "source"):
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        # Exercise Windows manifest paths even when the test runs on Linux.
        hashes[name.replace("/", "\\")] = sha256(root / name)
    protocol = dict(spec=dict(seeds=[0, 1]), folds={"sub-02": {}}, source_sha256=hashes)
    write_json(folder / "protocol.json", protocol)
    protocol_hash = sha256(folder / "protocol.json")
    models = {"sub-02": {}}
    for seed in (0, 1):
        entry = dict(protocol_sha256=protocol_hash)
        for kind, model in (
            ("morph", MorphNet(channels=2)),
            ("observer", BlindState(hidden=4)),
        ):
            path = folder / "folds/sub-02" / f"seed{seed}" / f"{kind}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(spec=model.spec, model=model.state_dict()), path)
            entry[f"{kind}_sha256"] = sha256(path)
        models["sub-02"][str(seed)] = entry
    write_json(folder / "locked.json", dict(protocol_sha256=protocol_hash, models=models))
    return root, folder


def optional_metadata(root):
    path = root / frozen.EXPERIMENT
    path.write_text(
        path.read_text(encoding="utf-8").replace(frozen.GUIDE_HASH, frozen.OPTIONAL_GUIDE_HASH),
        encoding="utf-8",
    )


def test_optional_metadata_preserves_exact_cpu_checkpoint_parameters(archive):
    root, folder = archive
    optional_metadata(root)
    provenance = frozen.validate(folder, root=root)
    assert set(provenance["compatibility_changes"]) == {frozen.EXPERIMENT}
    assert len(provenance["checkpoint_sha256"]) == 4
    models = frozen.load_models("sub-02", folder, expected=provenance, root=root)
    for seed, pair in enumerate(models):
        for kind, model in zip(("morph", "observer"), pair, strict=True):
            saved = torch.load(
                folder / "folds/sub-02" / f"seed{seed}" / f"{kind}.pt",
                weights_only=True,
                map_location="cpu",
            )
            assert not model.training
            for name, value in model.state_dict().items():
                assert value.device.type == "cpu"
                torch.testing.assert_close(value, saved["model"][name], rtol=0, atol=0)


def test_metadata_exception_rejects_any_other_code_change(archive):
    root, folder = archive
    optional_metadata(root)
    path = root / frozen.EXPERIMENT
    path.write_text(
        path.read_text() + "\ndef numerical_change():\n    return 1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="numerical source changed"):
        frozen.validate(folder, root=root)


def test_modified_frozen_source_is_rejected(archive):
    root, folder = archive
    path = folder / "source" / frozen.EXPERIMENT
    path.write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source snapshot changed"):
        frozen.validate(folder, root=root)


def test_modified_checkpoint_is_rejected(archive):
    root, folder = archive
    path = folder / "folds/sub-02/seed0/morph.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        frozen.validate(folder, root=root)


def test_modified_protocol_is_rejected(archive):
    root, folder = archive
    path = folder / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["spec"]["seeds"] = [0]
    write_json(path, protocol)
    with pytest.raises(ValueError, match="protocol changed"):
        frozen.validate(folder, root=root)


def test_even_allowed_metadata_changes_require_a_new_registration(archive):
    root, folder = archive
    original = frozen.validate(folder, root=root)
    optional_metadata(root)
    with pytest.raises(ValueError, match="differ from the registered ones"):
        frozen.load_models("sub-02", folder, expected=original, root=root)
