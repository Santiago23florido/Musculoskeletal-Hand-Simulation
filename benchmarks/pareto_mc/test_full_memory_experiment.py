"""L3 uses frozen L2 seeds and cannot start before its preceding gate passes."""

import pytest

from . import full_memory_experiment as experiment


def test_l3_protocol_uses_l2_classifiers_and_excludes_the_external_subject(monkeypatch):
    previous = dict(
        passed=True,
        actual=dict(global_roa_pct=91.65800814434645),
        selection=dict(checkpoint="L2/standard/round3/seed2/model.pt"),
    )

    def seed_selection(selection, seed):
        assert selection["checkpoint"].startswith("L2/")
        return dict(checkpoint_sha256=f"L2-seed-{seed}")

    monkeypatch.setattr(experiment, "seed_selection", seed_selection)
    monkeypatch.setattr(experiment, "sha256", lambda path: "input-digest")
    monkeypatch.setattr(experiment, "config_hash", lambda case: "config-digest")
    protocol = experiment.specification(previous, conservative=False, workers=4)
    assert protocol["gate_global_min"] == pytest.approx(92.65800814434645)
    assert protocol["classifier_sha256"] == {seed: f"L2-seed-{seed}" for seed in (0, 1, 2)}
    assert len(protocol["cases"]) == 80
    assert all(subject != "sub-05" for subject, _, _ in protocol["cases"])


def test_failed_l2_blocks_l3_before_registration_or_execution(monkeypatch):
    monkeypatch.setattr(experiment, "read", lambda path: dict(passed=False, active=True))

    def forbidden(*args, **kwargs):
        pytest.fail("A blocked L3 must not register or execute any trajectory")

    monkeypatch.setattr(experiment, "register", forbidden)
    monkeypatch.setattr(experiment, "run_variant", forbidden)
    with pytest.raises(ValueError, match="L2"):
        experiment.run(4)
