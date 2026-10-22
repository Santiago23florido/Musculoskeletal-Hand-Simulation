"""Exercise interruption recovery and real Optuna median selection without EMG IO."""

from collections import Counter

import pytest

from . import memory_search
from .common import write_new


def test_interrupted_search_resumes_same_45_trials_and_selects_median_seed(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    write_new(
        tmp_path / "C2_shared_identity_v1/result.json", dict(passed=True, memory_hit_rate=0.1)
    )
    write_new(tmp_path / "L0/validation/summary.json", dict(precision_pct=96))
    root = tmp_path / "L2/standard"
    root.mkdir(parents=True)
    monkeypatch.setattr(memory_search, "OUTPUT", tmp_path)
    monkeypatch.setattr(memory_search, "checked", lambda folder: None)
    monkeypatch.setattr(memory_search, "l1_selection", lambda seed: dict(seed=seed))
    calls = []

    def interrupted(pool, runs):
        calls.append(runs)
        raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(memory_search, "group_rollout", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        memory_search.search(root, None, {}, calibration_only=True, conservative=False)
    interrupted_parameters = [settings for _, _, settings in calls[0]]
    monkeypatch.setattr(memory_search, "group_rollout", lambda pool, runs: calls.append(runs))

    def score(folder):
        trial = int(folder.parent.name.removeprefix("trial"))
        seed = int(folder.name.removeprefix("seed"))
        return dict(
            global_roa_pct=90 + trial / 100, ramp5s_roa_pct=84 + seed / 100, precision_pct=96
        )

    monkeypatch.setattr(memory_search, "score_search", score)
    result = memory_search.search(root, None, {}, calibration_only=True, conservative=False)
    assert [settings for _, _, settings in calls[1]] == interrupted_parameters
    trials = memory_search.read(root / "trials.json")
    assert len(trials) == 45 and all(t["state"] == "COMPLETE" for t in trials)
    assert Counter(t["params"]["structure"] for t in trials) == dict.fromkeys(range(9), 5)
    assert result["winner"]["trial"] == 44 and result["winner"]["seed"] == 1


def test_missing_or_changed_prediction_seal_prevents_label_access(tmp_path, monkeypatch):
    touched = []
    monkeypatch.setattr(memory_search, "descriptor", lambda *args, **kwargs: touched.append(True))
    write_new(tmp_path / "sealed.json", dict(files={"predictions/changed.npz": "wrong"}))
    (tmp_path / "predictions").mkdir()
    (tmp_path / "predictions/changed.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="seal"):
        memory_search.score_search(tmp_path)
    assert not touched
