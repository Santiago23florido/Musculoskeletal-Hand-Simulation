"""Keep all searched structures causal and resume only unchanged predictions."""

import numpy as np
import pytest

from . import memory_experiment as experiment
from .common import write_new
from .memory_loop import MemoryConfig


def test_every_declared_structure_has_valid_bounds_and_inactive_momentum_is_not_searched():
    class Trial:
        def __init__(self, structure):
            self.structure = structure
            self.asked = []

        def suggest_categorical(self, name, values):
            assert self.structure in values
            return self.structure

        def suggest_float(self, name, low, high):
            self.asked.append(name)
            return (low + high) / 2

    for index, (bins, h) in enumerate(experiment.STRUCTURES):
        trial = Trial(index)
        config = MemoryConfig(
            **experiment.parameters(trial, calibration_only=True, conservative=False)
        )
        assert config.bins == bins and config.h == h and config.momentum == 0.95
        assert "momentum" not in trial.asked
        experiment.parameters(trial, calibration_only=False, conservative=False)
        assert "momentum" in trial.asked


def test_all_predictions_are_finished_and_sealed_before_scoring(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from .common import sha256, stub

    cases = [("sub-04", "triangular-ramp5s", 30), ("sub-04", "triangular-ramp5s", 20)]
    completed = []

    def predict(arguments):
        folder, case, *_ = arguments
        assert not (folder / "sealed.json").exists()
        destination = folder / "predictions" / f"{stub(*case)}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(destination, spikes=np.zeros((8, 2), dtype=np.uint8))
        write_new(
            folder / "metadata" / f"{stub(*case)}.json",
            dict(files={destination.relative_to(folder).as_posix(): sha256(destination)}),
        )
        completed.append(case)
        return stub(*case)

    monkeypatch.setattr(experiment, "rollout_task", predict)
    with ThreadPoolExecutor(2) as pool:
        experiment.rollout(pool, tmp_path, cases, {}, {}, detailed=False)
    seal = experiment.read(tmp_path / "sealed.json")
    assert len(completed) == 2 and len(seal["files"]) == 2
    assert seal["no_labels_during_rollout"]


def test_a_changed_completed_rollout_is_rejected_before_loading_any_data(tmp_path):
    from .common import stub

    case = ("sub-04", "triangular-ramp5s", 30)
    write_new(
        tmp_path / "metadata" / f"{stub(*case)}.json",
        dict(settings=dict(beta_max=0.1), head={}),
    )
    with pytest.raises(ValueError, match="different memory"):
        experiment.rollout_task((tmp_path, case, {}, dict(beta_max=0.8), False))


def test_precision_penalty_uses_percentage_points():
    good = dict(global_roa_pct=91, ramp5s_roa_pct=84, precision_pct=96)
    poor = dict(good, precision_pct=93)
    assert experiment.objective_value(good, 96) == 87.5
    assert experiment.objective_value(poor, 96) == 77.5
