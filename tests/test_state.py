import json

import numpy as np
import pytest

from hand_sim.state import save_states


def test_physical_state_archive_roundtrip_without_pickle(sim, tmp_path):
    samples = [sim.get_state(), sim.step(np.full(sim.model.nu, 0.02))]
    path = save_states(tmp_path / "states.npz", samples, metadata={"provenance": sim.provenance})
    with np.load(path, allow_pickle=False) as archive:
        assert archive["time"].shape == (2,)
        assert archive["qpos"].shape == (2, sim.model.nq)
        assert archive["native_sensor_data"].shape == (2, 0)
        assert archive["site_position_world"].shape == (2, sim.model.nsite, 3)
        for key in samples[0]:
            np.testing.assert_array_equal(archive[key][1], samples[1][key])
        assert json.loads(str(archive["metadata_json"]))["provenance"] == sim.provenance


def test_reject_empty_mismatched_or_object_state_archives(tmp_path):
    for samples in ([], [{"a": 1}, {"b": 1}], [{"a": object()}]):
        with pytest.raises(ValueError):
            save_states(tmp_path / "invalid.npz", samples)
    with pytest.raises(ValueError, match="suffix"):
        save_states(tmp_path / "invalid.npy", [{"time": 0}])
