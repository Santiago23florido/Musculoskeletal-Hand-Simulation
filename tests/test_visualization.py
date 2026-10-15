import os

import numpy as np
import pytest

from hand_sim.visualization import hand_camera, launch_viewer


def test_camera_frames_the_native_hand(sim):
    camera = hand_camera(sim)
    assert np.all(np.isfinite(camera.lookat))
    assert camera.distance > 0


@pytest.mark.graphics
@pytest.mark.skipif(
    os.environ.get("HAND_SIM_TEST_GRAPHICS") != "1",
    reason="Set HAND_SIM_TEST_GRAPHICS=1 on a graphical desktop",
)
def test_standard_mujoco_viewer_launches_and_steps(sim):
    assert launch_viewer(sim, duration=0.5) > 0
    assert sim.data.time > 0
