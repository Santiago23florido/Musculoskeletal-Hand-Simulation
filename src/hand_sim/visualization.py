"""Optional standard MuJoCo viewer; the simulation never imports GUI code."""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from hand_sim.simulation import HandSimulation


def hand_camera(sim: HandSimulation) -> mujoco.MjvCamera:
    """Frame the moving hand from native joint anchors and upstream fingertip sites."""
    points = np.concatenate(
        (sim.data.xanchor, sim.data.site_xpos[list(sim.identifiers.fingertip_ids)])
    )
    low, high = points.min(axis=0), points.max(axis=0)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = (low + high) / 2
    camera.distance = max(0.35, float(np.linalg.norm(high - low)) * 1.8)
    camera.azimuth = 140
    camera.elevation = -25
    return camera


def launch_viewer(
    sim: HandSimulation, *, duration: float | None = None, excitation=None, realtime: bool = True
) -> int:
    """Display native geometry and tendons while advancing the same CPU simulation.

    A finite duration is useful for an explicit graphics smoke test. The viewer
    stops after that many wall-clock seconds or when the window is closed.
    """
    sim._ensure_open()
    if duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and positive")
    if sys.platform.startswith("linux") and not (
        os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")
    ):
        raise RuntimeError("No graphical display. Use headless simulation or enable WSLg/X11.")
    import mujoco.viewer

    count = 0
    camera = hand_camera(sim)
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth = camera.azimuth
            viewer.cam.elevation = camera.elevation
            viewer.opt.tendongroup[:] = 1
            # Hundreds of muscle attachment sites would obscure the anatomy.
            viewer.opt.sitegroup[:] = 0
        started = time.perf_counter()
        while viewer.is_running():
            if duration is not None and time.perf_counter() - started >= duration:
                break
            tick = time.perf_counter()
            with viewer.lock():
                sim.step(excitation)
            viewer.sync()
            count += 1
            if realtime:
                time.sleep(max(0, sim.dt - (time.perf_counter() - tick)))
    return count
