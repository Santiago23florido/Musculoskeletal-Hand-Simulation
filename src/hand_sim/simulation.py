"""Load the official CPU hand and advance its unmodified MuJoCo physics."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np
import yaml

from hand_sim.identifiers import ModelIdentifiers
from hand_sim.sensors import PhysicalSensors

DEFAULT_ENV = "myoHandPoseFixed-v0"


class HandSimulation:
    def __init__(self, *, env_id: str = DEFAULT_ENV, seed: int = 0, substeps: int = 1):
        if env_id != DEFAULT_ENV:
            raise ValueError(f"This observation package supports only {DEFAULT_ENV!r}")
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        import myosuite

        self.seed = seed
        self.substeps = substeps
        self._closed = False
        # Native excitation units; do not use Gym's normalized [-1, 1] action transform.
        self._env = myosuite.make_env(env_id, backend="cpu", normalize_act=False)
        try:
            host = self._env.unwrapped
            self.model = host.model
            self.data = host.data
            if not isinstance(self.model, mujoco.MjModel) or not isinstance(self.data, mujoco.MjData):
                raise RuntimeError("Installed MyoSuite does not expose native CPU MjModel/MjData")
            self.identifiers = ModelIdentifiers(self.model, tuple(host.tip_sids))
            self.sensors = PhysicalSensors(self.model, self.data, self.identifiers)
            self.provenance = {
                "env_id": env_id,
                "backend": "cpu",
                "versions": {p: version(p) for p in ("myosuite", "myo-sim", "mujoco", "numpy")},
                "model_source": "https://github.com/MyoHub/myo_sim",
                "model_recipe": "hand_pose",
                "biomechanical_modifications": [],
                "seed": seed,
                "substeps": substeps,
            }
            self.reset()
        except BaseException:
            self._env.close()
            raise

    @property
    def dt(self) -> float:
        return self.substeps * float(self.model.opt.timestep)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("The simulation is closed")

    def reset(self, *, seed: int | None = None) -> dict:
        self._ensure_open()
        if seed is not None:
            self.seed = seed
        self._env.reset(seed=self.seed)
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, excitation=None, *, substeps: int | None = None) -> dict:
        """Advance native physics, holding the previous control when excitation is None.

        Inputs are a finite vector of length ``model.nu`` in native actuator units.
        Out-of-range commands raise instead of silently clipping. No reward or
        episode termination is computed; 200 steps always means 200 calls to this API.
        """
        self._ensure_open()
        count = self.substeps if substeps is None else substeps
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("substeps must be a positive integer")
        if excitation is not None:
            control = np.asarray(excitation, dtype=float)
            if control.shape != (self.model.nu,) or not np.all(np.isfinite(control)):
                raise ValueError(f"excitation must be a finite vector with shape ({self.model.nu},)")
            limits = self.model.actuator_ctrlrange
            limited = self.model.actuator_ctrllimited.astype(bool)
            if np.any(limited & ((control < limits[:, 0]) | (control > limits[:, 1]))):
                raise ValueError("excitation exceeds native actuator control ranges")
            self.data.ctrl[:] = control
        start = float(self.data.time)
        mujoco.mj_step(self.model, self.data, nstep=count)
        # Refresh position/velocity/force fields at the same time as returned qpos/qvel.
        mujoco.mj_forward(self.model, self.data)
        expected = start + count * float(self.model.opt.timestep)
        arrays = (self.data.qpos, self.data.qvel, self.data.act, self.data.actuator_force)
        if not np.isclose(self.data.time, expected, rtol=1e-10, atol=1e-12) or any(
            not np.all(np.isfinite(a)) for a in arrays
        ):
            raise RuntimeError("MuJoCo failed to advance to a finite physical state")
        return self.get_state()

    def get_state(self) -> dict:
        self._ensure_open()
        return {"time": float(self.data.time), "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(), "activation": self.data.act.copy(),
                "excitation": self.data.ctrl.copy(),
                "actuator_force": self.data.actuator_force.copy()}

    def get_model_inventory(self) -> dict:
        self._ensure_open()
        return {**self.identifiers.inventory(), "provenance": dict(self.provenance)}

    def get_joint_state(self, name: str) -> dict:
        self._ensure_open()
        return self.sensors.joint(name)

    def get_sensor_data(self, name: str) -> np.ndarray:
        self._ensure_open()
        return self.sensors.native_sensor(name)

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True

    def __enter__(self) -> HandSimulation:
        self._ensure_open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def load_model(config: str | Path | None = None, **overrides) -> HandSimulation:
    """Load validated YAML defaults or explicit constructor keyword arguments."""
    settings = {}
    if config is not None:
        with Path(config).open(encoding="utf-8") as stream:
            settings = yaml.safe_load(stream)
        if not isinstance(settings, dict):
            raise ValueError("Simulation configuration must be a YAML mapping")
        unknown = settings.keys() - {"env_id", "seed", "substeps"}
        if unknown:
            raise ValueError(f"Unknown simulation settings: {sorted(unknown)}")
    return HandSimulation(**(settings | overrides))
