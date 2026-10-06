"""Physical state observations, explicitly separate from native sensor channels."""

from __future__ import annotations

import mujoco
import numpy as np

from hand_sim.identifiers import ModelIdentifiers


class PhysicalSensors:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 identifiers: ModelIdentifiers):
        self.model = model
        self.data = data
        self.ids = identifiers

    def joint(self, name: str) -> dict:
        i = self.ids.resolve("joint", name)
        qs, vs = self.ids.joint_slices(i)
        return {
            "name": name, "id": i, "source": "MjData / MjModel",
            "position": self.data.qpos[qs].copy(),
            "velocity": self.data.qvel[vs].copy(),
            "qpos_indices": np.arange(qs.start, qs.stop),
            "qvel_indices": np.arange(vs.start, vs.stop),
            "limited": bool(self.model.jnt_limited[i]),
            "limits": self.model.jnt_range[i].copy() if self.model.jnt_limited[i] else None,
            "anchor_world": self.data.xanchor[i].copy(),
            "type": mujoco.mjtJoint(int(self.model.jnt_type[i])).name,
        }

    def native_sensor(self, name: str) -> np.ndarray:
        """Only return a registered MuJoCo sensor's sensordata slice."""
        i = self.ids.resolve("sensor", name)
        start = int(self.model.sensor_adr[i])
        dim = int(self.model.sensor_dim[i])
        return self.data.sensordata[start:start + dim].copy()

    def muscle(self, name: str) -> dict:
        i = self.ids.resolve("actuator", name)
        if i not in self.ids.muscle_ids:
            raise ValueError(f"Actuator {name!r} does not use the MuJoCo muscle force law")
        start = int(self.model.actuator_actadr[i])
        count = int(self.model.actuator_actnum[i])
        return {
            "name": name, "id": i, "source": "MjData",
            "excitation": float(self.data.ctrl[i]),
            "activation": self.data.act[start:start + count].copy()
            if start >= 0 and count else None,
            "activation_indices": list(range(start, start + count)) if start >= 0 else [],
            "actuator_force": float(self.data.actuator_force[i]),
            "actuator_length": float(self.data.actuator_length[i]),
            "actuator_velocity": float(self.data.actuator_velocity[i]),
        }

    def tendon(self, name: str) -> dict:
        i = self.ids.resolve("tendon", name)
        # Only a tendon transmission gives a direct scalar actuator contribution.
        attached = np.flatnonzero(
            (self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON)
            & (self.model.actuator_trnid[:, 0] == i)
        )
        contributions = [
            {"actuator_id": int(a), "actuator_name": self.ids.name("actuator", int(a)),
             "signed_force": float(self.model.actuator_gear[a, 0] * self.data.actuator_force[a])}
            for a in attached
        ]
        return {
            "name": name, "id": i, "source": "MjData",
            "length": float(self.data.ten_length[i]),
            "velocity": float(self.data.ten_velocity[i]),
            "actuator_force_contributions": contributions,
            "force_contribution_source": "derived: actuator_force * scalar transmission gear",
            "total_tendon_force": None,
            "unavailable_reason": "MjData exposes no total per-tendon force channel",
        }

    def capabilities(self) -> dict:
        return {
            "native_sensor_count": int(self.model.nsensor),
            "native_sensor_data_dimension": int(self.model.nsensordata),
            "direct_state": ["qpos", "qvel", "ctrl", "act", "actuator_force",
                             "actuator_length", "actuator_velocity", "ten_length",
                             "ten_velocity", "xpos", "xquat", "site_xpos", "xanchor"],
            "derived": ["site_world_velocity", "contact_wrench_contact_frame",
                        "actuator_contribution_to_tendon_force"],
            "unavailable": ["synthetic_emg", "motor_unit_activity",
                            "separate_muscle_fiber_force", "total_tendon_force_sensor"],
        }
