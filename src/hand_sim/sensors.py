"""Physical state observations, explicitly separate from native sensor channels."""

from __future__ import annotations

import mujoco
import numpy as np

from hand_sim.identifiers import ModelIdentifiers


class PhysicalSensors:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, identifiers: ModelIdentifiers):
        self.model = model
        self.data = data
        self.ids = identifiers

    def joint(self, name: str) -> dict:
        i = self.ids.resolve("joint", name)
        qs, vs = self.ids.joint_slices(i)
        return {
            "name": name,
            "id": i,
            "source": "MjData / MjModel",
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
        return self.data.sensordata[start : start + dim].copy()

    def muscle(self, name: str) -> dict:
        i = self.ids.resolve("actuator", name)
        if i not in self.ids.muscle_ids:
            raise ValueError(f"Actuator {name!r} does not use the MuJoCo muscle force law")
        start = int(self.model.actuator_actadr[i])
        count = int(self.model.actuator_actnum[i])
        return {
            "name": name,
            "id": i,
            "source": "MjData",
            "excitation": float(self.data.ctrl[i]),
            "activation": self.data.act[start : start + count].copy()
            if start >= 0 and count
            else None,
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
            {
                "actuator_id": int(a),
                "actuator_name": self.ids.name("actuator", int(a)),
                "signed_force": float(self.model.actuator_gear[a, 0] * self.data.actuator_force[a]),
            }
            for a in attached
        ]
        return {
            "name": name,
            "id": i,
            "source": "MjData",
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
            "direct_state": [
                "qpos",
                "qvel",
                "ctrl",
                "act",
                "actuator_force",
                "actuator_length",
                "actuator_velocity",
                "ten_length",
                "ten_velocity",
                "xpos",
                "xquat",
                "site_xpos",
                "xanchor",
            ],
            "derived": [
                "site_world_velocity",
                "contact_wrench_contact_frame",
                "actuator_contribution_to_tendon_force",
            ],
            "unavailable": [
                "synthetic_emg",
                "motor_unit_activity",
                "separate_muscle_fiber_force",
                "total_tendon_force_sensor",
            ],
        }

    def site_position(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self.ids.resolve("site", name)].copy()

    def site_velocity(self, name: str) -> np.ndarray:
        """World-frame translational velocity (m/s), derived from the native Jacobian."""
        i = self.ids.resolve("site", name)
        jacobian = np.empty((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacobian, None, i)
        return jacobian @ self.data.qvel

    def fingertips(self) -> dict:
        return {
            self.ids.name("site", i): {
                "id": i,
                "position_world": self.data.site_xpos[i].copy(),
                "velocity_world": self.site_velocity(self.ids.name("site", i)),
            }
            for i in self.ids.fingertip_ids
        }

    def body(self, name: str) -> dict:
        i = self.ids.resolve("body", name)
        return {
            "id": i,
            "name": name,
            "source": "MjData",
            "position_world": self.data.xpos[i].copy(),
            "quaternion_world_wxyz": self.data.xquat[i].copy(),
            "rotation_matrix_world": self.data.xmat[i].reshape(3, 3).copy(),
        }

    def contacts(self) -> list[dict]:
        """Native collision records; wrench is unavailable for non-constraint contacts."""
        records = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            active = contact.efc_address >= 0
            wrench = np.empty(6) if active else None
            if active:
                mujoco.mj_contactForce(self.model, self.data, i, wrench)
            records.append(
                {
                    "id": i,
                    "geom_ids": [int(contact.geom1), int(contact.geom2)],
                    "geom_names": [
                        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                        if g >= 0
                        else None
                        for g in (contact.geom1, contact.geom2)
                    ],
                    "position_world": contact.pos.copy(),
                    "distance": float(contact.dist),
                    "dimension": int(contact.dim),
                    "frame_world_to_contact": contact.frame.reshape(3, 3).copy(),
                    "constraint_active": active,
                    "wrench_contact_frame_force_then_torque": wrench,
                    "wrench_source": "mj_contactForce" if active else "unavailable: no constraint",
                }
            )
        return records
