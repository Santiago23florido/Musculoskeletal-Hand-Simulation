"""Native MuJoCo identifiers and model-derived anatomical inventory."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

OBJECTS = {
    "body": (mujoco.mjtObj.mjOBJ_BODY, "nbody"),
    "joint": (mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
    "actuator": (mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"),
    "tendon": (mujoco.mjtObj.mjOBJ_TENDON, "ntendon"),
    "site": (mujoco.mjtObj.mjOBJ_SITE, "nsite"),
    "sensor": (mujoco.mjtObj.mjOBJ_SENSOR, "nsensor"),
}
JOINT_WIDTHS = {
    mujoco.mjtJoint.mjJNT_FREE: (7, 6),
    mujoco.mjtJoint.mjJNT_BALL: (4, 3),
    mujoco.mjtJoint.mjJNT_SLIDE: (1, 1),
    mujoco.mjtJoint.mjJNT_HINGE: (1, 1),
}


class ModelIdentifiers:
    def __init__(self, model: mujoco.MjModel, fingertip_ids: tuple[int, ...] = ()):
        self.model = model
        self.fingertip_ids = tuple(int(i) for i in fingertip_ids)
        if any(i < 0 or i >= model.nsite for i in self.fingertip_ids):
            raise ValueError("Upstream fingertip IDs must identify existing sites")

    def resolve(self, kind: str, name: str) -> int:
        """Resolve an existing name; never let MuJoCo's -1 index select the last object."""
        obj, _ = OBJECTS[kind]
        result = mujoco.mj_name2id(self.model, obj, name)
        if result < 0:
            raise KeyError(f"Unknown {kind}: {name!r}")
        return result

    def name(self, kind: str, object_id: int) -> str | None:
        obj, count = OBJECTS[kind]
        if not 0 <= object_id < getattr(self.model, count):
            raise IndexError(f"Invalid {kind} ID: {object_id}")
        return mujoco.mj_id2name(self.model, obj, object_id)

    def joint_slices(self, joint_id: int) -> tuple[slice, slice]:
        self.name("joint", joint_id)
        kind = mujoco.mjtJoint(int(self.model.jnt_type[joint_id]))
        nq, nv = JOINT_WIDTHS[kind]
        q = int(self.model.jnt_qposadr[joint_id])
        v = int(self.model.jnt_dofadr[joint_id])
        return slice(q, q + nq), slice(v, v + nv)

    @property
    def muscle_ids(self) -> np.ndarray:
        """Identify muscle force laws from metadata, rather than actuator names."""
        return np.flatnonzero(self.model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)

    def inventory(self) -> dict:
        m = self.model
        result = {
            "schema_version": 1,
            "dimensions": {
                key: int(getattr(m, key))
                for key in (
                    "nq",
                    "nv",
                    "nu",
                    "na",
                    "nbody",
                    "njnt",
                    "ntendon",
                    "nsite",
                    "nsensor",
                    "nsensordata",
                )
            },
            "timestep_seconds": float(m.opt.timestep),
        }
        plural = {
            "body": "bodies",
            "joint": "joints",
            "actuator": "actuators",
            "tendon": "tendons",
            "site": "sites",
            "sensor": "sensors",
        }
        for kind, (_, count) in OBJECTS.items():
            entries = []
            for i in range(getattr(m, count)):
                entry = {"id": i, "name": self.name(kind, i)}
                if kind == "body":
                    entry["parent_id"] = int(m.body_parentid[i])
                elif kind == "joint":
                    qs, vs = self.joint_slices(i)
                    entry.update(
                        type=mujoco.mjtJoint(int(m.jnt_type[i])).name,
                        body_id=int(m.jnt_bodyid[i]),
                        qpos_indices=list(range(qs.start, qs.stop)),
                        qvel_indices=list(range(vs.start, vs.stop)),
                        dofs=vs.stop - vs.start,
                        limited=bool(m.jnt_limited[i]),
                        range=m.jnt_range[i].tolist() if m.jnt_limited[i] else None,
                    )
                elif kind == "actuator":
                    entry.update(
                        is_muscle=bool(i in self.muscle_ids),
                        dynamics_type=mujoco.mjtDyn(int(m.actuator_dyntype[i])).name,
                        gain_type=mujoco.mjtGain(int(m.actuator_gaintype[i])).name,
                        transmission_type=mujoco.mjtTrn(int(m.actuator_trntype[i])).name,
                        transmission_ids=m.actuator_trnid[i].tolist(),
                        activation_address=int(m.actuator_actadr[i]),
                        activation_dimension=int(m.actuator_actnum[i]),
                        control_limited=bool(m.actuator_ctrllimited[i]),
                        control_range=m.actuator_ctrlrange[i].tolist(),
                    )
                elif kind == "site":
                    entry.update(
                        body_id=int(m.site_bodyid[i]), is_fingertip=i in self.fingertip_ids
                    )
                elif kind == "sensor":
                    entry.update(
                        type=mujoco.mjtSensor(int(m.sensor_type[i])).name,
                        dimension=int(m.sensor_dim[i]),
                        data_address=int(m.sensor_adr[i]),
                        object_type=int(m.sensor_objtype[i]),
                        object_id=int(m.sensor_objid[i]),
                    )
                entries.append(entry)
            result[plural[kind]] = entries
        result["muscles"] = [dict(a) for a in result["actuators"] if a["is_muscle"]]
        result["fingertips"] = [dict(s) for s in result["sites"] if s["is_fingertip"]]
        return result

    def export(self, path: str | Path, *, metadata: dict | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        inventory = self.inventory()
        if metadata is not None:
            inventory["provenance"] = metadata
        destination.write_text(
            json.dumps(inventory, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return destination

    def export_names(self, path: str | Path) -> Path:
        """Write a readable list of every native anatomical name and ID."""
        inventory = self.inventory()
        lines = [
            "# Loaded MyoHand anatomical names",
            "",
            "Generated from the installed official model using `scripts/inspect_model.py`.",
            "Detailed indices, limits and provenance: [model_inventory.json](model_inventory.json).",
            "Regenerate both files with:",
            "",
            "```powershell",
            "uv run --locked python scripts/inspect_model.py "
            "--output docs/model_inventory.json --names-output docs/anatomical_names.md",
            "```",
            "",
        ]
        for group in ("bodies", "joints", "muscles", "tendons", "sites", "fingertips", "sensors"):
            entries = inventory[group]
            lines.extend([f"## {group.capitalize()} ({len(entries)})", ""])
            if entries:
                lines.extend(["| Native ID | Native name |", "| --- | --- |"])
                lines.extend(f"| {e['id']} | `{e['name']}` |" for e in entries)
            else:
                lines.append("No registered channels in this upstream environment.")
            lines.append("")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(lines), encoding="utf-8")
        return destination
