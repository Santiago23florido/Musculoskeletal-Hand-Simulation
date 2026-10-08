"""Owned numeric snapshots and portable NumPy archives for physical inspection."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import mujoco
import numpy as np

STATE_FIELDS = {
    "qpos": "qpos", "qvel": "qvel", "qacc": "qacc",
    "excitation": "ctrl", "activation": "act",
    "actuator_force": "actuator_force", "actuator_length": "actuator_length",
    "actuator_velocity": "actuator_velocity",
    "tendon_length": "ten_length", "tendon_velocity": "ten_velocity",
    "body_position_world": "xpos", "body_quaternion_world_wxyz": "xquat",
    "site_position_world": "site_xpos", "joint_anchor_world": "xanchor",
    "native_sensor_data": "sensordata",
}


def snapshot(data: mujoco.MjData) -> dict:
    """Copy every field so a subsequent physics step cannot mutate a past query."""
    return {"time": float(data.time), **{
        key: np.array(getattr(data, field), copy=True) for key, field in STATE_FIELDS.items()
    }}


def save_states(path: str | Path, states: Iterable[Mapping], *, metadata: dict | None = None
                ) -> Path:
    """Save one or more matching snapshots, loadable with ``allow_pickle=False``.

    Each numeric array has a leading sample dimension. Metadata is a scalar JSON
    string. These observations are for debugging; they are not restart checkpoints.
    """
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("State archives must have a .npz suffix")
    samples = list(states)
    if not samples:
        raise ValueError("At least one physical state is required")
    keys = set(samples[0])
    if "metadata_json" in keys or any(set(sample) != keys for sample in samples):
        raise ValueError("State snapshots must have matching keys without metadata_json")
    arrays = {key: np.stack([np.asarray(sample[key]) for sample in samples])
              for key in sorted(keys)}
    if any(a.dtype.kind not in "biufc" for a in arrays.values()):
        raise ValueError("State fields must be numeric; object arrays are not supported")
    arrays["metadata_json"] = np.asarray(json.dumps(
        {"schema_version": 1, **(metadata or {})}, allow_nan=False, sort_keys=True
    ))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination
