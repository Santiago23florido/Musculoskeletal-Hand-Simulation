"""Phases defined from the recorded joint angle instead of fixed time windows."""

import numpy as np


def angle_phases(angle_deg, fs, *, calibration_s=5.0, rest_deg=1.0, rest_speed=0.5, smooth_s=0.1):
    """Boolean sample masks: rest at the reference after calibration, moving, holds away.

    Speed is the derivative of a centred moving average over ``smooth_s`` seconds, so
    quantisation of the angle trace cannot create spurious movement.
    """
    angle = np.asarray(angle_deg, dtype=float)
    width = max(1, int(round(smooth_s * fs)))
    smooth = np.convolve(angle, np.ones(width) / width, mode="same")
    edge = width // 2 + 1
    smooth[:edge], smooth[-edge:] = angle[:edge], angle[-edge:]
    speed = np.abs(np.gradient(smooth) * fs)
    moving = speed > rest_speed
    at_reference = np.abs(angle) <= rest_deg
    after_calibration = np.arange(angle.size) >= int(round(calibration_s * fs))
    return dict(
        reference_rest=at_reference & ~moving & after_calibration,
        moving=moving,
        hold_away=~at_reference & ~moving,
    )


def describe(angle_deg, fs):
    """What the fixed reference windows actually contain."""
    angle = np.asarray(angle_deg, dtype=float)
    masks = angle_phases(angle, fs)
    first = slice(0, int(10 * fs))
    return dict(
        angle_at_5s_deg=float(angle[int(5 * fs)]),
        angle_at_10s_deg=float(angle[int(10 * fs) - 1]),
        max_abs_angle_first_10s_deg=float(np.abs(angle[first]).max()),
        moving_fraction_first_10s=float(masks["moving"][first].mean()),
        moving_fraction_last_5s=float(masks["moving"][-int(5 * fs) :].mean()),
        fraction=dict({k: float(v.mean()) for k, v in masks.items()}),
    )


def masked(binary, mask):
    """Zero spike trains outside a phase; the reference scorer then sees only that phase."""
    out = np.array(binary, copy=True)
    out[~mask] = 0
    return out
