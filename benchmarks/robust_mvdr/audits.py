"""Two audits requested by the independent review.

``phase_audit`` reports what the fixed reference phase windows contain, using the angle
traces only. ``target_reliability`` measures the noise ceiling of Stage B's
spike-triggered MUAP targets: if two disjoint halves of the same posture bin disagree as
much as the calibration template disagrees with the bin, the targets cannot teach a
morph network anything in the separation geometry. It uses only ``sub-01``, which is a
training subject in every Stage B partition, so no outer-subject label is opened.
"""

import numpy as np
import torch

from benchmarks.stage_b.common import RADIUS
from benchmarks.stage_b.data import prepare, windows_mean
from benchmarks.stage_b.inference import extended_template

from .common import CONDITIONS, DATA, OUTPUT, SUBJECTS, write_json
from .phases import describe

AUDIT_SUBJECT = "sub-01"
AUDIT_CONDITIONS = ("staircase", "triangular-ramp40s", "triangular-ramp10s")


def phase_audit():
    result = {}
    for subject in SUBJECTS:
        for condition in CONDITIONS:
            path = DATA / "data" / subject / "clean" / f"{subject}_FDSI_{condition}_angle.npz"
            with np.load(path) as archive:
                result[f"{subject}/{condition}"] = describe(archive["angle"], 2048)
    write_json(OUTPUT / "audits/phase_audit.json", result)
    for key, value in result.items():
        if key.startswith(SUBJECTS[1]):
            print(
                key,
                f"angle@5s={value['angle_at_5s_deg']:.2f}",
                f"angle@10s={value['angle_at_10s_deg']:.2f}",
                f"moving share of 0-10s={value['moving_fraction_first_10s']:.2f}",
                f"moving share of last 5s={value['moving_fraction_last_5s']:.2f}",
            )
    return result


def _shape(signal, times, amplitude):
    """Spike-triggered clean-EMG volume, normalised like Stage B targets."""
    t = windows_mean(signal, times) / amplitude
    return t.reshape(10, 10, 2 * RADIUS).permute(2, 0, 1)[None, None]


def _projected(volume, projection, amplitude):
    """The lag-zero extended template in the separator's projected geometry."""
    return extended_template(volume, projection, amplitude)[0]


def _nmse(a, b):
    return float((a - b).square().sum() / b.square().sum().clamp_min(1e-12))


@torch.no_grad()
def target_reliability(snr=30, minimum_events=40):
    """Split-half noise ceiling of posture-binned targets, raw and in separation geometry.

    ``||odd - even||^2`` estimates four times the noise energy of the full-bin average,
    hence the division by four. A change-to-noise ratio near one means the targets
    carry no learnable posture signal in that geometry.
    """
    from adapt_decomp.preprocessing import filter_kwargs, preprocess_emg
    from adapt_decomp.spikes import rate_of_agreement_paired

    rows = []
    for condition in AUDIT_CONDITIONS:
        record = prepare(AUDIT_SUBJECT, condition, snr)
        calibration, config = record["calibration"], record["config"]
        clean = DATA / "data" / AUDIT_SUBJECT / "clean"
        with np.load(
            clean / f"{AUDIT_SUBJECT}_FDSI_{condition}_spikes.npz", allow_pickle=True
        ) as a:
            timings = a["spikes"][calibration.gt_matched_indices]
        truth_cal = np.zeros_like(calibration.spikes)
        for unit, times in enumerate(timings):
            times = np.asarray(times, dtype=int)
            truth_cal[times[times < len(truth_cal)], unit] = 1
        _, _, lag = rate_of_agreement_paired(
            truth_cal, calibration.spikes, fs=config.fs, tol_spike_ms=2
        )
        with np.load(clean / f"{AUDIT_SUBJECT}_FDSI_{condition}_emg.npz") as a:
            signal = np.asarray(a["emg"], dtype=np.float32)
        signal = torch.as_tensor(
            preprocess_emg(signal, config.fs, **filter_kwargs(config)), dtype=torch.float32
        )
        with np.load(clean / f"{AUDIT_SUBJECT}_FDSI_{condition}_angle.npz") as a:
            q = a["angle"] / 40
        projection, amplitude = record["projection"], record["amplitude"]
        for unit, times in enumerate(timings):
            times = np.asarray(times, dtype=int) - int(lag[unit])
            times = times[(times >= RADIUS) & (times < len(signal) - RADIUS)]
            calibration_volume = record["template"][unit : unit + 1]
            for center in np.linspace(0, 1, 9):
                selected = times[np.abs(q[times] - center) <= 1 / 16 + 1e-5]
                if len(selected) < minimum_events:
                    continue
                full, odd, even = (
                    _shape(signal, selected, amplitude[unit]),
                    _shape(signal, selected[1::2], amplitude[unit]),
                    _shape(signal, selected[0::2], amplitude[unit]),
                )
                scale = amplitude[unit : unit + 1]
                rows.append(
                    dict(
                        condition=condition,
                        unit=unit,
                        posture=float(center),
                        events=int(len(selected)),
                        raw_target_noise_nmse=_nmse(odd, even) / 4,
                        raw_change_nmse=_nmse(calibration_volume, full),
                        projected_target_noise_nmse=_nmse(
                            _projected(odd, projection, scale), _projected(even, projection, scale)
                        )
                        / 4,
                        projected_change_nmse=_nmse(
                            _projected(calibration_volume, projection, scale),
                            _projected(full, projection, scale),
                        ),
                    )
                )
    summary = {}
    for key in (
        "raw_target_noise_nmse",
        "raw_change_nmse",
        "projected_target_noise_nmse",
        "projected_change_nmse",
    ):
        summary[key] = float(np.median([r[key] for r in rows])) if rows else None
    moved = [r for r in rows if r["posture"] >= 0.25]
    summary["projected_signal_to_noise_ratio_moved_bins"] = (
        float(
            np.median(
                [
                    r["projected_change_nmse"] / max(r["projected_target_noise_nmse"], 1e-12)
                    for r in moved
                ]
            )
        )
        if moved
        else None
    )
    write_json(
        OUTPUT / "audits/target_reliability.json",
        dict(subject=AUDIT_SUBJECT, snr_db=snr, summary=summary, rows=rows),
    )
    print(summary)
    return summary
