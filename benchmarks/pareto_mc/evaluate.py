"""Original event scorer and angle-defined diagnostics, only AFTER prediction seals."""

import numpy as np

from benchmarks.fifo_audit.harness import amplitude_metrics
from benchmarks.robust_mvdr.phases import angle_phases, masked

from .data import load_angle, load_truth


def ramp_cycles(angle, fs, *, calibration_s=5):
    """Connected excursions above 1 degree; one outward/return pair per cycle.

    Cycle boundaries depend on measured angle, never on assumed ramp durations.
    The <1 degree slivers at a turnaround are excluded consistently for all methods.
    """
    moving = np.asarray(angle) > 1
    moving[: round(calibration_s * fs)] = False
    edges = np.diff(np.r_[False, moving, False].astype(np.int8))
    return [
        (int(a), int(b))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
        if b - a >= fs and np.max(angle[a:b]) >= 35
    ]


def score_record(record, prediction, *, phase):
    from adapt_decomp.spikes import rate_of_agreement_paired

    truth = load_truth(record, phase=phase)
    angle = load_angle(record, phase=phase)
    fs = record["config"].fs
    binary = prediction["spikes"]
    roa, _, lag = rate_of_agreement_paired(truth, binary, fs=fs, tol_spike_ms=2)
    nr, npred = truth.sum(0), binary.sum(0)
    tp = np.rint(roa * (nr + npred) / (1 + roa))
    precision = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    recall = np.divide(tp, nr, out=np.zeros_like(tp), where=nr > 0)
    phases = angle_phases(angle, fs)
    phase_scores = {}
    for key, mask in phases.items():
        phase_scores[key] = (
            rate_of_agreement_paired(
                masked(truth, mask), masked(binary, mask), fs=fs, tol_spike_ms=2
            )[0].tolist()
            if mask.sum() >= fs
            else [None] * truth.shape[1]
        )
    cycles = ramp_cycles(angle, fs) if record["condition"].startswith("triangular") else []
    cycle_scores = [
        rate_of_agreement_paired(truth[a:b], binary[a:b], fs=fs, tol_spike_ms=2)[0].tolist()
        for a, b in cycles
    ]
    diagnostic = {}
    if prediction.get("sources") is not None:
        windows = {"calibration": (1, 5)}
        windows.update(
            {
                f"s{second}": (second, min(second + 1, len(truth) / fs))
                for second in range(5, int(len(truth) / fs))
            }
        )
        diagnostic["amplitude"] = amplitude_metrics(prediction["sources"], truth, fs, windows)
    rows = []
    for j in range(truth.shape[1]):
        rows.append(
            dict(
                subject=record["subject"],
                condition=record["condition"],
                snr_db=record["snr_db"],
                unit=j,
                roa=float(roa[j]),
                precision=float(precision[j]),
                recall=float(recall[j]),
                lag_samples=int(lag[j]),
                phases={key: value[j] for key, value in phase_scores.items()},
                cycles=[values[j] for values in cycle_scores],
            )
        )
    return dict(rows=rows, diagnostics=diagnostic, cycle_intervals_samples=cycles)


def summarize(results):
    rows = [row for item in results for row in item["rows"]]
    ramp = [r for r in rows if r["condition"] == "triangular-ramp5s"]

    def average(values):
        return float(np.mean(values) * 100) if len(values) else None

    return dict(
        records=len(results),
        instances=len(rows),
        global_roa_pct=average([r["roa"] for r in rows]),
        ramp5s_roa_pct=average([r["roa"] for r in ramp]),
        ge90_pct=average([r["roa"] >= 0.9 for r in rows]),
        precision_pct=average([r["precision"] for r in rows]),
        recall_pct=average([r["recall"] for r in rows]),
        phases={
            p: average([r["phases"][p] for r in rows if r["phases"][p] is not None])
            for p in ("reference_rest", "moving", "hold_away")
        },
        ramp5s_cycle1_roa_pct=average([r["cycles"][0] for r in ramp if r["cycles"]]),
        ramp5s_cycles2_8_roa_pct=average([v for r in ramp for v in r["cycles"][1:8]]),
        lost_by_condition={
            c: sum(r["recall"] < 0.5 for r in rows if r["condition"] == c)
            for c in sorted({r["condition"] for r in rows})
        },
    )
