"""Evaluate a tracking variant using the SAME calibrated units and frozen alignment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from metrics import score_spikes

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions", type=Path, help="NPZ, binary spikes[samples, calibrated units]"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "outputs/farina/tutorial_legacy/evaluation_reference.npz",
    )
    parser.add_argument("--start-seconds", type=float, default=30.0)
    parser.add_argument("--stop-seconds", type=float)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/farina/variant_scores.csv")
    args = parser.parse_args()
    with np.load(args.reference, allow_pickle=False) as reference:
        truth = reference["spikes"]
        fs = int(reference["fs"])
        lags = reference["alignment_samples"]
        gt_indices = reference["ground_truth_indices"]
    with np.load(args.predictions, allow_pickle=False) as archive:
        predicted = archive["spikes"]
    if predicted.shape != truth.shape or not np.isin(predicted, [0, 1]).all():
        raise ValueError(
            "Predictions must be binary and keep the calibrated units' shape and order"
        )
    start = round(args.start_seconds * fs)
    stop = len(truth) if args.stop_seconds is None else round(args.stop_seconds * fs)
    if not 0 <= start < stop <= len(truth):
        raise ValueError("Evaluation window must lie inside the recording")
    rows = []
    for unit in range(truth.shape[1]):
        result = score_spikes(
            np.flatnonzero(truth[:, unit]),
            np.flatnonzero(predicted[:, unit]),
            lag=int(lags[unit]),
            tolerance=round(fs * 0.002),
            start=start,
            stop=stop,
        )
        rows.append({"unit": unit, "gt_unit": int(gt_indices[unit]), **result})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    active = [row for row in rows if row["n_reference"] > 0]
    print(
        json.dumps(
            {
                "window_seconds": [start / fs, stop / fs],
                "units": len(rows),
                "active_reference_units": len(active),
                "mean_over_active_reference_units": {
                    key: float(np.mean([row[key] for row in active])) if active else None
                    for key in ("roa", "precision", "recall", "f1")
                },
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
