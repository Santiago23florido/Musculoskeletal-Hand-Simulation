"""Plot actual plateau scores saved by run_tutorial.py (median and interquartile range)."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/farina/tutorial_legacy")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with (args.run / "scores.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, ax = plt.subplots(figsize=(7.8, 4.5), layout="constrained")
    for method, label, color in (
        ("fixed", "Fixed calibration", "#b85425"),
        ("adapted", "Adaptive decomposition", "#2464a2"),
    ):
        selected = [r for r in rows if r["method"] == method and r["window"].startswith("plateau_")]
        angles = sorted({float(r["window"][8:-3]) for r in selected}, reverse=True)
        scores = [
            [float(r["roa"]) * 100 for r in selected if float(r["window"][8:-3]) == angle]
            for angle in angles
        ]
        quartiles = np.percentile(scores, [25, 50, 75], axis=1)
        x = -np.asarray(angles)
        ax.plot(x, quartiles[1], "o-", color=color, label=label)
        ax.fill_between(x, quartiles[0], quartiles[2], color=color, alpha=0.16)
    ax.set(
        xlabel="Wrist flexion from neutral (degrees)",
        ylabel="Rate of agreement (%)",
        title="Official NeuroMotion tutorial: 30 dB noise, 18 calibrated units",
        xticks=[0, 10, 20, 30, 40],
        ylim=(-2, 102),
    )
    ax.text(
        0.02,
        0.03,
        "Median and IQR across units; pairing and lag frozen at calibration",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.07))
    destination = args.output or args.run / "comparison.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
