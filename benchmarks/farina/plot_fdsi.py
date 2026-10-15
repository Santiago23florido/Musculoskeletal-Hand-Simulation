"""Plot measured FDSI cache reproduction results; no hard-coded baseline scores."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "frozen": ("Frozen", "#7b8490"),
    "contrast": ("Contrast loss", "#5896bd"),
    "pareto": ("Pareto (label-free tuning)", "#217a60"),
    "oracle": ("Oracle (GT tuning)", "#ce8240"),
}
CONDITIONS = [
    "staircase",
    "triangular-ramp40s",
    "triangular-ramp20s",
    "triangular-ramp10s",
    "triangular-ramp5s",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "outputs/farina/fdsi_cached/summary.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/farina/fdsi_cached/comparison.png"
    )
    args = parser.parse_args()
    result = json.loads(args.summary.read_text(encoding="utf-8"))
    if result["status"] != "passed":
        raise ValueError("Plot requires a complete, accepted reproduction")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.65, 1]})
    positions = np.arange(len(CONDITIONS))
    width = 0.19
    for index, (method, (label, color)) in enumerate(METHODS.items()):
        scores = [
            next(
                row["mean_roa_pct"]
                for row in result["by_condition"]
                if row["condition"] == condition and row["method"] == method
            )
            for condition in CONDITIONS
        ]
        axes[0].bar(positions + (index - 1.5) * width, scores, width, color=color, label=label)
        snrs = [15, 20, 25, 30]
        means = result["overall"][method]["mean_roa_by_snr_pct"]
        axes[1].plot(snrs, [means[str(snr)] for snr in snrs], "o-", color=color)
    axes[0].set_xticks(positions, ["Staircase", "Ramp 40 s", "Ramp 20 s", "Ramp 10 s", "Ramp 5 s"])
    axes[0].set_title("Mean RoA by wrist movement")
    axes[1].set_title("Mean RoA by noise level")
    axes[1].set_xticks([15, 20, 25, 30])
    axes[1].set_xlabel("SNR (dB)")
    for axis in axes:
        axis.set_ylim(0, 100)
        axis.set_ylabel("Mean RoA (%)")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("FDSI: 100 recordings, 768 calibrated unit-instances per method", fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncols=4, frameon=False)
    fig.text(
        0.5,
        0.075,
        "Recomputed from authors' cached spikes; full recordings including calibration. "
        "Fixed learning-rate mode.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
