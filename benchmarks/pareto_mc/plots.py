"""Publication-style diagnostics from already sealed validation predictions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .common import OUTPUT, sha256, stub  # noqa: E402


def plot(destination, *, variant="all_feedback", snr=20):
    name = stub("sub-04", "triangular-ramp5s", snr)
    folders = {"Pareto-rerun": OUTPUT / "L0/validation"}
    for current in ("all_feedback", "guarded_feedback"):
        folder = OUTPUT / "L1" / current / "final_validation"
        if (folder / "sealed.json").exists():
            folders[f"L1 {current}"] = folder
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for label, folder in folders.items():
        path = folder / "predictions" / f"{name}.npz"
        seal = json.loads((folder / "sealed.json").read_text(encoding="utf-8"))
        if sha256(path) != seal["files"][path.relative_to(folder).as_posix()]:
            raise ValueError("A diagnostic prediction was changed after sealing")
        with np.load(path) as archive:
            prediction = {key: archive[key] for key in archive.files}
        result = json.loads((folder / "scores" / f"{name}.json").read_text(encoding="utf-8"))
        amplitude = result["diagnostics"]["amplitude"]
        seconds = sorted(
            int(key[1:]) for key in amplitude if key.startswith("s") and key[1:].isdigit()
        )
        ratio = [amplitude[f"s{second}"]["median_ratio_vs_calibration"] for second in seconds]
        axes[0, 0].plot(seconds, ratio, label=label)
        centroids = prediction["centroid"]
        time = np.arange(len(centroids)) * 204 / 2048
        normalized = centroids[:, 1] / np.maximum(centroids[0, 1], 1e-7)
        axes[0, 1].plot(time, np.median(normalized, axis=1), label=label)
        cycle_rows = [row["cycles"] for row in result["rows"]]
        cycle_mean = np.mean(cycle_rows, axis=0) * 100
        axes[1, 0].plot(np.arange(1, len(cycle_mean) + 1), cycle_mean, "o-", label=label)
        if "state" in prediction:
            state = prediction["state"]
            axes[1, 1].plot(time, state[:, 0] * 40, label=label)
            axes[2, 0].scatter(state[:, 2], np.ones(len(state)), s=3, alpha=0.2, label=label)
        if "batch_ms" in prediction:
            durations = prediction["batch_ms"]
            axes[2, 1].plot(np.sort(durations), np.linspace(0, 1, len(durations)), label=label)
    axes[0, 0].axhline(1, color="gray", ls=":")
    axes[0, 0].set(xlabel="Time (s)", ylabel="True-spike amplitude / calibration median")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Median spike centroid / initial centroid")
    axes[1, 0].set(xlabel="Angular excursion cycle", ylabel="Mean unit RoA (%)", ylim=(0, 100))
    axes[1, 1].set(xlabel="Time (s)", ylabel="Blind angle estimate (deg)", ylim=(0, 40))
    axes[2, 0].set(xlabel="Blind-state speed (1/s)", ylabel="Whitening multiplier (identity)")
    axes[2, 1].axvline(22, color="firebrick", ls="--", label="22 ms target")
    axes[2, 1].set(
        xlabel="CPU batch duration (ms)", ylabel="Empirical cumulative fraction", xscale="log"
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    fig.suptitle(
        f"Pareto-MC validation diagnostics — sub-04 ramp5s, {snr} dB\n"
        "C2 memory inactive; C3 multipliers=1, beta=0 at L0/L1"
    )
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Never overwrite a published figure")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--snr", type=int, default=20)
    args = parser.parse_args()
    plot(args.destination, snr=args.snr)
