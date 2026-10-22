"""Export sealed memory-level diagnostics without rewriting previous experiments."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .common import ROOT, sha256, stub, write_new  # noqa: E402
from .memory_experiment import read  # noqa: E402


def publish(root):
    gate = read(root / "gate.json")
    level, variant = root.parent.name, root.name
    destination = ROOT / "docs/benchmarks" / f"pareto_mc_{level.lower()}_{variant}"
    for name in ("gate.json", "protocol.json", "selected_memory.json", "trials.json"):
        path = destination / name
        if not path.exists():
            write_new(path, read(root / name))
    source = root / "final_validation"
    name = stub("sub-04", "triangular-ramp5s", 20)
    path = source / "predictions" / f"{name}.npz"
    seal = read(source / "sealed.json")
    if sha256(path) != seal["files"][path.relative_to(source).as_posix()]:
        raise ValueError("Diagnostic prediction changed after its seal")
    with np.load(path) as archive:
        prediction = {key: archive[key] for key in archive.files}
    score = read(source / "scores" / f"{name}.json")
    figure_path = ROOT / "docs/images" / f"pareto-mc-{level.lower()}-{variant}.png"
    if figure_path.exists():
        return
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    amplitude = score["diagnostics"]["amplitude"]
    seconds = sorted(int(k[1:]) for k in amplitude if k.startswith("s") and k[1:].isdigit())
    axes[0, 0].plot(seconds, [amplitude[f"s{s}"]["median_ratio_vs_calibration"] for s in seconds])
    axes[0, 0].axhline(1, color="gray", ls=":")
    axes[0, 0].set(xlabel="Time (s)", ylabel="True-spike amplitude / calibration")
    centroid, memory, state = (prediction[key] for key in ("centroid", "memory", "state"))
    time = np.arange(len(state)) * 204 / 2048
    axes[0, 1].plot(time, np.median(centroid[:, 1] / centroid[0, 1].clip(1e-7), axis=1))
    axes[0, 1].set(xlabel="Time (s)", ylabel="Spike centroid / initial centroid")
    cycles = np.mean([row["cycles"] for row in score["rows"]], axis=0) * 100
    axes[0, 2].plot(np.arange(1, len(cycles) + 1), cycles, "o-")
    axes[0, 2].set(xlabel="Angle-defined cycle", ylabel="Mean RoA (%)", ylim=(0, 100))
    axes[1, 0].plot(time, state[:, 0] * 40)
    axes[1, 0].set(xlabel="Time (s)", ylabel="Blind angle estimate (deg)", ylim=(0, 40))
    axes[1, 1].plot(time, memory[:, 3].mean(axis=1))
    axes[1, 1].set(xlabel="Time (s)", ylabel="Fraction of units with active memory", ylim=(0, 1))
    axes[1, 2].scatter(state[:, 0], memory[:, 0].mean(axis=1), s=3, alpha=0.2)
    axes[1, 2].set(xlabel="Blind state q", ylabel="Mean memory confidence", ylim=(0, 1))
    axes[2, 0].scatter(state[:, 2], memory[:, 1].mean(axis=1), s=3, alpha=0.2)
    axes[2, 0].set(xlabel="Blind speed (1/s)", ylabel="Mean memory beta")
    axes[2, 1].scatter(state[:, 2], np.ones(len(state)), s=3, alpha=0.2)
    axes[2, 1].set(xlabel="Blind speed (1/s)", ylabel="C3 wh/SV multiplier (identity)")
    durations = prediction["batch_ms"]
    axes[2, 2].plot(np.sort(durations), np.linspace(0, 1, len(durations)))
    axes[2, 2].axvline(22, color="firebrick", ls="--", label="22 ms target")
    axes[2, 2].set(xlabel="CPU duration (ms)", ylabel="Cumulative fraction", xscale="log")
    axes[2, 2].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"Pareto-MC {level}/{variant}, sub-04 ramp5s 20dB; gate passed: {gate['passed']}\n"
        "Exploratory validation; latency measured with concurrent record workers"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    publish(parser.parse_args().root)
