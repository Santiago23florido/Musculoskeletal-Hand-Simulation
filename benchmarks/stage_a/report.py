"""Standalone research figures drawn solely from executed experiment artifacts."""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import CONDITIONS, OUTPUT


def plot_learning():
    names = ["full_fifo_squared", "balanced_gru", "balanced_mlp", "state3_gru"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name in names:
        history = json.loads((OUTPUT / "runs" / name / "history.json").read_text())
        rows = history["epochs"]
        values = [
            0.0001 * r["validation"]["whitening_loss"] + 0.1 * r["validation"]["contrast"]
            for r in rows
        ]
        ax.plot([r["epoch"] for r in rows], values, "o-", label=name)
    ax.set(
        xlabel="Epoch",
        ylabel="0.0001 x KL loss + 0.1 x contrast loss",
        title="Stage A: common label-free validation objective",
    )
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.text(
        0.5,
        0.015,
        "Validation: sub-01 ramp10s, 30 dB. Tiny state-smoothness term omitted from plot. "
        "This is not RoA.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT / "learning.png", dpi=180)
    plt.close(fig)


def plot_results():
    folder = OUTPUT / "evaluations/p1_complete"
    result = json.loads((folder / "summary.json").read_text())
    if not result["complete_p1"]:
        raise ValueError("A complete P1 evaluation is required")
    methods = {
        "pareto": ("Published Pareto", "#217a60"),
        "stage_a": ("Stage A, coupled", "#386cb0"),
        "stage_a_no_coupling": ("Stage A, uncoupled", "#df8d35"),
        "identity_sv_sd": ("Fixed whitening + SV/SD", "#8c9198"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    positions = np.arange(len(CONDITIONS))
    for index, (method, (label, color)) in enumerate(methods.items()):
        values = result["summaries"][method]["by_condition"]
        axes[0].bar(
            positions + (index - 1.5) * 0.19,
            [values[c] for c in CONDITIONS],
            0.19,
            label=label,
            color=color,
        )
    axes[0].set_xticks(positions, ["Staircase", "40 s", "20 s", "10 s", "5 s"])
    axes[0].set(xlabel="Wrist movement / ramp duration", ylabel="Mean RoA (%)", ylim=(0, 100))
    stats = result["statistics"]
    deltas = [s["mean_delta_pp"] for s in stats]
    intervals = np.array([s["subject_cluster_bootstrap_95_ci_pp"] for s in stats])
    # A bootstrap percentile interval need not contain the observed mean; show
    # interval segments directly instead of invalid negative matplotlib yerr.
    for index, (mean, interval) in enumerate(zip(deltas, intervals, strict=True)):
        axes[1].plot([index, index], interval, color="#386cb0", linewidth=2)
        axes[1].plot(index, mean, "o", color="#386cb0")
    axes[1].axhline(0, color="#666666", linewidth=1)
    axes[1].axhline(5, color="#217a60", linestyle="--", label="+5 pp ramp target")
    axes[1].set_xticks(
        range(len(stats)), [s["condition"].replace("triangular-ramp", "") for s in stats]
    )
    axes[1].set(
        ylabel="Stage A minus Pareto (percentage points)",
        title="Paired change; subject-cluster 95% CI",
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Stage A P1: 100 recordings, 768 unit-instances per method — {result['status']}")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncols=2, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-only", action="store_true")
    args = parser.parse_args()
    plot_learning()
    if not args.learning_only:
        plot_results()
