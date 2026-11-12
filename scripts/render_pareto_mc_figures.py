"""Render the Pareto-MC README figures from the versioned result summaries.

    .venv-benchmarks/Scripts/python.exe scripts/render_pareto_mc_figures.py

Writes docs/images/pareto_mc_decoder.png (how the decoder works) and
docs/images/pareto_mc_results.png (accuracy by development stage). The results figure
reads docs/benchmarks/pareto_mc_p3_closed_context/analysis.json and, when present,
heldout_analysis.json for the reserved subject.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/benchmarks/pareto_mc_p3_closed_context"
IMAGES = ROOT / "docs/images"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL_FILL = "#efeeea"
NEUTRAL_EDGE = "#8a8984"
LEARNED_FILL = "#e3eefc"
LEARNED_EDGE = "#2a78d6"
# Ordinal blue ramp (validated: monotone lightness, light end >= 2:1 on the surface).
STAGES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def box(ax, x, y, w, h, title, body="", learned=False, size=10.5):
    fill, edge = (LEARNED_FILL, LEARNED_EDGE) if learned else (NEUTRAL_FILL, NEUTRAL_EDGE)
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.25,rounding_size=1.2",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    cx = x + w / 2
    if body:
        ax.text(cx, y + h - 1.6, title, ha="center", va="top", fontsize=size, weight="bold")
        ax.text(
            cx,
            y + h - 4.6,
            body,
            ha="center",
            va="top",
            fontsize=size - 1.5,
            color=INK_2,
            linespacing=1.35,
        )
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=size, weight="bold")


def arrow(ax, points, label="", label_at=None, dashed=False, color=INK_2, align="center"):
    style = (0, (4, 3)) if dashed else "-"
    for start, end in zip(points[:-2], points[1:-1], strict=True):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=1.4, linestyle=style)
    head = FancyArrowPatch(
        points[-2],
        points[-1],
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color=color,
        linestyle=style,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(head)
    if label:
        lx, ly = label_at
        ax.text(
            lx,
            ly,
            label,
            ha=align,
            va="center",
            fontsize=9,
            color=INK_2,
            backgroundcolor=SURFACE,
            linespacing=1.3,
        )


def closed_loop_panel(ax):
    ax.text(
        0.5,
        45,
        "a   Closed-loop decoding, one 100-ms batch at a time",
        fontsize=13,
        weight="bold",
        va="top",
    )
    box(ax, 1, 21, 14, 10, "HD-sEMG", "100-ms batch\n(raw samples)")
    box(
        ax,
        20,
        20,
        26,
        12,
        "Pareto adaptive decomposition",
        "causal filtering, whitening,\nseparation vectors, peak detection\n"
        "+ calibration filter memory (L2)",
    )
    box(ax, 51, 21, 15, 10, "Candidates", "about 50 per\ntrue discharge")
    box(
        ax,
        71,
        20,
        19,
        12,
        "Context head",
        "learned decision for each\ncandidate (setrec24,\npanel b)",
        learned=True,
    )
    box(ax, 95, 21, 14, 10, "Motor-unit\nspike trains")
    box(
        ax,
        71,
        2,
        19,
        10,
        "Posture observer",
        "GRU on EMG features only;\nestimates the wrist angle q",
        learned=True,
    )
    arrow(ax, [(15.6, 26), (19.4, 26)])
    arrow(ax, [(46.6, 26), (50.4, 26)])
    arrow(ax, [(66.6, 26), (70.4, 26)])
    arrow(ax, [(90.6, 26), (94.4, 26)])
    arrow(ax, [(8, 20.4), (8, 7), (70.4, 7)], label="EMG features", label_at=(40, 7))
    arrow(ax, [(80.5, 12.6), (80.5, 19.4)], label="q, Δq", label_at=(82, 16), align="left")
    arrow(
        ax,
        [(80.5, 32.6), (80.5, 38), (33, 38), (33, 32.6)],
        color=LEARNED_EDGE,
        label="accepted discharges update each unit's filter and centroids (closed loop)",
        label_at=(56.5, 38),
    )


def context_head_panel(ax):
    ax.text(
        0.5,
        45,
        "b   Context head (setrec24), for the candidates of one motor unit in batch t",
        fontsize=13,
        weight="bold",
        va="top",
    )
    box(
        ax,
        1,
        24.5,
        19,
        12,
        "Inputs per candidate",
        "22 decomposition features\nposture q and 2-s change Δq\ntime offset in the batch",
    )
    box(ax, 25, 26, 14, 9, "Embedding", "MLP 25 → 32 → 32", learned=True)
    box(ax, 44, 25.5, 14, 10, "FiLM", "scale and shift\nfrom unit memory", learned=True)
    box(
        ax,
        63,
        22.75,
        20,
        15.5,
        "Attention ×2",
        "4 heads, width 32\nqueries: candidates\nof batch t\nkeys: batches t−1 and t\n"
        "+ time-distance bias",
        learned=True,
    )
    box(
        ax,
        88,
        25.5,
        21,
        10,
        "Decision",
        "Pareto prior ± 4·tanh(·)\n→ p(discharge) ≥ threshold",
        learned=True,
    )
    box(
        ax,
        41,
        7,
        20,
        10,
        "Per-unit GRU",
        "state after batch t−1\n(mean, max, count, posture)",
        learned=True,
    )
    box(ax, 64, 7, 18, 10, "Batch t−1", "embedded candidates\nof the same unit")
    arrow(ax, [(20.6, 30.5), (24.4, 30.5)])
    arrow(ax, [(39.6, 30.5), (43.4, 30.5)])
    arrow(ax, [(58.6, 30.5), (62.4, 30.5)])
    arrow(ax, [(83.6, 30.5), (87.4, 30.5)])
    arrow(ax, [(51, 17.6), (51, 24.9)], label="γ, β", label_at=(52.5, 21.2), align="left")
    arrow(
        ax,
        [(32, 25.4), (32, 12), (40.4, 12)],
        dashed=True,
        label="summary of batch t\nupdates the state",
        label_at=(25, 17.5),
    )
    arrow(ax, [(73, 17.6), (73, 22.15)], label="keys", label_at=(74.5, 19.9), align="left")
    ax.text(
        1,
        0.3,
        "The output layer starts at zero, so an untrained head reproduces Pareto's decisions. "
        "Training uses DAgger: three rounds of closed-loop rollouts on 60 training records,\n"
        "labelled with the true discharges; threshold and strength are tuned on the "
        "validation subject.",
        fontsize=9.5,
        color=INK_2,
        va="bottom",
        linespacing=1.4,
    )


def decoder_figure(path):
    plt.rcParams["text.color"] = INK
    fig = plt.figure(figsize=(13.5, 10.2), facecolor=SURFACE)
    top = fig.add_axes([0.02, 0.52, 0.96, 0.44])
    bottom = fig.add_axes([0.02, 0.03, 0.96, 0.45])
    for ax in (top, bottom):
        ax.set_xlim(0, 110)
        ax.set_ylim(0, 46)
        ax.axis("off")
    closed_loop_panel(top)
    context_head_panel(bottom)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def seed_mean(arms, prefix, key):
    values = [v[key] for n, v in arms.items() if n.startswith(prefix) and v.get(key) is not None]
    return sum(values) / len(values) if values else None


def results_figure(path):
    analysis = json.loads((RESULTS / "analysis.json").read_text(encoding="utf-8"))
    heldout_path = RESULTS / "heldout_analysis.json"
    heldout = None
    if heldout_path.exists():
        heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    systems = [
        ("Pareto (original adaptive method)", "pareto"),
        ("L2: learned decisions + filter memory", "l2"),
        ("P2: context head, open loop", "p2"),
        ("P3: context head, closed loop", "cl"),
    ]
    metrics = [
        ("global_roa_pct", "All conditions"),
        ("ramp5s_roa_pct", "Fastest movement (5-s ramps)"),
        ("ge90_pct", "Units with RoA ≥ 90%"),
    ]
    rows = [("Validation subject (sub-04, 124 motor-unit instances)", analysis["arms"])]
    if heldout:
        heading = f"Held-out subject (sub-05, {heldout['instances']} instances, opened once)"
        rows.append((heading, heldout["arms"]))
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(13.5, 1.6 + 2.9 * len(rows)),
        sharey=True,
        squeeze=False,
        facecolor=SURFACE,
    )
    labels = [name for name, _ in systems]
    for r, (_heading, arms) in enumerate(rows):
        for c, (key, title) in enumerate(metrics):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            for i, (_, prefix) in enumerate(systems):
                if prefix in arms:
                    value = arms[prefix][key]
                else:
                    value = seed_mean(arms, prefix + "_seed", key)
                y = len(systems) - 1 - i
                if value is None:
                    ax.text(1.5, y, "not evaluated", va="center", fontsize=9, color=INK_2)
                    continue
                ax.barh(y, value, height=0.62, color=STAGES[i], edgecolor=SURFACE, linewidth=2)
                ax.text(value + 1.2, y, f"{value:.1f}%", va="center", fontsize=10, color=INK)
            ax.set_xlim(0, 112)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.tick_params(colors=INK_2, labelsize=9, length=0)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.set_title(title if r == 0 else "", fontsize=11, color=INK, loc="left", pad=8)
            if c == 0:
                ax.set_yticks(range(len(systems)))
                ax.set_yticklabels(labels[::-1], fontsize=10, color=INK)
    height = fig.get_figheight()
    fig.tight_layout(rect=(0, 0.3 / height, 1, 1 - 1.15 / height), h_pad=6.0)
    for r, (heading, _) in enumerate(rows):
        top = axes[r][0].get_position().y1
        fig.text(
            0.012,
            top + (0.4 if r == 0 else 0.15) / height,
            heading,
            fontsize=11.5,
            weight="bold",
            color=INK,
            va="bottom",
        )
    fig.text(
        0.012,
        axes[0][0].get_position().y1 + 0.8 / height,
        "Motor-unit decoding accuracy (rate of agreement, seed means of three trainings)",
        fontsize=13,
        weight="bold",
        color=INK,
        va="bottom",
    )
    fig.text(
        0.012,
        0.08 / height,
        "RoA = TP / (TP + FP + FN) against the simulated ground-truth "
        "discharges. Thresholds of every learned system were tuned on sub-04.",
        fontsize=9,
        color=INK_2,
    )
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    decoder_figure(IMAGES / "pareto_mc_decoder.png")
    results_figure(IMAGES / "pareto_mc_results.png")


if __name__ == "__main__":
    main()
