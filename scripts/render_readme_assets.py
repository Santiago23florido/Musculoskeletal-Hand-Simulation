"""Render real model geometry and physical observations for the README figures."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from hand_sim import load_model
from hand_sim.visualization import hand_camera

BACKGROUND = "#080d14"
PANEL = "#101925"
TEXT = "#eaf2fb"
MUTED = "#8da4bd"
CYAN = "#55d9cf"
CORAL = "#ff7b80"


def render_anatomy(sim, destination: Path) -> None:
    # Rendering-only options: no biomechanical arrays or native physics are changed.
    sim.model.vis.headlight.ambient[:] = 0.5
    sim.model.vis.headlight.diffuse[:] = 0.8
    sim.model.vis.global_.offwidth = 1200
    sim.model.vis.global_.offheight = 800
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.tendongroup[:] = 1
    camera = hand_camera(sim)
    frames = []
    with mujoco.Renderer(sim.model, height=800, width=1200) as renderer:
        for azimuth, elevation in ((140, -25), (320, 20)):
            camera.azimuth, camera.elevation = azimuth, elevation
            renderer.update_scene(sim.data, camera=camera, scene_option=options)
            frames.append(renderer.render().copy())

    fig = plt.figure(figsize=(16, 9), facecolor=BACKGROUND)
    fig.text(
        0.05,
        0.94,
        "MYOSUITE + MUJOCO  /  INDEPENDENT RESEARCH INFRASTRUCTURE",
        fontsize=11,
        color=CYAN,
        weight="bold",
    )
    fig.text(0.05, 0.865, "Musculoskeletal Hand Simulation", fontsize=32, color=TEXT, weight="bold")
    fig.text(
        0.05,
        0.812,
        "A physical foundation for movement-robust MUAP identification research",
        fontsize=15,
        color=MUTED,
    )
    for bounds, frame, label in (
        ([0.02, 0.25, 0.57, 0.52], frames[0], "NATIVE BONE GEOMETRY"),
        ([0.61, 0.30, 0.36, 0.40], frames[1], "MUSCLE-TENDON PATHS"),
    ):
        ax = fig.add_axes(bounds)
        ax.imshow(frame)
        ax.axis("off")
        ax.set_title(label, color=MUTED, fontsize=10, loc="left", pad=12)
    for x, value, label in (
        (0.07, "23", "JOINT DOFs"),
        (0.30, "39", "MUSCLE ACTUATORS"),
        (0.57, "39", "TENDONS"),
        (0.80, "5", "FINGERTIPS"),
    ):
        fig.text(x, 0.18, value, fontsize=30, weight="bold", color=CYAN)
        fig.text(x, 0.135, label, fontsize=10, color=MUTED)
    fig.text(
        0.05,
        0.055,
        "Official MyoHand model: MyoSuite 3.0.0 / myo-sim 0.2.3",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.05,
        0.025,
        "Actual MuJoCo renders. Current scope: physical simulation and observation.",
        fontsize=9,
        color=MUTED,
    )
    fig.savefig(destination, dpi=120, facecolor=BACKGROUND)
    plt.close(fig)


def render_observations(sim, destination: Path) -> None:
    sim.reset()
    history = [sim.get_state()]
    command = np.full(sim.model.nu, 0.02)
    for _ in range(200):
        history.append(sim.step(command))
    times = np.array([state["time"] for state in history])
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), facecolor=BACKGROUND)
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.07, right=0.97, hspace=0.65, wspace=0.25)
    fig.text(0.05, 0.94, "PHYSICAL OBSERVATIONS", fontsize=11, color=CYAN, weight="bold")
    fig.text(
        0.05,
        0.875,
        "Inspect the mechanics before defining the algorithms",
        fontsize=26,
        color=TEXT,
        weight="bold",
    )
    fig.text(
        0.05,
        0.825,
        "200 native CPU steps | 0.4 seconds | constant excitation = 0.02",
        fontsize=13,
        color=MUTED,
    )
    for ax in axes.flat:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#243549")
        ax.grid(color="#27384a", alpha=0.5, linewidth=0.6)
        ax.set_xlabel("Simulation time [s]", color=MUTED, fontsize=10)
        ax.set_xlim(times[0], times[-1])

    muscle = sim.get_muscle_state("FDS2_r")
    activation_index = muscle["activation_indices"][0]
    axes[0, 0].plot(
        times,
        [s["excitation"][muscle["id"]] for s in history],
        color=CORAL,
        linestyle="--",
        label="FDS2_r excitation",
    )
    axes[0, 0].plot(
        times,
        [s["activation"][activation_index] for s in history],
        color=CYAN,
        label="FDS2_r activation",
    )
    axes[0, 0].set_ylabel("Dimensionless", color=MUTED, fontsize=10)
    axes[0, 0].set_title("Native commands and activation dynamics", color=TEXT, loc="left", pad=10)

    for name, color in (("FDS2_r", CYAN), ("EDC2_r", CORAL)):
        i = sim.identifiers.resolve("actuator", name)
        axes[0, 1].plot(times, [s["actuator_force"][i] for s in history], color=color, label=name)
    axes[0, 1].set_ylabel("Signed actuator force [N]", color=MUTED, fontsize=10)
    axes[0, 1].set_title("Muscle-tendon actuator forces", color=TEXT, loc="left", pad=10)

    for name, color in (("mcp2_flexion_r", CYAN), ("mcp3_flexion_r", CORAL)):
        index = sim.get_joint_state(name)["qpos_indices"][0]
        axes[1, 0].plot(times, [s["qpos"][index] for s in history], color=color, label=name)
    axes[1, 0].set_ylabel("Generalized position [rad]", color=MUTED, fontsize=10)
    axes[1, 0].set_title("Joint coordinates from model metadata", color=TEXT, loc="left", pad=10)

    for name, color in (("IFtip_r", CYAN), ("THtip_r", CORAL)):
        i = sim.identifiers.resolve("site", name)
        positions = np.stack([s["site_position_world"][i] for s in history])
        displacement = np.linalg.norm(positions - positions[0], axis=1) * 1000
        axes[1, 1].plot(times, displacement, color=color, label=name)
    axes[1, 1].set_ylabel("Displacement from reset [mm]", color=MUTED, fontsize=10)
    axes[1, 1].set_title("Named fingertip landmarks", color=TEXT, loc="left", pad=10)
    for ax in axes.flat:
        ax.legend(facecolor=PANEL, edgecolor="none", labelcolor=TEXT, fontsize=9)
    fig.text(
        0.05,
        0.035,
        "Actual model outputs from a smoke check. No EMG or MUAP signal is generated.",
        fontsize=11,
        color=MUTED,
    )
    fig.savefig(destination, dpi=120, facecolor=BACKGROUND)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/images"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with load_model(seed=0) as sim:
        render_anatomy(sim, args.output_dir / "myohand-overview.png")
        render_observations(sim, args.output_dir / "physical-observations.png")
    print(f"Real model figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
