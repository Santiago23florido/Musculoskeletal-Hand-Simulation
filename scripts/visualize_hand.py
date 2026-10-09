"""Open the original hand in the standard MuJoCo viewer."""

import argparse

from hand_sim import load_model
from hand_sim.visualization import launch_viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--duration", type=float, help="Close after this many wall-clock seconds")
    args = parser.parse_args()
    with load_model(args.config) as sim:
        print("Native hand viewer. Drag to rotate; scroll to zoom; close window to stop.")
        print("Physical variables and names: uv run python scripts/inspect_model.py")
        steps = launch_viewer(sim, duration=args.duration)
        print(f"Viewer closed after {steps} steps; simulation time {sim.data.time:.6f} s")


if __name__ == "__main__":
    main()
