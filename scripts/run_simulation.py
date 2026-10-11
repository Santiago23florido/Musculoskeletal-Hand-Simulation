"""Advance the physical hand with a constant native excitation for a smoke check."""

import argparse
import json

import numpy as np

from hand_sim import load_model
from hand_sim.state import save_states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--excitation", type=float, default=0.02,
                        help="Constant excitation in native actuator units (hand range: 0 to 1)")
    parser.add_argument("--output", help="Optional .npz archive of initial and stepped states")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    with load_model(args.config) as sim:
        command = np.full(sim.model.nu, args.excitation)
        samples = [sim.get_state()] if args.output else []
        for _ in range(args.steps):
            state = sim.step(command)
            if args.output:
                samples.append(state)
        if args.output:
            save_states(args.output, samples, metadata={
                "provenance": sim.provenance,
                "purpose": "constant native excitation smoke check",
                "inventory": sim.get_model_inventory(),
            })
        print(json.dumps({
            "steps": args.steps, "time_seconds": float(sim.data.time),
            "step_seconds": sim.dt, "native_excitation": args.excitation,
            "qpos_shape": list(sim.data.qpos.shape),
            "activation_shape": list(sim.data.act.shape),
            "actuator_force_finite": bool(np.all(np.isfinite(sim.data.actuator_force))),
            "output": args.output,
        }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
