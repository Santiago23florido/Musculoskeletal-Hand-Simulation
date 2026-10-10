"""Inspect native anatomical names, physical states, and available measurements."""

import argparse
import json

from hand_sim import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--output", help="Export the complete inventory to this JSON path")
    parser.add_argument("--list", choices=("all", "bodies", "joints", "actuators", "muscles",
                                         "tendons", "sites", "sensors", "fingertips"))
    for kind in ("joint", "muscle", "tendon", "site", "body", "sensor"):
        parser.add_argument(f"--{kind}", help=f"Query an existing {kind} by its exact native name")
    args = parser.parse_args()
    with load_model(args.config) as sim:
        inventory = sim.get_model_inventory()
        result = {
            "dimensions": inventory["dimensions"],
            "provenance": inventory["provenance"],
            "measurements": sim.sensors.capabilities(),
            "fingertips": sim.sensors.fingertips(),
        }
        if args.list:
            result["inventory"] = inventory if args.list == "all" else inventory[args.list]
        queries = {
            "joint": sim.get_joint_state, "muscle": sim.get_muscle_state,
            "tendon": sim.get_tendon_state, "site": sim.get_site_position,
            "body": sim.get_body_state, "sensor": sim.get_sensor_data,
        }
        for kind, query in queries.items():
            name = getattr(args, kind)
            if name is not None:
                result[kind] = query(name)
        if args.output:
            sim.identifiers.export(args.output, metadata=sim.provenance)
            result["exported_inventory"] = args.output
        print(json.dumps(result, indent=2, allow_nan=False, default=lambda value: value.tolist()))


if __name__ == "__main__":
    main()
