"""Reuse completed immutable inputs in a fresh registered run, with a hash manifest."""

import argparse
import json
import os
from pathlib import Path

from .common import ROOT, now, sha256, write_new


def fork(origin, destination):
    origin, destination = Path(origin).resolve(), Path(destination).resolve()
    expected = (ROOT / "outputs/pareto_mc").resolve()
    if not origin.is_relative_to(expected) or not destination.is_relative_to(expected):
        raise ValueError("Both paths must be inside outputs/pareto_mc")
    if destination.exists():
        raise ValueError("The destination must be new")
    destination.mkdir(parents=True)
    folders = ["C1", "L0", "L1/open/features", "L1/all_feedback/round0/seed0"]
    if not (origin / "C1/seed0/complete.json").exists():
        raise ValueError("The reused observer must already be frozen")
    copied = {}
    for relative in folders:
        source_folder = origin / relative
        for source in source_folder.rglob("*"):
            if source.is_file():
                target = destination / source.relative_to(origin)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
                copied[source.relative_to(origin).as_posix()] = sha256(source)
    write_new(destination / "phase0.json", json.loads((origin / "phase0.json").read_text()))
    write_new(
        destination / "input_reuse.json",
        dict(
            created_at_utc=now(),
            origin=str(origin),
            destination=str(destination),
            reuse=(
                "read-only hard links of completed inputs/checkpoints; "
                "never rewrite linked artifacts"
            ),
            files=copied,
            rationale="lossless training cache; fixed four-thread numerical reference; attainable "
            "0.8 posterior feedback guard under the original bounded residual",
            previous_c4_validation_event_metrics_inspected=False,
        ),
    )
    print("Reused", len(copied), "immutable artifacts", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origin")
    parser.add_argument("destination")
    args = parser.parse_args()
    fork(args.origin, args.destination)
