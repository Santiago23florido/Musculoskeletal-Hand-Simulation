"""P3 orchestration only: the steps of closed_context.run_all, adapted to this laptop.

The registered module is unchanged; this driver calls its functions in the same order
(round rollouts -> three fits -> select, three rounds; final rollouts; analyse; latency).

Incidents (journal, 10-11 November 2026):
- three seeds started in parallel built the same new label cache at once
  (FileExistsError): labels of each round's new trajectories are built first;
- three parallel fits saturated GPU memory (~7.7/8 GB) and system commit (59/61.5 GB)
  next to the user's applications and stalled: seeds are trained one at a time
  (deterministic and independent, so the result does not depend on the order);
- completed, sealed rounds are not re-entered, so no idle process pools are spawned
  during training. Every P3 trajectory is still checked against its seal when the
  training reads it (closed_context.values_for).
"""

import os
import sys
import time

from benchmarks.pareto_mc import closed_context as cc
from benchmarks.pareto_mc.common import TRAIN, VALIDATION, records
from benchmarks.pareto_mc.experiment import checked
from benchmarks.pareto_mc.memory_experiment import read
from benchmarks.pareto_mc.train_likelihood import labelled


def prewarm(index):
    """Build (or verify) the labels of this round's new trajectories in this process."""
    for phase in ("training", "validation"):
        for path, case in cc.round_paths(index, phase):
            labelled(path, case, phase)
    print("P3 labels ready", index, cc.now(), flush=True)


def sequential_fits(index):
    waiting = [int(s) for s in os.environ.get("P3_WAIT_SEEDS", "").split(",") if s]
    for seed in waiting:  # already running as a separate process: only wait for it
        path = cc.FOLDER / f"round{index}/seed{seed}/complete.json"
        while not path.exists():
            time.sleep(60)
    for seed in cc.SEEDS:
        if (cc.FOLDER / f"round{index}/seed{seed}/complete.json").exists():
            continue
        print("P3 fit start", index, seed, cc.now(), flush=True)
        if cc.gpu("fit", str(index), str(seed)).wait():
            raise RuntimeError(f"Round {index} seed {seed} training failed")


def main(workers):
    cc.register_protocol()
    cases = records(TRAIN) + records(VALIDATION)
    for index in cc.ROUNDS:
        folder = cc.FOLDER / f"round{index}"
        if (folder / "selected.json").exists():
            continue
        checked(cc.FOLDER)
        if not (folder / "rollout/sealed.json").exists():
            cc.rollout(folder / "rollout", cases, cc.round_head(index), workers)
        prewarm(index)
        sequential_fits(index)
        if cc.gpu("select", str(index)).wait():
            raise RuntimeError(f"Round {index} selection failed")
        print("P3 round", index, read(folder / "selected.json")["selection"], cc.now())
    checked(cc.FOLDER)
    for seed, spec in cc.final_heads().items():
        cc.rollout(cc.FOLDER / f"final_validation/seed{seed}", records(VALIDATION), spec, workers)
    result = cc.analyse(workers)
    print("P3 decisions", result["decisions"], cc.now(), flush=True)
    print("P3 latency", cc.latency()["systems"], cc.now(), flush=True)
    return result


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
