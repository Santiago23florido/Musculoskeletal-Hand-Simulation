"""L3 entry point with explicit provenance from the retained L2 checkpoints."""

import argparse
import os

import torch

from .common import OUTPUT, TRAIN, VALIDATION, now, records, register, sha256, stub, write_new
from .experiment import checked, config_hash
from .memory_experiment import SEEDS, STRUCTURES, TRIALS, read, run_variant, seed_selection


def specification(previous, *, conservative, workers):
    classifiers = {
        seed: seed_selection(previous["selection"], seed)["checkpoint_sha256"] for seed in SEEDS
    }
    return dict(
        level="L3",
        reference=previous,
        cases=records(TRAIN) + records(VALIDATION),
        memory="per-record effective filters and centroids; immutable calibration snapshot",
        insertion="rate/calibration rate, centroid gap/calibration gap and IQR fraction confidence",
        retrieval="past bins only; current blind C1 mean; uncertainty-attenuated blending",
        structures=STRUCTURES,
        tpe_trials=TRIALS,
        tpe_execution=(
            "ask/tell groups of nine; identical raw preprocessing, whitening and C1 shared"
        ),
        continuous="threshold [.5,.95], kappa [.02,.3], beta_max [.1,.8], momentum [.8,.99]",
        conservative=conservative,
        structural_alternative="multiply beta by squared local rate/gap/IQR confidence",
        iterations_maximum=2,
        search_classifier="retained L2 round3 seeds 0/1/2, with retained decoding parameters",
        classifier_sha256=classifiers,
        observer_sha256=sha256(OUTPUT / "C1/seed0/observer.pt"),
        active_identity_sha256=sha256(OUTPUT / "C2_full_memory_identity_v1/result.json"),
        repeats="top three settings with all three trained seeds; choose the median objective",
        objective=".5 global RoA + .5 ramp5s RoA; -5*precision loss beyond 1pp vs Pareto",
        dagger="three new rounds; aggregate sealed L1, retained L2 and preceding L3 trajectories",
        c4_training="GPU AdamW from identity; three seeds; BCE checkpoint and validation Jaccard",
        final_evaluation="ordinary complete native CPU loop; seal all predictions before scoring",
        gate_global_min=previous["actual"]["global_roa_pct"] + 1,
        gate_cycles="ramp5s cycles2-8 >= current cycle1 + 5pp",
        limiter="post-QR memory residual <=kappa relative to that batch's Pareto proposal",
        centroids="saved with effective filters; live centroids continue gradual L1 EMA",
        workers=workers,
        torch_threads=4,
        openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        sub05="never opened in development",
        scope="exploratory validation only; no new confirmatory data or external evaluation",
        input_config_sha256={
            stub(*case): config_hash(case) for case in records(TRAIN) + records(VALIDATION)
        },
    )


def run(workers):
    previous = read(OUTPUT / "L2/retained.json")
    identity = read(OUTPUT / "C2_full_memory_identity_v1/result.json")
    if not previous["passed"] or not identity["passed"] or not identity["active"]:
        raise ValueError("Retained L2 and real active full-memory identity must pass before L3")
    checked(OUTPUT / previous["run_root"])
    root = OUTPUT / "L3"
    if (root / "retained.json").exists():
        return read(root / "retained.json")
    for conservative in (False, True):
        folder = root / ("confidence_gated" if conservative else "standard")
        if not (folder / "protocol.json").exists():
            register(folder, specification(previous, conservative=conservative, workers=workers))
        result = run_variant(
            folder,
            previous,
            calibration_only=False,
            conservative=conservative,
            workers=workers,
        )
        if result["passed"]:
            write_new(root / "retained.json", result)
            return result
        if not conservative and not (folder / "failure_analysis.json").exists():
            write_new(
                folder / "failure_analysis.json",
                dict(
                    analysed_at_utc=now(),
                    failed_gates=result["gates"],
                    metrics=result["actual"],
                    hypothesis="returning filters can reinforce poorly supported local decisions",
                    single_change="multiply memory beta by squared local rate/gap/IQR confidence",
                    decision="one further registered structure, then stop if it fails",
                ),
            )
    result = dict(
        completed_at_utc=now(),
        passed=False,
        retained_level="L2",
        reason="Both L3 structures failed the preregistered gates; stop before L4",
    )
    if not (root / "stopped.json").exists():
        write_new(root / "stopped.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.workers)
