"""Stage A experiment commands. Run from the repository root with python -m."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import DATA, OUTPUT, reference, stub, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=6)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    train = sub.add_parser("train")
    train.add_argument("name")
    train.add_argument("--epochs", type=int, default=6)
    train.add_argument("--samples", type=int, default=2000)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--contrast-weight", type=float, default=0.01)
    train.add_argument("--whitening-weight", type=float, default=1.0)
    train.add_argument("--validation-whitening-weight", type=float, default=0.0001)
    train.add_argument("--validation-contrast-weight", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--state-dim", type=int, default=1)
    train.add_argument("--encoder", choices=["gru", "mlp"], default="gru")
    train.add_argument("--resume", type=Path)
    train.add_argument("--loss-mode", choices=["delta", "squared"], default="squared")
    select = sub.add_parser("select")
    select.add_argument("runs", nargs="+")
    merge = sub.add_parser("merge")
    merge.add_argument("evaluations", nargs="+")
    audit = sub.add_parser("audit")
    audit.add_argument("--condition", default="triangular-ramp40s")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--name", required=True)
    evaluate.add_argument("--subjects", nargs="+")
    evaluate.add_argument("--conditions", nargs="+")
    evaluate.add_argument("--snrs", type=int, nargs="+")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    reference()
    if args.command == "prepare":
        from .train import prepare_pool

        for record in prepare_pool():
            print(json.dumps(record["metadata"]), flush=True)
    elif args.command == "train":
        from .train import train_run

        train_run(
            args.name,
            epochs=args.epochs,
            samples=args.samples,
            learning_rate=args.learning_rate,
            contrast_weight=args.contrast_weight,
            seed=args.seed,
            state_dim=args.state_dim,
            encoder=args.encoder,
            resume=args.resume,
            loss_mode=args.loss_mode,
            whitening_weight=args.whitening_weight,
            validation_whitening_weight=args.validation_whitening_weight,
            validation_contrast_weight=args.validation_contrast_weight,
        )
    elif args.command == "select":
        from .selection import select_model

        select_model(args.runs)
    elif args.command == "merge":
        from .merge import merge_evaluations

        merge_evaluations(args.evaluations)
    elif args.command == "audit":
        from adapt_decomp import AdaptationResult
        from adapt_decomp.spikes import rate_of_agreement_paired
        from adapt_decomp.utils import load_gt

        from .data import prepare
        from .inference import predict

        record = prepare("sub-01", args.condition, 30, persist=args.condition != "staircase")
        binary, timings, _ = predict(record, adapt_sv=False, adapt_sd=False)
        name = stub("sub-01", args.condition, 30)
        path = DATA / "outputs/adaptation/sub-01/fixed" / f"{name}_adapt_fixed.pkl"
        cached = AdaptationResult.load(path)
        truth = load_gt(
            DATA / "data/sub-01/clean" / f"sub-01_FDSI_{args.condition}_spikes.npz",
            n_samples=len(binary),
        )[:, record["calibration"].gt_matched_indices]
        roa, _, _ = rate_of_agreement_paired(truth, binary, fs=2048, tol_spike_ms=2)
        result = dict(
            spike_bit_agreement=float(np.mean(binary == cached.spikes.numpy())),
            spike_disagreements=int(np.count_nonzero(binary != cached.spikes.numpy())),
            max_roa_difference=float(np.max(np.abs(roa - cached.roa))),
            timing=timings,
        )
        write_json(OUTPUT / "audit.json", result)
        print(json.dumps(result, indent=2))
        if result["max_roa_difference"] > 1e-10:
            raise SystemExit("Identity adapter does not reproduce frozen cache")
    else:
        from .evaluate import evaluate as run_evaluation

        run_evaluation(
            args.checkpoint,
            args.name,
            subjects=args.subjects,
            conditions=args.conditions,
            snrs=args.snrs,
        )


if __name__ == "__main__":
    main()
