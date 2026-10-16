"""Prepare/audit, pre-register, fit, freeze, and finally score independent folds."""

import argparse

import torch

from .common import CONDITIONS, SUBJECTS, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--subjects", nargs="+", default=SUBJECTS)
    prepare.add_argument("--conditions", nargs="+", default=CONDITIONS)
    prepare.add_argument("--snrs", nargs="+", type=int, default=[30, 25, 20, 15])
    sub.add_parser("register")
    train = sub.add_parser("train")
    train.add_argument("--test", required=True)
    train.add_argument("--seed", type=int, required=True)
    sub.add_parser("freeze")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--test", required=True)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    reference()
    if args.command == "prepare":
        from .data import prepare as prepare_record

        for subject in args.subjects:
            for condition in args.conditions:
                for snr in args.snrs:
                    result = prepare_record(subject, condition, snr)
                    print(
                        subject,
                        condition,
                        snr,
                        "identity disagreements",
                        result["metadata"]["identity_spike_disagreements"],
                        flush=True,
                    )
    elif args.command == "register":
        from .experiment import register

        register()
    elif args.command == "train":
        from .experiment import train

        train(args.test, args.seed)
    elif args.command == "freeze":
        from .experiment import freeze

        freeze()
    elif args.command == "evaluate":
        from .evaluate import evaluate

        evaluate(args.test)
    elif args.command == "report":
        from .report import report

        report()


if __name__ == "__main__":
    main()
