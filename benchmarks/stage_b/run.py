"""Stage B CLI with separate input, training, prediction and scoring phases."""

import argparse

import torch

from .common import SUBJECTS, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "targets",
            "register",
            "train",
            "freeze",
            "predict",
            "seal",
            "score",
            "report",
            "audit",
        ],
    )
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--seed", type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    reference()
    from . import data, evaluate, experiment, report

    if args.command == "prepare":
        data.prepare_subject(args.subject)
    elif args.command == "targets":
        data.prepare_targets(args.subject)
    elif args.command == "register":
        experiment.register()
    elif args.command == "train":
        if args.subject not in SUBJECTS[1:] or args.seed is None:
            parser.error("train requires an outer subject and seed")
        experiment.train(args.subject, args.seed)
    elif args.command == "freeze":
        experiment.freeze()
    elif args.command == "predict":
        evaluate.predict_subject(args.subject)
    elif args.command == "seal":
        evaluate.seal()
    elif args.command == "score":
        evaluate.score()
    elif args.command == "report":
        report.report()
    else:
        from .audit import audit

        audit()


if __name__ == "__main__":
    main()
