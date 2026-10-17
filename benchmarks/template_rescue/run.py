"""Reproducible CLI: register, select, predict, seal, score, report."""

import argparse

import torch

from .common import SUBJECTS, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "command", choices=["register", "select", "predict", "seal", "score", "report", "audit"]
    )
    parser.add_argument("--subject", choices=SUBJECTS[1:])
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    reference()
    from . import evaluate, experiment, report

    if args.command == "register":
        experiment.register()
    elif args.command == "select":
        experiment.select()
    elif args.command == "predict":
        evaluate.predict(args.subject)
    elif args.command == "seal":
        evaluate.seal_predictions()
    elif args.command == "score":
        evaluate.score()
    elif args.command == "report":
        report.report()
    else:
        from .audit import audit

        audit()


if __name__ == "__main__":
    main()
