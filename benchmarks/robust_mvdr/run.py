"""Robust analytic decoding CLI: audits, registration, prediction, sealing and scoring."""

import argparse

import torch

from .common import SUBJECTS, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "command",
        choices=[
            "phase-audit",
            "target-reliability",
            "register",
            "predict",
            "seal",
            "score",
            "report",
        ],
    )
    parser.add_argument("--subject", choices=SUBJECTS[1:])
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    reference()
    from . import audits, evaluate, report

    if args.command == "phase-audit":
        audits.phase_audit()
    elif args.command == "target-reliability":
        audits.target_reliability()
    elif args.command == "register":
        evaluate.register()
    elif args.command == "predict":
        if args.subject is None:
            parser.error("predict requires --subject")
        evaluate.predict_subject(args.subject)
    elif args.command == "seal":
        evaluate.seal()
    elif args.command == "score":
        evaluate.score([args.subject] if args.subject else None)
    else:
        report.report()


if __name__ == "__main__":
    main()
