import argparse

import torch

from .common import SUBJECTS, reference


def main():
    parser = argparse.ArgumentParser(description="True-event likelihood repair")
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "register",
            "train",
            "freeze",
            "predict",
            "seal",
            "start-scoring",
            "score",
            "report",
            "audit",
        ],
    )
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--seed", type=int, choices=[0, 1])
    args = parser.parse_args()
    torch.set_num_threads(2)
    reference()
    from . import data, evaluate, experiment, report

    if args.command == "prepare":
        data.prepare_subject(args.subject)
    elif args.command == "register":
        experiment.register()
    elif args.command == "train":
        if args.subject not in SUBJECTS[1:] or args.seed is None:
            parser.error("train requires subject and seed")
        experiment.train(args.subject, args.seed)
    elif args.command == "freeze":
        experiment.freeze()
    elif args.command == "predict":
        evaluate.predict_subject(args.subject)
    elif args.command == "seal":
        evaluate.seal()
    elif args.command == "start-scoring":
        evaluate.start_scoring()
    elif args.command == "score":
        evaluate.score(args.subject)
    elif args.command == "audit":
        from .audit import audit

        audit()
    else:
        report.report()


if __name__ == "__main__":
    main()
