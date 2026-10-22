"""Run each preregistered Pareto-MC stage explicitly."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["identity", "identity-head", "train-observer", "l1"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "identity":
        from .identity import run

        if not run()["passed"]:
            raise SystemExit(3)
    elif args.command == "identity-head":
        from .identity_head import run

        run()
    elif args.command == "train-observer":
        from .observer import train

        train()
    elif args.command == "l1":
        from .experiment import run_l1

        if not run_l1()["passed"]:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
