"""Run each preregistered Pareto-MC stage explicitly."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["identity"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "identity":
        from .identity import run

        if not run()["passed"]:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
