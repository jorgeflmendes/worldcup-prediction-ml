"""Command-line entry point for the public workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worldcup2026",
        description="Probabilistic FIFA World Cup forecasting research",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Train, benchmark, and simulate the selected model",
    )
    run.add_argument("--simulations", type=int, default=100_000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from .research_benchmark import run_benchmark

    run_benchmark(args.simulations)


if __name__ == "__main__":
    main()
