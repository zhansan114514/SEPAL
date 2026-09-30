"""Aggregate the three final-round Actors with no Judge call."""

from __future__ import annotations

# ruff: noqa: E402

import argparse

from _utils import add_config_argument, load_config, setup_logging
from src.multi_acccollab.majority import aggregate_role_evaluations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config.run.seed)
    aggregate_role_evaluations(config)


if __name__ == "__main__":
    main()
