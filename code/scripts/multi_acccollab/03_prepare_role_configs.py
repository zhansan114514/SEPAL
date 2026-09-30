"""Materialize the three exact original ACC-Collab role configs."""

from __future__ import annotations

# ruff: noqa: E402

import argparse

from _utils import add_config_argument, load_config, setup_logging
from src.multi_acccollab.config import write_role_configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config.run.seed)
    write_role_configs(config)


if __name__ == "__main__":
    main()

