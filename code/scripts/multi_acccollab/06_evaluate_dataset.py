"""Evaluate all three trained roles on one dataset, then aggregate without Judge."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from _utils import (
    PROJECT_ROOT,
    SCRIPTS_DIR,
    add_config_argument,
    load_config,
    offline_subprocess_env,
    setup_logging,
)
from src.acccollab.io import read_json
from src.multi_acccollab.config import MultiACCCollabPaths

logger = logging.getLogger(__name__)
ACCCOLLAB_PIPELINE = PROJECT_ROOT / "scripts" / "acccollab" / "06_pipeline.py"
PREPARE_ROLES = SCRIPTS_DIR / "03_prepare_role_configs.py"
AGGREGATE = SCRIPTS_DIR / "04_aggregate_majority.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument(
        "--devices",
        required=True,
        help="Comma-separated physical GPU ids assigned exclusively to this dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and print the role/shard schedule without writing outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    devices = parse_devices(args.devices)
    setup_logging(config.run.seed)
    logger.info(
        "Dataset evaluation plan: dataset=%s roles=%s devices=%s shards_per_role=%d",
        config.base_config().data.dataset,
        [role.name for role in config.roles],
        devices,
        len(devices),
    )
    if args.dry_run:
        return

    _run(
        [
            sys.executable,
            str(PREPARE_ROLES),
            "--config",
            str(args.config),
        ]
    )
    source_paths = (
        MultiACCCollabPaths(Path(config.evaluation.policy_source_output_dir))
        if config.evaluation.policy_source_output_dir is not None
        else config.paths
    )
    device_arg = ",".join(str(device) for device in devices)
    for role in config.roles:
        source_registry = source_paths.role_output(role.name) / "registry" / "final.json"
        _require_file(source_registry)
        role_config = config.paths.role_config(role.name)
        _require_file(role_config)
        logger.info(
            "Evaluating dataset=%s role=%s with %d shard(s) on GPU(s) %s",
            config.base_config().data.dataset,
            role.name,
            len(devices),
            device_arg,
        )
        _run(
            [
                sys.executable,
                str(ACCCOLLAB_PIPELINE),
                "--config",
                str(role_config),
                "--devices",
                device_arg,
                "--training-devices",
                str(devices[0]),
                "--only",
                "evaluate",
            ]
        )
        _require_success(
            config.paths.role_output(role.name) / "eval" / "_SUCCESS",
            "evaluate",
        )

    _run(
        [
            sys.executable,
            str(AGGREGATE),
            "--config",
            str(args.config),
        ]
    )
    _require_success(config.paths.majority_eval_dir / "_SUCCESS", "aggregate_majority")
    logger.info(
        "Completed dataset evaluation and majority aggregation: %s",
        config.base_config().data.dataset,
    )


def parse_devices(raw: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --devices value: {raw!r}") from exc
    if not devices or any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative GPU ids")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")
    return devices


def _run(command: list[str]) -> None:
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=offline_subprocess_env(),
        check=True,
    )


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required formal-evaluation artifact is missing: {path}")


def _require_success(path: Path, stage: str) -> None:
    _require_file(path)
    marker = read_json(path)
    if marker.get("status") != "complete" or marker.get("stage") != stage:
        raise RuntimeError(f"Invalid completion marker for {stage}: {path}")


if __name__ == "__main__":
    main()
