"""Run SFT, three independent original ACC-Collab pipelines, then majority vote."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import logging
import subprocess
import sys
import time
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
from src.multi_acccollab.config import MultiACCCollabPaths, resolve_config_path

logger = logging.getLogger(__name__)

STAGES = ("sft-data", "sft-train", "prepare-roles", "role-pipelines", "aggregate")
ACCCOLLAB_PIPELINE = PROJECT_ROOT / "scripts" / "acccollab" / "06_pipeline.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--only", nargs="*", choices=STAGES, default=[])
    parser.add_argument("--skip", nargs="*", choices=STAGES, default=[])
    parser.add_argument(
        "--force-role-stages",
        action="store_true",
        help="Re-enter original ACC-Collab role stages; valid checkpoints remain reusable.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    setup_logging(config.run.seed)
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or STAGES) - set(args.skip)
    config_path = str(resolve_config_path(args.config))

    for stage in STAGES:
        if stage not in selected:
            logger.info("Skipping multi-role stage %s", stage)
            continue
        logger.info("Starting multi-role stage %s", stage)
        if stage == "sft-data":
            _run_sft_data(config, config_path)
        elif stage == "sft-train":
            _run_sft_training(config, config_path)
        elif stage == "prepare-roles":
            _run_script("03_prepare_role_configs.py", config_path)
            _require_file(config.paths.role_manifest)
        elif stage == "role-pipelines":
            _run_role_pipelines(
                config,
                config_path=config_path,
                force=bool(args.force_role_stages),
            )
        elif stage == "aggregate":
            _run_script("04_aggregate_majority.py", config_path)
            _require_success(config.paths.majority_eval_dir / "_SUCCESS", "aggregate_majority")
        logger.info("Completed multi-role stage %s", stage)

    logger.info("Multi-role ACC-Collab pipeline complete")


def _run_sft_data(config, config_path: str) -> None:
    devices = config.runtime.sft_generation_devices
    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        for shard_idx, device in enumerate(devices):
            command = _script_command(
                "01_build_actor_sft_data.py",
                config_path,
                "--device",
                str(device),
                "--shard-idx",
                str(shard_idx),
                "--num-shards",
                str(len(devices)),
            )
            processes.append((f"sft-data-shard-{shard_idx}", _popen(command)))
        _wait_all(processes)
    except Exception:
        _terminate_all(processes)
        raise
    _run_script(
        "01_build_actor_sft_data.py",
        config_path,
        "--merge-shards",
        str(len(devices)),
    )
    _require_success(config.paths.sft_data_dir / "_SUCCESS", "actor_sft_data")


def _run_sft_training(config, config_path: str) -> None:
    if config.runtime.parallel_role_pipelines:
        jobs = [
            (
                f"sft-train-{role.name}",
                role.device,
                _script_command(
                    "02_train_actor_sft.py",
                    config_path,
                    "--role",
                    role.name,
                    "--device",
                    str(role.device),
                ),
            )
            for role in config.roles
        ]
        _run_conflict_free_device_batches(jobs)
    else:
        for role in config.roles:
            _run_script(
                "02_train_actor_sft.py",
                config_path,
                "--role",
                role.name,
                "--device",
                str(role.device),
            )
    _run_script("02_train_actor_sft.py", config_path, "--finalize-only")
    _require_file(config.paths.sft_registry)


def _run_role_pipelines(config, *, config_path: str, force: bool) -> None:
    # This stage is cheap and must always refresh role configs. Otherwise an
    # --only role-pipelines resume could silently use a stale manifest.
    _run_script("03_prepare_role_configs.py", config_path)
    _require_file(config.paths.role_manifest)
    jobs: list[tuple[str, int, list[str]]] = []
    for role in config.roles:
        role_config = config.paths.role_config(role.name)
        _require_file(role_config)
        command = [
            sys.executable,
            str(ACCCOLLAB_PIPELINE),
            "--config",
            str(role_config),
            "--devices",
            str(role.device),
            "--training-devices",
            str(role.device),
        ]
        if force:
            command.append("--force")
        if config.evaluation.policy_source_output_dir is not None:
            command.extend(("--only", "evaluate"))
        if config.runtime.parallel_role_pipelines:
            jobs.append((f"acccollab-{role.name}", role.device, command))
        else:
            _run(command)
    if jobs:
        _run_conflict_free_device_batches(jobs)
    source_paths = (
        MultiACCCollabPaths(Path(config.evaluation.policy_source_output_dir))
        if config.evaluation.policy_source_output_dir is not None
        else config.paths
    )
    for role in config.roles:
        output = config.paths.role_output(role.name)
        # An evaluation-only run writes metrics into ``output`` while loading
        # the authenticated policy registry from the training run. Requiring a
        # second registry in the evaluation directory incorrectly rejects a
        # successful cross-dataset evaluation.
        policy_output = source_paths.role_output(role.name)
        _require_file(policy_output / "registry" / "final.json")
        _require_success(output / "eval" / "_SUCCESS", "evaluate")


def _script_command(script: str, config_path: str, *extra: str) -> list[str]:
    return [sys.executable, str(SCRIPTS_DIR / script), "--config", config_path, *extra]


def _run_script(script: str, config_path: str, *extra: str) -> None:
    _run(_script_command(script, config_path, *extra))


def _run(command: list[str]) -> None:
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=offline_subprocess_env(),
        check=True,
    )


def _popen(command: list[str]) -> subprocess.Popen:
    return subprocess.Popen(command, cwd=PROJECT_ROOT, env=offline_subprocess_env())


def _wait_all(processes: list[tuple[str, subprocess.Popen]]) -> None:
    pending = {name: process for name, process in processes}
    while pending:
        for name, process in list(pending.items()):
            code = process.poll()
            if code is None:
                continue
            del pending[name]
            if code != 0:
                _terminate_all(list(pending.items()))
                raise RuntimeError(f"Parallel stage failed: {name}={code}")
        if pending:
            time.sleep(1.0)


def _run_conflict_free_device_batches(
    jobs: list[tuple[str, int, list[str]]],
) -> None:
    """Run as many role jobs as possible without sharing a GPU concurrently."""
    pending = list(jobs)
    while pending:
        used_devices: set[int] = set()
        batch: list[tuple[str, int, list[str]]] = []
        deferred: list[tuple[str, int, list[str]]] = []
        for job in pending:
            if job[1] in used_devices:
                deferred.append(job)
                continue
            used_devices.add(job[1])
            batch.append(job)
        processes = [(name, _popen(command)) for name, _device, command in batch]
        try:
            _wait_all(processes)
        except Exception:
            _terminate_all(processes)
            raise
        pending = deferred


def _terminate_all(processes: list[tuple[str, subprocess.Popen]]) -> None:
    for _name, process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(
        process.poll() is None for _name, process in processes
    ):
        time.sleep(0.1)
    for _name, process in processes:
        if process.poll() is None:
            process.kill()


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Expected pipeline artifact is missing: {path}")


def _require_success(path: Path, stage: str) -> None:
    _require_file(path)
    payload = read_json(path)
    if payload.get("status") != "complete" or payload.get("stage") != stage:
        raise RuntimeError(f"Invalid completion marker for {stage}: {path}")


if __name__ == "__main__":
    main()
