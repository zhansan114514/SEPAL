"""Run formal training with phase-aware GPU use and the fixed evaluation schedule."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import datetime as dt
import logging
import subprocess
import sys
import time
from pathlib import Path

from _utils import (
    PROJECT_ROOT,
    SCRIPTS_DIR,
    load_config,
    offline_subprocess_env,
    setup_logging,
)
from src.acccollab.io import read_json, write_json
from src.multi_acccollab.config import config_snapshot
from src.utils.artifacts import file_sha256, stable_fingerprint

logger = logging.getLogger(__name__)
PRIMARY_CONFIG = "configs/multi_acccollab/llama3_8b_instruct_mmlu_sft10k.yaml"
BOOLQ_CONFIG = "configs/multi_acccollab/llama3_8b_instruct_eval_boolq.yaml"
SCIQ_CONFIG = "configs/multi_acccollab/llama3_8b_instruct_eval_sciq.yaml"
STAGES = ("sft", "role-training", "cross-eval", "mmlu-eval")
MULTI_PIPELINE = SCRIPTS_DIR / "05_pipeline.py"
PREPARE_ROLES = SCRIPTS_DIR / "03_prepare_role_configs.py"
DATASET_EVALUATION = SCRIPTS_DIR / "06_evaluate_dataset.py"
ACCCOLLAB_PIPELINE = PROJECT_ROOT / "scripts" / "acccollab" / "06_pipeline.py"
DEFAULT_DEVICES = "0,1,2,3"
ROLE_TRAINING_STAGES = (
    "critic-data",
    "critic-train",
    "actor-data",
    "actor-train",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=PRIMARY_CONFIG)
    parser.add_argument("--boolq-config", default=BOOLQ_CONFIG)
    parser.add_argument("--sciq-config", default=SCIQ_CONFIG)
    parser.add_argument(
        "--devices",
        default=DEFAULT_DEVICES,
        help="Comma-separated physical GPU ids reserved exclusively for this run.",
    )
    parser.add_argument("--only", nargs="*", choices=STAGES, default=[])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs, GPU visibility, and the complete schedule without writing outputs.",
    )
    parser.add_argument(
        "--reuse-existing-role-configs",
        action="store_true",
        help=(
            "Recovery-only: preserve authenticated role configs so durable generation "
            "checkpoints remain reusable after an operational code fix."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    primary = load_config(args.config)
    boolq = load_config(args.boolq_config)
    sciq = load_config(args.sciq_config)
    devices = _parse_devices(args.devices)
    setup_logging(primary.run.seed)
    _validate_plan(primary, boolq, sciq)
    visible_gpus = _visible_gpu_ids()
    if not set(devices).issubset(visible_gpus):
        raise RuntimeError(
            f"Formal experiment requires GPUs {list(devices)}; visible={visible_gpus}"
        )
    role_devices = {role.device for role in primary.roles}
    if not role_devices.issubset(set(devices)):
        raise RuntimeError(
            f"Role devices {sorted(role_devices)} are outside reserved GPUs {list(devices)}"
        )
    selected = set(args.only or STAGES)
    plan = _formal_plan(args, primary, boolq, sciq, devices)
    logger.info("Validated formal experiment schedule: %s", plan["gpu_schedule"])
    if args.dry_run:
        logger.info("Dry run complete; no experiment outputs were written")
        return

    marker_dir = primary.paths.output_dir / "formal_orchestration"
    marker_dir.mkdir(parents=True, exist_ok=True)
    write_json(marker_dir / "plan.json", plan)

    for stage in STAGES:
        if stage not in selected:
            logger.info("Skipping formal stage %s", stage)
            continue
        started = time.monotonic()
        logger.info("Starting formal stage %s", stage)
        if stage == "sft":
            _run(
                [
                    sys.executable,
                    str(MULTI_PIPELINE),
                    "--config",
                    str(args.config),
                    "--only",
                    "sft-data",
                    "sft-train",
                    "prepare-roles",
                ]
            )
        elif stage == "role-training":
            _run_role_training(
                primary,
                str(args.config),
                devices,
                reuse_existing_role_configs=args.reuse_existing_role_configs,
            )
        elif stage == "cross-eval":
            _run_cross_evaluations(
                str(args.boolq_config),
                str(args.sciq_config),
                devices,
            )
            _require_success(boolq.paths.majority_eval_dir / "_SUCCESS", "aggregate_majority")
            _require_success(sciq.paths.majority_eval_dir / "_SUCCESS", "aggregate_majority")
        elif stage == "mmlu-eval":
            _run(
                [
                    sys.executable,
                    str(DATASET_EVALUATION),
                    "--config",
                    str(args.config),
                    "--devices",
                    _device_arg(devices),
                ]
            )
            _require_success(
                primary.paths.majority_eval_dir / "_SUCCESS",
                "aggregate_majority",
            )
        elapsed = time.monotonic() - started
        write_json(
            marker_dir / f"{stage}.json",
            {
                "schema_version": 1,
                "pipeline": "multi_acccollab_formal",
                "stage": stage,
                "status": "complete",
                "elapsed_seconds": elapsed,
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "plan_fingerprint": plan["fingerprint"],
            },
        )
        logger.info("Completed formal stage %s in %.1f seconds", stage, elapsed)

    if all(_formal_stage_complete(marker_dir / f"{stage}.json", stage) for stage in STAGES):
        write_json(
            marker_dir / "_SUCCESS",
            {
                "schema_version": 1,
                "pipeline": "multi_acccollab_formal",
                "stage": "formal_experiment",
                "status": "complete",
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "plan_fingerprint": plan["fingerprint"],
            },
        )
        logger.info("Formal multi-role ACC-Collab experiment complete")


def _formal_stage_complete(path: Path, stage: str) -> bool:
    if not path.is_file():
        return False
    payload = read_json(path)
    return (
        payload.get("pipeline") == "multi_acccollab_formal"
        and payload.get("stage") == stage
        and payload.get("status") == "complete"
    )


def _validate_plan(primary, boolq, sciq) -> None:
    if primary.profile != "primary" or primary.base_config().data.dataset != "mmlu":
        raise ValueError("Primary formal config must use profile=primary and MMLU")
    if primary.base_config().method.alternating_iterations != 1:
        raise ValueError("The phase-aware formal scheduler requires exactly one alternation")
    expected = ((boolq, "boolq", 3270), (sciq, "sciq", 1000))
    primary_output = primary.paths.output_dir.resolve(strict=False)
    outputs = {primary_output}
    for config, dataset, sample_count in expected:
        base = config.base_config()
        if config.profile != "cross_eval" or base.data.dataset != dataset:
            raise ValueError(f"Formal {dataset} config must use profile=cross_eval")
        if base.data.eval.expected_samples != sample_count or base.evaluation.trials != 1:
            raise ValueError(f"Formal {dataset} coverage/trials are not locked")
        source = Path(str(config.evaluation.policy_source_output_dir)).resolve(strict=False)
        if source != primary_output:
            raise ValueError(f"Formal {dataset} policies do not point to the primary run")
        output = config.paths.output_dir.resolve(strict=False)
        if output in outputs:
            raise ValueError("Formal output directories must be distinct")
        outputs.add(output)
        if config.evaluation.use_judge:
            raise ValueError("Formal evaluation must not use Judge")


def _formal_plan(args, primary, boolq, sciq, devices: tuple[int, ...]) -> dict[str, object]:
    config_paths = [Path(args.config), Path(args.boolq_config), Path(args.sciq_config)]
    if len(devices) == 1:
        cross_eval_phase = "phase_1_sequential"
        boolq_devices = sciq_devices = devices
    else:
        cross_eval_phase = "phase_1_concurrent"
        boolq_devices, sciq_devices = _two_workload_split(devices, 3270, 1000)
    payload: dict[str, object] = {
        "schema_version": 1,
        "pipeline": "multi_acccollab_formal",
        "primary_output": str(primary.paths.output_dir),
        "evaluation_outputs": {
            "boolq": str(boolq.paths.output_dir),
            "sciq": str(sciq.paths.output_dir),
            "mmlu": str(primary.paths.output_dir),
        },
        "coverage": {
            "boolq": {"split": "validation", "samples": 3270, "trials": 1},
            "sciq": {"split": "test", "samples": 1000, "trials": 1},
            "mmlu": {"split": "test", "samples": 14042, "trials": 1},
        },
        "gpu_schedule": {
            "role_training": {
                "critic_data": {
                    "execution": "roles_sequential",
                    "devices": list(devices),
                    "shards_per_role": len(devices),
                },
                "critic_train": {
                    "execution": "roles_parallel",
                    "devices_by_role": {
                        role.name: role.device for role in primary.roles
                    },
                },
                "actor_data": {
                    "execution": "roles_sequential",
                    "devices": list(devices),
                    "shards_per_role": len(devices),
                },
                "actor_train": {
                    "execution": "roles_parallel",
                    "devices_by_role": {
                        role.name: role.device for role in primary.roles
                    },
                },
            },
            cross_eval_phase: {
                "boolq": {
                    "devices": list(boolq_devices),
                    "shards_per_role": len(boolq_devices),
                },
                "sciq": {
                    "devices": list(sciq_devices),
                    "shards_per_role": len(sciq_devices),
                },
            },
            "phase_2_after_both_complete": {
                "mmlu": {"devices": list(devices), "shards_per_role": len(devices)}
            },
        },
        "uses_judge": False,
        "reuse_existing_role_configs": bool(args.reuse_existing_role_configs),
        "config_files": {
            str(path): file_sha256(path) for path in config_paths
        },
        "config_fingerprints": {
            "primary": stable_fingerprint(config_snapshot(primary)),
            "boolq": stable_fingerprint(config_snapshot(boolq)),
            "sciq": stable_fingerprint(config_snapshot(sciq)),
        },
    }
    payload["fingerprint"] = stable_fingerprint(payload)
    return payload


def _run_role_training(
    primary,
    config_path: str,
    devices: tuple[int, ...],
    *,
    reuse_existing_role_configs: bool = False,
) -> None:
    if reuse_existing_role_configs:
        _validate_existing_role_configs(primary)
        logger.warning(
            "Recovery mode is reusing existing authenticated role configs; "
            "role configs will not be regenerated"
        )
    else:
        _run(
            [
                sys.executable,
                str(PREPARE_ROLES),
                "--config",
                config_path,
            ]
        )
    role_configs = []
    for role in primary.roles:
        role_config = primary.paths.role_config(role.name)
        _require_file(role_config)
        role_configs.append((role, role_config))

    # Generation is the dominant cost. Give each role every reserved GPU in
    # turn. DPO training remains single-GPU and runs in conflict-free batches.
    for selector in ROLE_TRAINING_STAGES:
        if selector.endswith("-data"):
            for role, role_config in role_configs:
                logger.info(
                    "Running %s for role %s on %d generation shards",
                    selector,
                    role.name,
                    len(devices),
                )
                _run(_role_stage_command(role, role_config, selector, devices))
            continue

        jobs = [
            (
                role.name,
                role.device,
                _role_stage_command(role, role_config, selector, devices),
            )
            for role, role_config in role_configs
        ]
        _run_conflict_free_device_batches(jobs)
    for role in primary.roles:
        _require_file(primary.paths.role_output(role.name) / "registry" / "final.json")


def _validate_existing_role_configs(primary) -> None:
    """Require the exact manifest-authenticated configs used by old checkpoints."""
    _require_file(primary.paths.role_manifest)
    manifest = read_json(primary.paths.role_manifest)
    if manifest.get("pipeline") != "multi_acccollab":
        raise RuntimeError(
            f"Invalid role-config recovery manifest: {primary.paths.role_manifest}"
        )
    manifest_roles = manifest.get("roles")
    if not isinstance(manifest_roles, dict):
        raise RuntimeError(
            f"Role-config recovery manifest has no role mapping: {primary.paths.role_manifest}"
        )
    for role in primary.roles:
        role_config = primary.paths.role_config(role.name)
        _require_file(role_config)
        role_manifest = manifest_roles.get(role.name)
        if not isinstance(role_manifest, dict):
            raise RuntimeError(f"Role {role.name!r} is absent from recovery manifest")
        expected_hash = str(role_manifest.get("config_sha256") or "")
        actual_hash = file_sha256(role_config)
        if not expected_hash or actual_hash != expected_hash:
            raise RuntimeError(
                f"Role config no longer matches recovery manifest for {role.name}: "
                f"{actual_hash} != {expected_hash or '<missing>'}"
            )


def _role_stage_command(
    role,
    role_config: Path,
    selector: str,
    devices: tuple[int, ...] = (0, 1, 2, 3),
) -> list[str]:
    if selector not in ROLE_TRAINING_STAGES:
        raise ValueError(f"Unsupported role-training selector: {selector}")
    generation_devices = (
        _device_arg(devices)
        if selector.endswith("-data")
        else str(role.device)
    )
    return [
        sys.executable,
        str(ACCCOLLAB_PIPELINE),
        "--config",
        str(role_config),
        "--devices",
        generation_devices,
        "--training-devices",
        str(role.device),
        "--only",
        selector,
    ]


def _run_cross_evaluations(
    boolq_config: str,
    sciq_config: str,
    devices: tuple[int, ...],
) -> None:
    if len(devices) == 1:
        _run(
            [
                sys.executable,
                str(DATASET_EVALUATION),
                "--config",
                boolq_config,
                "--devices",
                str(devices[0]),
            ]
        )
        _run(
            [
                sys.executable,
                str(DATASET_EVALUATION),
                "--config",
                sciq_config,
                "--devices",
                str(devices[0]),
            ]
        )
        return
    boolq_devices, sciq_devices = _two_workload_split(devices, 3270, 1000)
    commands = {
        "boolq": [
            sys.executable,
            str(DATASET_EVALUATION),
            "--config",
            boolq_config,
            "--devices",
            _device_arg(boolq_devices),
        ],
        "sciq": [
            sys.executable,
            str(DATASET_EVALUATION),
            "--config",
            sciq_config,
            "--devices",
            _device_arg(sciq_devices),
        ],
    }
    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        for name, command in commands.items():
            processes.append(
                (
                    name,
                    subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        env=offline_subprocess_env(),
                    ),
                )
            )
        _wait_all(processes)
    except Exception:
        _terminate_all(processes)
        raise


def _run(command: list[str]) -> None:
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=offline_subprocess_env(),
        check=True,
    )


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
                raise RuntimeError(f"Parallel formal stage failed: {name}={code}")
        if pending:
            time.sleep(1.0)


def _run_conflict_free_device_batches(
    jobs: list[tuple[str, int, list[str]]],
) -> None:
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
        processes = [
            (
                name,
                subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=offline_subprocess_env(),
                ),
            )
            for name, _device, command in batch
        ]
        try:
            _wait_all(processes)
        except Exception:
            _terminate_all(processes)
            raise
        pending = deferred


def _parse_devices(raw: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --devices value: {raw!r}") from exc
    if not devices or any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative GPU ids")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")
    return devices


def _device_arg(devices: tuple[int, ...]) -> str:
    return ",".join(str(device) for device in devices)


def _two_workload_split(
    devices: tuple[int, ...],
    left_weight: int,
    right_weight: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(devices) < 2:
        raise ValueError("Concurrent workload split requires at least two GPUs")
    left_count = round(len(devices) * left_weight / (left_weight + right_weight))
    left_count = max(1, min(len(devices) - 1, left_count))
    return devices[:left_count], devices[left_count:]


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


def _visible_gpu_ids() -> set[int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        cwd=PROJECT_ROOT,
        env=offline_subprocess_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    return {int(line.strip()) for line in completed.stdout.splitlines() if line.strip()}


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required formal artifact is missing: {path}")


def _require_success(path: Path, stage: str) -> None:
    _require_file(path)
    marker = read_json(path)
    if marker.get("status") != "complete" or marker.get("stage") != stage:
        raise RuntimeError(f"Invalid formal completion marker for {stage}: {path}")


if __name__ == "__main__":
    main()
