"""Train and evaluate the full three-role method without Actor SFT."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acccollab.config import InitializationConfig, load_acccollab_config
from src.acccollab.io import read_json, write_json
from src.ablations.config import (
    DATASET_ORDER,
    ROLE_NAMES,
    load_ablation_manifest,
    resolve_manifest_path,
)
from src.ablations.policy_lattice import aggregate_role_records
from src.ablations.runtime import offline_ablation_env
from src.utils.artifacts import file_sha256, stable_fingerprint

ACCCOLLAB_SCRIPTS = PROJECT_ROOT / "scripts/acccollab"
CRITIC_DATA = ACCCOLLAB_SCRIPTS / "01_build_critic_dpo_data.py"
CRITIC_TRAIN = ACCCOLLAB_SCRIPTS / "02_train_critic_dpo.py"
ACTOR_DATA = ACCCOLLAB_SCRIPTS / "03_build_actor_dpo_data.py"
ACTOR_TRAIN = ACCCOLLAB_SCRIPTS / "04_train_actor_dpo.py"
EVALUATE = ACCCOLLAB_SCRIPTS / "05_evaluate.py"
NO_SFT_ORCHESTRATOR_VERSION = "multi_acccollab_no_sft_full_all_datasets_v2"
_ACTIVE: list[subprocess.Popen[Any]] = []


@dataclass(frozen=True)
class RoleRun:
    config_path: Path
    config: Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    devices = _parse_devices(args.devices)
    manifest = load_ablation_manifest(args.manifest)
    output_root = _resolve_path(manifest["output_root"]) / "no_sft"
    training_configs = _load_role_configs(manifest)
    evaluation_configs = _build_evaluation_configs(
        manifest,
        training_configs=training_configs,
        output_root=output_root,
        persist=not args.dry_run,
    )
    plan = _plan(
        manifest,
        training_configs=training_configs,
        evaluation_configs=evaluation_configs,
        devices=devices,
    )
    if args.dry_run:
        print(plan)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "orchestration_plan.json"
    success_path = output_root / "_SUCCESS"
    if _completed(success_path, plan["fingerprint"]):
        return
    write_json(plan_path, plan)
    _install_signal_handlers()

    _run_sharded_stage(
        script=CRITIC_DATA,
        configs=training_configs,
        devices=devices,
        logical_shards=int(manifest["datasets"]["mmlu"]["logical_shards"]),
        stage_name="critic_data",
        iteration=1,
    )
    _run_training_stage(
        script=CRITIC_TRAIN,
        configs=training_configs,
        devices=devices,
        stage_name="critic_train",
        iteration=1,
    )
    _run_sharded_stage(
        script=ACTOR_DATA,
        configs=training_configs,
        devices=devices,
        logical_shards=int(manifest["datasets"]["mmlu"]["logical_shards"]),
        stage_name="actor_data",
        iteration=1,
    )
    _run_training_stage(
        script=ACTOR_TRAIN,
        configs=training_configs,
        devices=devices,
        stage_name="actor_train",
        iteration=1,
    )
    aggregate_dirs: dict[str, Path] = {}
    for dataset_name in DATASET_ORDER:
        logical_shards = int(
            manifest["datasets"][dataset_name]["logical_shards"]
        )
        configs = evaluation_configs[dataset_name]
        _run_sharded_stage(
            script=EVALUATE,
            configs=configs,
            devices=devices,
            logical_shards=logical_shards,
            stage_name=f"evaluate_{dataset_name}",
            iteration=None,
        )
        role_records = {
            role: role_run.config.paths.eval_dir / "records.jsonl"
            for role, role_run in configs.items()
        }
        aggregate_dirs[dataset_name] = aggregate_role_records(
            role_records,
            output_dir=output_root / f"aggregate/{dataset_name}",
            dataset_name=dataset_name,
            variant="no_sft_full",
            expected_samples=int(
                manifest["datasets"][dataset_name]["expected_samples"]
            ),
            reparse_completions=True,
        )

    write_json(
        success_path,
        {
            "schema_version": 1,
            "pipeline": "multi_acccollab_ablation",
            "stage": "no_sft_full",
            "status": "complete",
            "implementation_version": NO_SFT_ORCHESTRATOR_VERSION,
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fingerprint": plan["fingerprint"],
            "artifacts": {
                "plan": {
                    "path": str(plan_path),
                    "sha256": file_sha256(plan_path),
                },
                "aggregate_success": {
                    dataset_name: {
                        "path": str(aggregate_dir / "_SUCCESS"),
                        "fingerprint": read_json(
                            aggregate_dir / "_SUCCESS"
                        )["fingerprint"],
                    }
                    for dataset_name, aggregate_dir in aggregate_dirs.items()
                },
            },
        },
    )


def _run_sharded_stage(
    *,
    script: Path,
    configs: dict[str, RoleRun],
    devices: tuple[int, ...],
    logical_shards: int,
    stage_name: str,
    iteration: int | None,
) -> None:
    jobs = deque(
        (role, shard_idx)
        for role in ROLE_NAMES
        for shard_idx in range(logical_shards)
    )

    def command(role: str, shard_idx: int, device: int) -> list[str]:
        result = [
            sys.executable,
            str(script),
            "--config",
            str(configs[role].config_path),
        ]
        if iteration is not None:
            result.extend(("--iteration", str(iteration)))
        result.extend(
            (
                "--shard-idx",
                str(shard_idx),
                "--num-shards",
                str(logical_shards),
                "--device",
                str(device),
            )
        )
        return result

    _schedule_jobs(
        jobs,
        devices=devices,
        stage_name=stage_name,
        command_builder=command,
    )
    for role in ROLE_NAMES:
        merge = [
            sys.executable,
            str(script),
            "--config",
            str(configs[role].config_path),
        ]
        if iteration is not None:
            merge.extend(("--iteration", str(iteration)))
        merge.extend(("--merge-shards", str(logical_shards)))
        _run(merge)


def _run_training_stage(
    *,
    script: Path,
    configs: dict[str, RoleRun],
    devices: tuple[int, ...],
    stage_name: str,
    iteration: int,
) -> None:
    jobs = deque((role, 0) for role in ROLE_NAMES)

    def command(role: str, _unused: int, device: int) -> list[str]:
        return [
            sys.executable,
            str(script),
            "--config",
            str(configs[role].config_path),
            "--iteration",
            str(iteration),
            "--device",
            str(device),
        ]

    _schedule_jobs(
        jobs,
        devices=devices,
        stage_name=stage_name,
        command_builder=command,
    )


def _schedule_jobs(
    jobs: deque[tuple[str, int]],
    *,
    devices: tuple[int, ...],
    stage_name: str,
    command_builder: Any,
) -> None:
    running: dict[int, tuple[str, int, subprocess.Popen[Any]]] = {}
    while jobs or running:
        for device in devices:
            if device in running or not jobs:
                continue
            role, index = jobs.popleft()
            process = subprocess.Popen(
                command_builder(role, index, device),
                cwd=PROJECT_ROOT,
                env=_offline_env(),
            )
            running[device] = (role, index, process)
            _ACTIVE.append(process)
        for device, (role, index, process) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            running.pop(device)
            if process in _ACTIVE:
                _ACTIVE.remove(process)
            if code != 0:
                _terminate_active()
                raise RuntimeError(
                    f"{stage_name}/{role}/{index} failed on GPU {device}: exit={code}"
                )
        if running:
            time.sleep(1)


def _load_role_configs(manifest: dict[str, Any]) -> dict[str, RoleRun]:
    records = dict(manifest["no_sft_role_configs"])
    if set(records) != set(ROLE_NAMES):
        raise RuntimeError("No-SFT role config set is invalid")
    configs: dict[str, RoleRun] = {}
    for role in ROLE_NAMES:
        record = dict(records[role])
        path = resolve_manifest_path(record["path"], project_root=PROJECT_ROOT)
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"No-SFT config changed after materialization: {path}")
        config = load_acccollab_config(str(path))
        if (
            config.prompt_role.name != role
            or config.initialization.actor_adapter is not None
            or config.initialization.critic_adapter is not None
        ):
            raise RuntimeError(f"No-SFT initialization is invalid for {role}")
        configs[role] = RoleRun(config_path=path, config=config)
    return configs


def _build_evaluation_configs(
    manifest: dict[str, Any],
    *,
    training_configs: dict[str, RoleRun],
    output_root: Path,
    persist: bool,
) -> dict[str, dict[str, RoleRun]]:
    """Derive five-dataset configs that authenticate the no-SFT trained policies."""
    result: dict[str, dict[str, RoleRun]] = {"mmlu": training_configs}
    for dataset_name in DATASET_ORDER:
        if dataset_name == "mmlu":
            continue
        dataset_roles = dict(manifest["datasets"][dataset_name]["roles"])
        role_configs: dict[str, RoleRun] = {}
        for role in ROLE_NAMES:
            record = dict(dataset_roles[role]["config"])
            source_path = resolve_manifest_path(
                record["path"],
                project_root=PROJECT_ROOT,
            )
            if file_sha256(source_path) != record["sha256"]:
                raise RuntimeError(
                    f"Source evaluation config changed after materialization: "
                    f"{source_path}"
                )
            template = load_acccollab_config(str(source_path))
            if template.data.dataset != dataset_name or template.prompt_role.name != role:
                raise RuntimeError(
                    f"Invalid source evaluation config for {dataset_name}/{role}"
                )
            training_output = str(training_configs[role].config.run.output_dir)
            evaluation_output = output_root / f"eval_runs/{dataset_name}/roles/{role}"
            derived = replace(
                template,
                run=replace(
                    template.run,
                    name=f"{template.run.name}_no_sft",
                    output_dir=str(evaluation_output),
                ),
                evaluation=replace(
                    template.evaluation,
                    policy_output_dir=training_output,
                ),
                initialization=InitializationConfig(
                    actor_adapter=None,
                    critic_adapter=None,
                ),
            )
            derived.validate()
            config_path = (
                output_root
                / f"resolved_eval_configs/{dataset_name}/{role}.yaml"
            )
            if persist:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                OmegaConf.save(
                    config=OmegaConf.create(asdict(derived)),
                    f=config_path,
                )
            role_configs[role] = RoleRun(config_path=config_path, config=derived)
        result[dataset_name] = role_configs
    if tuple(result) != DATASET_ORDER:
        result = {dataset: result[dataset] for dataset in DATASET_ORDER}
    return result


def _plan(
    manifest: dict[str, Any],
    *,
    training_configs: dict[str, RoleRun],
    evaluation_configs: dict[str, dict[str, RoleRun]],
    devices: tuple[int, ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": "multi_acccollab_ablation",
        "stage": "no_sft_full",
        "implementation_version": NO_SFT_ORCHESTRATOR_VERSION,
        "manifest_fingerprint": manifest["fingerprint"],
        "model": dict(manifest["model"]),
        "training_dataset": "mmlu",
        "evaluation_datasets": {
            dataset_name: {
                "expected_samples": int(
                    manifest["datasets"][dataset_name]["expected_samples"]
                ),
                "logical_shards": int(
                    manifest["datasets"][dataset_name]["logical_shards"]
                ),
                "role_configs": {
                    role: {
                        "path": str(role_run.config_path),
                        "semantic_fingerprint": stable_fingerprint(
                            asdict(role_run.config)
                        ),
                        "policy_output_dir": (
                            role_run.config.evaluation.policy_output_dir
                        ),
                    }
                    for role, role_run in evaluation_configs[dataset_name].items()
                },
            }
            for dataset_name in DATASET_ORDER
        },
        "physical_devices": list(devices),
        "role_configs": {
            role: {
                "path": str(role_run.config_path),
                "sha256": file_sha256(role_run.config_path),
                "actor_initialization": "base_model",
                "critic_initialization": "base_model",
            }
            for role, role_run in training_configs.items()
        },
        "stage_order": [
            "critic_data",
            "critic_train",
            "actor_data",
            "actor_train",
            *[
                f"evaluate_{dataset_name}"
                for dataset_name in DATASET_ORDER
            ],
            *[
                f"majority_aggregate_{dataset_name}"
                for dataset_name in DATASET_ORDER
            ],
        ],
        "uses_judge": False,
        "preserves_four_logical_shards_on_smaller_physical_gpu_pools": True,
    }
    payload["fingerprint"] = stable_fingerprint(payload)
    return payload


def _completed(path: Path, expected_fingerprint: str) -> bool:
    if not path.is_file():
        return False
    payload = read_json(path)
    return (
        payload.get("status") == "complete"
        and payload.get("stage") == "no_sft_full"
        and payload.get("fingerprint") == expected_fingerprint
    )


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, env=_offline_env(), check=True)


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


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _offline_env() -> dict[str, str]:
    return offline_ablation_env(PROJECT_ROOT, base=os.environ)


def _install_signal_handlers() -> None:
    def stop(_signum: int, _frame: Any) -> None:
        _terminate_active()
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, stop)


def _terminate_active() -> None:
    for process in list(_ACTIVE):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 20
    while any(process.poll() is None for process in _ACTIVE) and time.monotonic() < deadline:
        time.sleep(0.5)
    for process in list(_ACTIVE):
        if process.poll() is None:
            process.kill()
    _ACTIVE.clear()


if __name__ == "__main__":
    main()
