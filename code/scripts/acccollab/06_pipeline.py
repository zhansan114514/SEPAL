"""Run the resumable, paper-original ACC-Collab alternation pipeline."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from _preference_data import (
    expected_preference_stage_fingerprint,
    preference_stage_name,
    require_merged_preference_stage,
)
from _training import final_registry_exists, registry_matches_iteration
from _utils import (
    add_config_arguments,
    load_config,
    offline_subprocess_env,
    parse_device_list,
    setup_logging,
)
from src.acccollab.config import (
    ACCCollabConfig,
    resolve_acccollab_config_path,
)
from src.acccollab.registry import evaluation_state
from src.acccollab.stages import (
    PipelineStage,
    build_pipeline_stages,
    evaluation_stage_fingerprint,
    pipeline_marker_matches,
    validate_stage_success,
    write_pipeline_marker,
)
from src.acccollab.training import dpo_training_fingerprint
from src.utils.artifacts import completed_adapter_path

logger = logging.getLogger(__name__)
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Build the orchestration CLI."""
    parser = argparse.ArgumentParser(description="Run original ACC-Collab")
    add_config_arguments(parser)
    parser.add_argument(
        "--devices",
        default=None,
        help="Generation/evaluation GPU ids, e.g. 0,1,2,3; defaults to config.",
    )
    parser.add_argument(
        "--training-devices",
        default=None,
        help="Training GPU ids, e.g. 0; defaults to config.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        metavar="STAGE",
        help="Run exact stage keys or selectors such as critic-data/evaluate.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="STAGE",
        help="Skip exact stage keys or selectors.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-enter selected stages; workers still reuse valid checkpoints/artifacts.",
    )
    return parser


def main() -> None:
    """Validate dependencies, execute stages in Algorithm 1 order, and mark them."""
    args = build_parser().parse_args()
    config = load_config(args)
    setup_logging(seed=config.run.seed)
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    config.paths.marker_dir.mkdir(parents=True, exist_ok=True)

    generation_devices = parse_device_list(args.devices) or list(config.runtime.generation_devices)
    training_devices = parse_device_list(args.training_devices) or list(config.runtime.training_devices)
    stages = build_pipeline_stages(config)
    selected = _resolve_stage_tokens(stages, args.only, option="--only") if args.only else {
        stage.key for stage in stages
    }
    skipped = _resolve_stage_tokens(stages, args.skip, option="--skip")
    config_path = str(resolve_acccollab_config_path(str(args.config)))
    overrides = list(args.override or [])

    for index, stage in enumerate(stages, start=1):
        if stage.key not in selected or stage.key in skipped:
            logger.info("Skipping stage %s: %s", stage.key, stage.description)
            continue
        fingerprint = _expected_stage_fingerprint(
            config,
            stage=stage,
            num_generation_shards=len(generation_devices),
        )
        valid_artifact = _stage_artifact_is_valid(
            config,
            stage=stage,
            fingerprint=fingerprint,
            num_generation_shards=len(generation_devices),
        )
        marker_valid = pipeline_marker_matches(
            config,
            stage=stage,
            expected_fingerprint=fingerprint,
        )
        if not args.force and marker_valid and valid_artifact:
            logger.info(
                "[%d/%d] Reusing completed stage %s (fingerprint=%s)",
                index,
                len(stages),
                stage.key,
                fingerprint[:12],
            )
            continue

        logger.info("[%d/%d] Running %s: %s", index, len(stages), stage.key, stage.description)
        started = time.monotonic()
        execution = _run_stage(
            config,
            stage=stage,
            config_path=config_path,
            overrides=overrides,
            generation_devices=generation_devices,
            training_devices=training_devices,
        )
        # Validate both semantic fingerprints and durable output markers after the child exits.
        if not _stage_artifact_is_valid(
            config,
            stage=stage,
            fingerprint=fingerprint,
            num_generation_shards=len(generation_devices),
        ):
            raise RuntimeError(
                f"Stage {stage.key} completed but its expected artifact marker is invalid"
            )
        marker = write_pipeline_marker(
            config,
            stage=stage,
            fingerprint=fingerprint,
            execution={
                **execution,
                "elapsed_seconds": time.monotonic() - started,
                "forced": bool(args.force),
            },
        )
        logger.info("Marked %s complete: %s", stage.key, marker)

    logger.info("Original ACC-Collab pipeline complete.")


def _run_stage(
    config: ACCCollabConfig,
    *,
    stage: PipelineStage,
    config_path: str,
    overrides: list[str],
    generation_devices: list[int],
    training_devices: list[int],
) -> dict[str, object]:
    """Run one stage, using deterministic data parallelism where appropriate."""
    if stage.shardable:
        return _run_shardable_stage(
            stage,
            config_path=config_path,
            overrides=overrides,
            devices=generation_devices,
        )
    return _run_training_stage(
        stage,
        config_path=config_path,
        overrides=overrides,
        device=training_devices[0],
    )


def _run_shardable_stage(
    stage: PipelineStage,
    *,
    config_path: str,
    overrides: list[str],
    devices: list[int],
) -> dict[str, object]:
    """Spawn one worker per physical generation GPU, then merge their shards."""
    if not devices:
        raise ValueError("At least one generation device is required")
    if len(devices) == 1:
        extra = [
            "--shard-idx",
            "0",
            "--num-shards",
            "1",
            "--device",
            str(devices[0]),
        ]
        if stage.iteration is not None:
            extra[0:0] = ["--iteration", str(stage.iteration)]
        command = _script_command(
            stage,
            config_path=config_path,
            overrides=overrides,
            extra=tuple(extra),
        )
        _run_process(command)
        return {"mode": "single_process", "devices": devices, "num_shards": 1}

    processes: list[tuple[int, subprocess.Popen]] = []
    try:
        for shard_idx, device in enumerate(devices):
            extra = [
                "--shard-idx",
                str(shard_idx),
                "--num-shards",
                str(len(devices)),
                "--device",
                str(device),
            ]
            if stage.iteration is not None:
                extra[0:0] = ["--iteration", str(stage.iteration)]
            command = _script_command(
                stage,
                config_path=config_path,
                overrides=overrides,
                extra=tuple(extra),
            )
            logger.info("Starting %s shard %d/%d on GPU %d", stage.key, shard_idx, len(devices), device)
            processes.append((shard_idx, subprocess.Popen(command, cwd=PROJECT_ROOT, env=offline_subprocess_env())))
        _wait_processes(processes, stage.key)
    except Exception:
        _terminate_processes(processes)
        raise

    merge_extra = ["--merge-shards", str(len(devices))]
    if stage.iteration is not None:
        merge_extra[0:0] = ["--iteration", str(stage.iteration)]
    merge_command = _script_command(
        stage,
        config_path=config_path,
        overrides=overrides,
        extra=tuple(merge_extra),
    )
    _run_process(merge_command)
    return {
        "mode": "data_parallel",
        "devices": devices,
        "num_shards": len(devices),
    }


def _run_training_stage(
    stage: PipelineStage,
    *,
    config_path: str,
    overrides: list[str],
    device: int,
) -> dict[str, object]:
    """Run one Critic or Actor DPO job on one selected GPU."""
    command = _script_command(
        stage,
        config_path=config_path,
        overrides=overrides,
        extra=("--iteration", str(stage.iteration), "--device", str(device)),
    )
    _run_process(command)
    return {"mode": "single_training_process", "device": device}


def _script_command(
    stage: PipelineStage,
    *,
    config_path: str,
    overrides: list[str],
    extra: tuple[str, ...],
) -> list[str]:
    command = [sys.executable, str(SCRIPTS_DIR / stage.script), "--config", config_path]
    for override in overrides:
        command.extend(("--override", override))
    if stage.iteration is not None and "--iteration" not in extra:
        command.extend(("--iteration", str(stage.iteration)))
    command.extend(extra)
    return command


def _run_process(command: list[str]) -> None:
    logger.debug("Running command: %s", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, env=offline_subprocess_env(), check=True)


def _wait_processes(processes: list[tuple[int, subprocess.Popen]], stage_key: str) -> None:
    for shard_idx, process in processes:
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"{stage_key} shard {shard_idx} failed with exit code {return_code}")


def _terminate_processes(processes: list[tuple[int, subprocess.Popen]]) -> None:
    for _shard_idx, process in processes:
        if process.poll() is None:
            process.terminate()
    for _shard_idx, process in processes:
        if process.poll() is None:
            process.kill()


def _resolve_stage_tokens(
    stages: list[PipelineStage],
    tokens: list[str],
    *,
    option: str,
) -> set[str]:
    """Expand exact keys and selector aliases, rejecting typos."""
    by_key = {stage.key: stage.key for stage in stages}
    selectors = {stage.selector for stage in stages}
    unknown = sorted(set(tokens) - set(by_key) - selectors)
    if unknown:
        valid = sorted(set(by_key) | selectors)
        raise ValueError(f"Unknown {option} stage(s): {unknown}; valid values: {valid}")
    selected: set[str] = set()
    for token in tokens:
        if token in by_key:
            selected.add(token)
        else:
            selected.update(stage.key for stage in stages if stage.selector == token)
    return selected


def _expected_stage_fingerprint(
    config: ACCCollabConfig,
    *,
    stage: PipelineStage,
    num_generation_shards: int,
) -> str:
    """Derive the exact fingerprint used by the child stage and marker."""
    if stage.selector in {"critic-data", "actor-data"}:
        _state, fingerprint = expected_preference_stage_fingerprint(
            config,
            agent="critic" if stage.selector == "critic-data" else "actor",
            iteration=int(stage.iteration),
            num_shards=num_generation_shards,
        )
        return fingerprint
    if stage.selector == "evaluate":
        state = evaluation_state(config)
        return evaluation_stage_fingerprint(
            config,
            state=state,
            num_shards=num_generation_shards,
        )
    if stage.selector == "critic-train":
        pair_path, _num_shards, _data_fp = require_merged_preference_stage(
            config,
            agent="critic",
            iteration=int(stage.iteration),
        )
        from src.acccollab.registry import critic_training_initial_adapter

        initial = critic_training_initial_adapter(config, int(stage.iteration))
        return dpo_training_fingerprint(
            config,
            agent="critic",
            iteration=int(stage.iteration),
            pair_path=pair_path,
            initial_lora_path=initial,
        )
    if stage.selector == "actor-train":
        pair_path, _num_shards, _data_fp = require_merged_preference_stage(
            config,
            agent="actor",
            iteration=int(stage.iteration),
        )
        from src.acccollab.registry import actor_training_initial_adapter

        initial = actor_training_initial_adapter(config, int(stage.iteration))
        return dpo_training_fingerprint(
            config,
            agent="actor",
            iteration=int(stage.iteration),
            pair_path=pair_path,
            initial_lora_path=initial,
        )
    raise ValueError(f"Unsupported pipeline selector: {stage.selector}")


def _stage_artifact_is_valid(
    config: ACCCollabConfig,
    *,
    stage: PipelineStage,
    fingerprint: str,
    num_generation_shards: int,
) -> bool:
    """Check the actual output artifact in addition to the pipeline marker."""
    if stage.selector in {"critic-data", "actor-data"}:
        agent = "critic" if stage.selector == "critic-data" else "actor"
        return (
            validate_stage_success(
                config.paths.data_dir(int(stage.iteration), agent) / "_SUCCESS",
                expected_stage=preference_stage_name(agent),
                expected_fingerprint=fingerprint,
            )
            is not None
        )
    if stage.selector == "evaluate":
        return (
            validate_stage_success(
                config.paths.eval_dir / "_SUCCESS",
                expected_stage="evaluate",
                expected_fingerprint=fingerprint,
            )
            is not None
        )
    agent = "critic" if stage.selector == "critic-train" else "actor"
    valid_adapter = bool(
        completed_adapter_path(
            config.paths.training_output(int(stage.iteration), agent),
            expected_fingerprint=fingerprint,
        )
    )
    if not valid_adapter:
        return False
    if agent == "actor":
        if not registry_matches_iteration(config, int(stage.iteration)):
            return False
        if int(stage.iteration) == config.method.alternating_iterations:
            return final_registry_exists(config)
    return True


if __name__ == "__main__":
    main()
