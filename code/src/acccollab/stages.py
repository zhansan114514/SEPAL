"""Stage identities, success markers, and ordering for the ACC-Collab workflow."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.acccollab.config import ACCCollabConfig
from src.acccollab.io import read_json, write_json
from src.acccollab.prompts import ACCCOLLAB_PROMPT_VERSION, prompt_version_for_role
from src.acccollab.registry import PolicyState
from src.inference.vllm_server import inference_prompt_format_version
from src.utils.artifacts import file_sha256, path_identity, stable_fingerprint

PREFERENCE_AGENTS = {"actor", "critic"}
PIPELINE_STAGE_KINDS = {"critic-data", "critic-train", "actor-data", "actor-train", "evaluate"}


@dataclass(frozen=True)
class PipelineStage:
    """One ordered subprocess stage in an ACC-Collab or ACC-Collab+ run."""

    key: str
    selector: str
    script: str
    description: str
    iteration: int | None
    shardable: bool


def build_pipeline_stages(config: ACCCollabConfig) -> list[PipelineStage]:
    """Return Algorithm 1's strict Critic→Actor alternation followed by evaluation."""
    stages: list[PipelineStage] = []
    for iteration in range(1, config.method.alternating_iterations + 1):
        tag = f"iteration-{iteration:02d}"
        stages.extend(
            [
                PipelineStage(
                    key=f"{tag}-critic-data",
                    selector="critic-data",
                    script="01_build_critic_dpo_data.py",
                    description=f"Iteration {iteration}: Critic DPO data",
                    iteration=iteration,
                    shardable=True,
                ),
                PipelineStage(
                    key=f"{tag}-critic-train",
                    selector="critic-train",
                    script="02_train_critic_dpo.py",
                    description=f"Iteration {iteration}: Critic DPO training",
                    iteration=iteration,
                    shardable=False,
                ),
                PipelineStage(
                    key=f"{tag}-actor-data",
                    selector="actor-data",
                    script="03_build_actor_dpo_data.py",
                    description=f"Iteration {iteration}: Actor DPO data",
                    iteration=iteration,
                    shardable=True,
                ),
                PipelineStage(
                    key=f"{tag}-actor-train",
                    selector="actor-train",
                    script="04_train_actor_dpo.py",
                    description=f"Iteration {iteration}: Actor DPO training",
                    iteration=iteration,
                    shardable=False,
                ),
            ]
        )
    stages.append(
        PipelineStage(
            key="evaluate",
            selector="evaluate",
            script="05_evaluate.py",
            description="Five-round final-Actor-only evaluation",
            iteration=None,
            shardable=True,
        )
    )
    return stages


def config_snapshot(config: ACCCollabConfig) -> dict[str, Any]:
    """Return the resolved strict config as a JSON-fingerprintable mapping."""
    payload = asdict(config)
    # Preserve fingerprints of paper-original runs created before the optional
    # multi-role extension existed. Specialized runs retain both sections.
    if (
        config.initialization.actor_adapter is None
        and config.initialization.critic_adapter is None
        and config.prompt_role.name == "original"
        and not config.prompt_role.actor_instruction
        and not config.prompt_role.critic_instruction
    ):
        payload.pop("initialization", None)
        payload.pop("prompt_role", None)
        # This optional field was added for aligned multi-role subsampling.
        # Omitting it when unset preserves all paper-original fingerprints.
        for split_name in ("preference", "eval"):
            split = payload["data"][split_name]
            if split.get("sampling_seed") is None:
                split.pop("sampling_seed", None)
        # Added for released-code-style repeated preference trajectories.
        # A single trajectory preserves identities of all historical runs.
        if payload["data"].get("preference_trials") == 1:
            payload["data"].pop("preference_trials", None)
    return payload


def config_prompt_version(config: ACCCollabConfig) -> str:
    """Return the prompt identity selected by a strict config."""
    role = config.prompt_role
    if role.name == "original":
        # Keep this module-level compatibility seam: provenance tests and old
        # callers intentionally monkeypatch the original version constant.
        return ACCCOLLAB_PROMPT_VERSION
    return prompt_version_for_role(role.name, role.implementation_version)


def policy_state_identity(
    state: PolicyState,
    *,
    hash_adapter_weights: bool = False,
) -> dict[str, Any]:
    """Attach durable adapter identities to a logical Actor/Critic state."""
    return {
        **state.to_dict(),
        "actor_identity": (
            path_identity(state.actor_adapter, hash_weights=hash_adapter_weights)
            if state.actor_adapter
            else "base_model"
        ),
        "critic_identity": (
            path_identity(state.critic_adapter, hash_weights=hash_adapter_weights)
            if state.critic_adapter
            else "base_model"
        ),
    }


def preference_stage_fingerprint(
    config: ACCCollabConfig,
    *,
    agent: str,
    iteration: int,
    state: PolicyState,
    num_shards: int,
) -> str:
    """Fingerprint all semantic inputs to one merged preference-data stage."""
    if agent not in PREFERENCE_AGENTS:
        raise ValueError(f"agent must be actor or critic, got {agent!r}")
    config.validate_iteration(iteration)
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    return stable_fingerprint(
        {
            "pipeline": "acccollab_original",
            "stage": f"{agent}_dpo_data",
            "iteration": iteration,
            "num_shards": num_shards,
            "prompt_version": config_prompt_version(config),
            "inference_prompt_format_version": inference_prompt_format_version(
                config.model.type,
                config.model.name,
            ),
            "config": config_snapshot(config),
            "policy_state": policy_state_identity(
                state,
                hash_adapter_weights=config.prompt_role.name != "original",
            ),
        }
    )


def preference_shard_fingerprint(
    *,
    stage_fingerprint: str,
    shard_idx: int,
    num_shards: int,
    sample_ids: Sequence[str],
    batch_size: int,
) -> str:
    """Fingerprint one deterministic shard/checkpoint layout."""
    return stable_fingerprint(
        {
            "stage_fingerprint": stage_fingerprint,
            "shard_idx": shard_idx,
            "num_shards": num_shards,
            "sample_ids": list(sample_ids),
            "batch_size": batch_size,
        }
    )


def evaluation_stage_fingerprint(
    config: ACCCollabConfig,
    *,
    state: PolicyState,
    num_shards: int,
) -> str:
    """Fingerprint the complete multi-trial evaluation protocol and final policies."""
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    return stable_fingerprint(
        {
            "pipeline": "acccollab_original",
            "stage": "evaluate",
            "num_shards": num_shards,
            "prompt_version": config_prompt_version(config),
            "inference_prompt_format_version": inference_prompt_format_version(
                config.model.type,
                config.model.name,
            ),
            "config": config_snapshot(config),
            "policy_state": policy_state_identity(
                state,
                hash_adapter_weights=config.prompt_role.name != "original",
            ),
            "decision_rule": "single_final_actor_answer",
            "uses_majority_vote": False,
            "uses_judge_fallback": False,
        }
    )


def evaluation_shard_fingerprint(
    *,
    stage_fingerprint: str,
    shard_idx: int,
    num_shards: int,
    sample_ids: Sequence[str],
    batch_size: int,
    trials: int,
) -> str:
    """Fingerprint one evaluation shard's trial/batch checkpoint layout."""
    return stable_fingerprint(
        {
            "stage_fingerprint": stage_fingerprint,
            "shard_idx": shard_idx,
            "num_shards": num_shards,
            "sample_ids": list(sample_ids),
            "batch_size": batch_size,
            "trials": trials,
        }
    )


def write_stage_success(
    path: str | Path,
    *,
    stage: str,
    fingerprint: str,
    artifacts: Mapping[str, str | Path],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a durable JSON success marker with hashes of materialized files."""
    artifact_payload: dict[str, Any] = {}
    for name, artifact_path in artifacts.items():
        target = Path(artifact_path)
        if not target.is_file():
            raise RuntimeError(f"Cannot mark stage complete; artifact is missing: {target}")
        artifact_payload[name] = {
            "path": str(target),
            "size": target.stat().st_size,
            "sha256": file_sha256(target),
        }
    payload = {
        "schema_version": 1,
        "pipeline": "acccollab_original",
        "status": "complete",
        "stage": stage,
        "fingerprint": fingerprint,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifacts": artifact_payload,
        "metadata": dict(metadata or {}),
    }
    write_json(path, payload)
    return payload


def validate_stage_success(
    path: str | Path,
    *,
    expected_stage: str,
    expected_fingerprint: str,
    verify_hashes: bool = True,
) -> dict[str, Any] | None:
    """Return a valid stage marker, otherwise ``None`` without accepting stale files."""
    target = Path(path)
    if not target.is_file():
        return None
    try:
        payload = read_json(target)
    except (OSError, ValueError):
        return None
    if (
        payload.get("pipeline") != "acccollab_original"
        or payload.get("status") != "complete"
        or payload.get("stage") != expected_stage
        or payload.get("fingerprint") != expected_fingerprint
    ):
        return None
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        return None
    for artifact_any in artifacts.values():
        if not isinstance(artifact_any, Mapping):
            return None
        artifact = dict(artifact_any)
        artifact_path = Path(str(artifact.get("path") or ""))
        if not artifact_path.is_file():
            return None
        if int(artifact.get("size", -1)) != artifact_path.stat().st_size:
            return None
        if verify_hashes and artifact.get("sha256") != file_sha256(artifact_path):
            return None
    return payload


def marker_path(config: ACCCollabConfig, stage_key: str) -> Path:
    """Return the pipeline-level marker path for one exact dynamic stage key."""
    return config.paths.marker_dir / f"{stage_key}.json"


def write_pipeline_marker(
    config: ACCCollabConfig,
    *,
    stage: PipelineStage,
    fingerprint: str,
    execution: Mapping[str, Any],
) -> Path:
    """Persist a fingerprint-aware orchestration marker."""
    path = marker_path(config, stage.key)
    write_json(
        path,
        {
            "schema_version": 1,
            "pipeline": "acccollab_original",
            "status": "complete",
            "stage_key": stage.key,
            "selector": stage.selector,
            "iteration": stage.iteration,
            "script": stage.script,
            "fingerprint": fingerprint,
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "execution": dict(execution),
        },
    )
    return path


def pipeline_marker_matches(
    config: ACCCollabConfig,
    *,
    stage: PipelineStage,
    expected_fingerprint: str,
) -> bool:
    """Check a pipeline marker's exact stage and semantic fingerprint."""
    path = marker_path(config, stage.key)
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return False
    return bool(
        payload.get("pipeline") == "acccollab_original"
        and payload.get("status") == "complete"
        and payload.get("stage_key") == stage.key
        and payload.get("fingerprint") == expected_fingerprint
    )
