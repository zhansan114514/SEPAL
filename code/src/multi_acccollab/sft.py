"""SFT data loading, training identity, and registry helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.acccollab.io import iter_jsonl, read_json, write_json
from src.multi_acccollab.config import MultiACCCollabConfig, RoleConfig, config_snapshot
from src.utils.artifacts import (
    completed_adapter_path,
    file_sha256,
    path_identity,
    stable_fingerprint,
)


def load_role_sft_rows(
    config: MultiACCCollabConfig,
    role_name: str,
) -> list[dict[str, str]]:
    """Load only the narrow prompt/response schema for one role."""
    path = config.paths.sft_data_dir / "sft_rows.jsonl"
    return [
        {"prompt": str(row["prompt"]), "response": str(row["response"])}
        for row in iter_jsonl(path)
        if str(row.get("role")) == role_name
    ]


def select_matched_sft_rows(
    config: MultiACCCollabConfig,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one correct target per role only on the common sample intersection."""
    temperature_order = {
        float(value): index for index, value in enumerate(config.sft.generation.temperatures)
    }
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    sample_ids: dict[int, str] = {}
    for row in candidates:
        sample_index = int(row["sample_index"])
        role_name = str(row["role"])
        sample_ids[sample_index] = str(row["sample_id"])
        if row["correct"] and row["response"].strip() and not row["truncated"]:
            grouped.setdefault((sample_index, role_name), []).append(row)

    selected: list[dict[str, Any]] = []
    for sample_index in sorted(sample_ids):
        per_role: dict[str, dict[str, Any]] = {}
        for role in config.roles:
            usable = grouped.get((sample_index, role.name), [])
            if not usable:
                break
            per_role[role.name] = min(
                usable,
                key=lambda row: temperature_order[float(row["temperature"])],
            )
        if len(per_role) != len(config.roles):
            continue
        for role in config.roles:
            row = per_role[role.name]
            selected.append(
                {
                    "sample_index": sample_index,
                    "sample_id": sample_ids[sample_index],
                    "role": role.name,
                    "temperature": row["temperature"],
                    "prompt_version": row["prompt_version"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "answer": row["answer"],
                    "selection_rule": "one_correct_per_role_on_common_sample_intersection",
                }
            )
    return selected


def validate_balanced_sft_rows(
    config: MultiACCCollabConfig,
    rows: list[Mapping[str, Any]],
) -> int:
    """Require one row per role/sample and identical sample sets across roles."""
    role_samples = {role.name: set() for role in config.roles}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        role = str(row.get("role") or "")
        sample_id = str(row.get("sample_id") or "")
        if role not in role_samples or not sample_id:
            raise RuntimeError(f"Malformed balanced SFT row: role={role!r}, sample={sample_id!r}")
        key = (role, sample_id)
        if key in seen:
            raise RuntimeError(f"Duplicate balanced SFT row: {key}")
        seen.add(key)
        role_samples[role].add(sample_id)
    sample_sets = list(role_samples.values())
    if any(samples != sample_sets[0] for samples in sample_sets[1:]):
        raise RuntimeError("Role SFT rows do not cover the same source sample ids")
    return len(sample_sets[0]) if sample_sets else 0


def validate_sft_data_stage(config: MultiACCCollabConfig) -> dict[str, Any]:
    """Authenticate balanced SFT rows against the current config and source tree."""
    marker_path = config.paths.sft_data_dir / "_SUCCESS"
    try:
        marker = read_json(marker_path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Missing or malformed SFT data marker: {marker_path}") from exc
    expected_config = stable_fingerprint(config_snapshot(config))
    if (
        marker.get("pipeline") != "multi_acccollab"
        or marker.get("status") != "complete"
        or marker.get("stage") != "actor_sft_data"
        or marker.get("config_fingerprint") != expected_config
    ):
        raise RuntimeError(f"SFT data marker is stale for the current experiment: {marker_path}")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise RuntimeError(f"SFT data marker has no authenticated artifacts: {marker_path}")
    for artifact_any in artifacts.values():
        if not isinstance(artifact_any, Mapping):
            raise RuntimeError(f"Malformed SFT artifact marker: {marker_path}")
        path = Path(str(artifact_any.get("path") or ""))
        if not path.is_file() or artifact_any.get("sha256") != file_sha256(path):
            raise RuntimeError(f"SFT data artifact is missing or corrupted: {path}")
    return marker


def sft_training_fingerprint(
    config: MultiACCCollabConfig,
    role: RoleConfig,
) -> str:
    """Fingerprint exact role data, prompts, initialization, and optimizer settings."""
    rows_path = config.paths.sft_data_dir / "sft_rows.jsonl"
    base = config.base_config()
    return stable_fingerprint(
        {
            "pipeline": "multi_acccollab",
            "training_kind": "role_actor_sft",
            "config": config_snapshot(config),
            "role": {
                "name": role.name,
                "actor_instruction": role.actor_instruction,
                "seed": config.run.seed + role.seed_offset,
            },
            "base_model": path_identity(base.model.name),
            "sft_rows": {"path": str(rows_path), "sha256": file_sha256(rows_path)},
            "lora": {"r": base.training.lora.r, "alpha": base.training.lora.alpha},
            "training": {
                "learning_rate": config.sft.training.learning_rate,
                "batch_size": config.sft.training.batch_size,
                "gradient_accumulation_steps": (
                    config.sft.training.gradient_accumulation_steps
                ),
                "epochs": config.sft.training.epochs,
                "max_length": config.sft.training.max_length,
                "warmup_ratio": config.sft.training.warmup_ratio,
                "weight_decay": config.sft.training.weight_decay,
                "max_grad_norm": config.sft.training.max_grad_norm,
                "optim": config.sft.training.optim,
            },
        }
    )


def write_sft_registry(config: MultiACCCollabConfig) -> dict[str, Any]:
    """Authenticate all three role adapters and write their exact identities."""
    validate_sft_data_stage(config)
    roles: dict[str, Any] = {}
    for role in config.roles:
        fingerprint = sft_training_fingerprint(config, role)
        completed = completed_adapter_path(
            config.paths.sft_training_output(role.name),
            expected_fingerprint=fingerprint,
        )
        if not completed:
            raise RuntimeError(f"Missing completed SFT adapter for role {role.name}")
        roles[role.name] = {
            "adapter": completed,
            "identity": path_identity(completed, hash_weights=True),
            "training_fingerprint": fingerprint,
            "source_rows": len(load_role_sft_rows(config, role.name)),
        }
    payload = {
        "schema_version": 1,
        "pipeline": "multi_acccollab",
        "stage": "actor_sft_train",
        "status": "complete",
        "base_model": config.base_config().model.name,
        "initializes": "actor_iteration_zero_only",
        "critic_initialization": "base_model",
        "config_fingerprint": stable_fingerprint(config_snapshot(config)),
        "roles": roles,
    }
    write_json(config.paths.sft_registry, payload)
    return payload
