"""Registry helpers for paired Actor/Critic adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.paired import ACTOR_NAMES, paired_critic_name


def actor_names(configured: list[str] | tuple[str, ...] | None = None) -> list[str]:
    names = list(configured) if configured is not None else list(ACTOR_NAMES)
    expected = list(ACTOR_NAMES)
    if names != expected:
        raise ValueError(f"Actor names are fixed to {expected}, got {names}")
    return names


def load_actor_registry(path: str | Path) -> dict[str, str]:
    """Load actor_name -> adapter path from the Actor SFT registry."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    actors = payload.get("actors", {})
    if not isinstance(actors, dict) or not actors:
        raise ValueError(f"No actors found in registry: {path}")
    result: dict[str, str] = {}
    for name, info in actors.items():
        if not isinstance(info, dict):
            continue
        adapter = str(info.get("model_path") or info.get("lora_path") or "")
        if adapter:
            result[str(name)] = adapter
    if not result:
        raise ValueError(f"Actor registry has no adapter paths: {path}")
    return result


def load_final_registry(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_critic_registry(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    critics = payload.get("critics", {})
    if not isinstance(critics, dict) or not critics:
        raise ValueError(f"No critics found in registry: {path}")
    return payload


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def build_final_registry(
    *,
    base_model: str,
    actor_sft_paths: dict[str, str],
    actor_dpo_paths: dict[str, str],
    critic_dpo_paths: dict[str, str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    actors = {}
    critics = {}
    names = actor_names()
    missing = [name for name in names if name not in actor_sft_paths]
    extra = sorted(set(actor_sft_paths) - set(names))
    if missing or extra:
        raise ValueError(
            f"Actor SFT registry must contain exactly {names}; "
            f"missing={missing}, extra={extra}"
        )
    for actor_name in names:
        critic_name = paired_critic_name(actor_name)
        actors[actor_name] = {
            "name": actor_name,
            "base_model": base_model,
            "sft_lora_path": actor_sft_paths.get(actor_name, ""),
            "model_path": actor_dpo_paths.get(actor_name) or actor_sft_paths.get(actor_name, ""),
            "paired_critic": critic_name,
        }
        critics[critic_name] = {
            "name": critic_name,
            "base_model": base_model,
            "parent_actor": actor_name,
            "model_path": critic_dpo_paths.get(critic_name, ""),
            "initialized_from": actor_sft_paths.get(actor_name, ""),
        }
    return {
        "schema_version": 1,
        "pipeline": "paired_actor_critic",
        "base_model": base_model,
        "actors": actors,
        "critics": critics,
        "metadata": metadata or {},
    }


def build_critic_registry(
    *,
    base_model: str,
    actor_sft_paths: dict[str, str],
    critic_dpo_paths: dict[str, str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    names = actor_names()
    critics = {}
    for actor_name in names:
        critic_name = paired_critic_name(actor_name)
        critics[critic_name] = {
            "name": critic_name,
            "base_model": base_model,
            "parent_actor": actor_name,
            "model_path": critic_dpo_paths.get(critic_name, ""),
            "initialized_from": actor_sft_paths.get(actor_name, ""),
        }
    return {
        "schema_version": 1,
        "pipeline": "paired_actor_critic_critic_dpo",
        "base_model": base_model,
        "actors": {
            name: {
                "name": name,
                "base_model": base_model,
                "model_path": actor_sft_paths.get(name, ""),
                "paired_critic": paired_critic_name(name),
            }
            for name in names
        },
        "critics": critics,
        "metadata": metadata or {},
    }
