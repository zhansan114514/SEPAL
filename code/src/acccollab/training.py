"""DPO training facade for the paper-original ACC-Collab Actor and Critic."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.acccollab.config import ACCCollabConfig
from src.acccollab.io import read_jsonl
from src.training.dpo_trainer import train_dpo
from src.training.chat_format import training_prompt_format_version
from src.utils.artifacts import (
    completed_adapter_path,
    file_sha256,
    path_identity,
    stable_fingerprint,
)


def load_dpo_pairs(path: str | Path) -> list[dict[str, Any]]:
    """Load and minimally validate prompt/chosen/rejected JSONL records."""
    rows = read_jsonl(path)
    for index, row in enumerate(rows):
        for key in ("prompt", "chosen", "rejected"):
            if not str(row.get(key) or ""):
                raise ValueError(f"{path}:{index + 1} has an empty {key!r} field")
    return rows


def pairs_to_dataset(pairs: list[dict[str, Any]]):
    """Convert JSON preference rows to the narrow TRL dataset schema."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "prompt": [str(pair["prompt"]) for pair in pairs],
            "chosen": [str(pair["chosen"]) for pair in pairs],
            "rejected": [str(pair["rejected"]) for pair in pairs],
        }
    )


def dpo_training_fingerprint(
    config: ACCCollabConfig,
    *,
    agent: str,
    iteration: int,
    pair_path: str | Path,
    initial_lora_path: str | None,
) -> str:
    """Fingerprint exact data, parent policy, and all optimization settings."""
    if agent not in {"actor", "critic"}:
        raise ValueError(f"agent must be actor or critic, got {agent!r}")
    config.validate_iteration(iteration)
    dpo = config.training.dpo
    payload: dict[str, Any] = {
        "pipeline": "acccollab_original",
        "training_kind": "dpo",
        "agent": agent,
        "iteration": iteration,
        "base_model": path_identity(config.model.name),
        "initial_lora": (
            path_identity(
                initial_lora_path,
                hash_weights=config.prompt_role.name != "original",
            )
            if initial_lora_path
            else "base_model"
        ),
        "pairs": {
            "path": str(Path(pair_path)),
            "sha256": file_sha256(pair_path),
        },
        "lora": {
            "r": config.training.lora.r,
            "alpha": config.training.lora.alpha,
        },
        "dpo": {
            "learning_rate": dpo.learning_rate,
            "batch_size": dpo.batch_size,
            "gradient_accumulation_steps": dpo.gradient_accumulation_steps,
            "epochs": dpo.epochs,
            "beta": dpo.beta,
            "loss_type": dpo.loss_type,
            "nll_weight": dpo.nll_weight,
            "warmup_ratio": dpo.warmup_ratio,
            "weight_decay": dpo.weight_decay,
            "max_grad_norm": dpo.max_grad_norm,
            "optim": dpo.optim,
            "gradient_checkpointing": dpo.gradient_checkpointing,
        },
        "tokens": {
            "max_length": config.tokens.dpo_max_length,
            "prompt": config.tokens.dpo_prompt,
            "completion": config.tokens.dpo_completion,
        },
        "seed": config.run.seed + iteration * 100 + (0 if agent == "critic" else 1),
    }
    if config.prompt_role.name != "original":
        # Keep the paper-original fingerprint byte-for-byte compatible while
        # binding specialized DPO adapters to the exact role prompt contract.
        payload["specialized_prompt_role"] = asdict(config.prompt_role)
    prompt_format_version = training_prompt_format_version(config.model.type)
    if prompt_format_version != "raw_prompt_v1":
        # Model-specific training templates are semantic inputs. Binding them
        # here prevents an adapter trained with a raw prompt from being reused
        # after enabling Mistral/Gemma instruction formatting.
        payload["prompt_format_version"] = prompt_format_version
    return stable_fingerprint(payload)


def train_role_dpo(
    config: ACCCollabConfig,
    *,
    agent: str,
    iteration: int,
    pair_path: str | Path,
    initial_lora_path: str | None,
    device: int,
) -> str:
    """Train or safely reuse one Actor/Critic adapter for an alternation."""
    if agent not in {"actor", "critic"}:
        raise ValueError(f"agent must be actor or critic, got {agent!r}")
    pairs = load_dpo_pairs(pair_path)
    if len(pairs) < config.reward.min_pairs_per_stage:
        raise RuntimeError(
            f"ACC-Collab {agent} DPO requires at least "
            f"{config.reward.min_pairs_per_stage} pairs; found {len(pairs)}"
        )

    output_dir = config.paths.training_output(iteration, agent)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = dpo_training_fingerprint(
        config,
        agent=agent,
        iteration=iteration,
        pair_path=pair_path,
        initial_lora_path=initial_lora_path,
    )
    existing = completed_adapter_path(output_dir, expected_fingerprint=fingerprint)
    if existing:
        return existing

    dpo = config.training.dpo
    train_dpo(
        model_name_or_path=config.model.name,
        preference_dataset=pairs_to_dataset(pairs),
        output_dir=str(output_dir),
        initial_lora_path=initial_lora_path,
        model_type=config.model.type,
        lora_r=config.training.lora.r,
        lora_alpha=config.training.lora.alpha,
        learning_rate=dpo.learning_rate,
        batch_size=dpo.batch_size,
        gradient_accumulation_steps=dpo.gradient_accumulation_steps,
        num_epochs=dpo.epochs,
        dpo_max_length=config.tokens.dpo_max_length,
        dpo_max_prompt_length=config.tokens.dpo_prompt,
        dpo_max_completion_length=config.tokens.dpo_completion,
        warmup_ratio=dpo.warmup_ratio,
        beta=dpo.beta,
        loss_type=dpo.loss_type,
        nll_weight=dpo.nll_weight,
        max_grad_norm=dpo.max_grad_norm,
        optim=dpo.optim,
        weight_decay=dpo.weight_decay,
        seed=config.run.seed + iteration * 100 + (0 if agent == "critic" else 1),
        gradient_checkpointing=dpo.gradient_checkpointing,
        merge_lora=False,
        device=int(device),
        timeout_per_1k=dpo.timeout_per_1k,
        save_steps=dpo.checkpoint_steps,
        save_total_limit=dpo.checkpoint_total_limit,
        resume_from_checkpoint=dpo.resume_from_checkpoint,
        resume_optimizer_state=dpo.resume_optimizer_state,
        training_fingerprint=fingerprint,
        wandb_project="acccollab-original",
    )
    completed = completed_adapter_path(output_dir, expected_fingerprint=fingerprint)
    if not completed:
        raise RuntimeError(f"DPO runner returned without a completed adapter: {output_dir}")
    return completed
