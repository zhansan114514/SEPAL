"""Training helpers for paired Actor/Critic DPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset

from src.training.dpo_trainer import train_dpo
from src.utils.model_utils import detect_model_type


def pairs_to_dataset(pairs: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_dict({
        "prompt": [str(pair["prompt"]) for pair in pairs],
        "chosen": [str(pair["chosen"]) for pair in pairs],
        "rejected": [str(pair["rejected"]) for pair in pairs],
    })


def load_jsonl_pairs(path: str | Path) -> list[dict[str, Any]]:
    import json

    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def train_pairs_dpo(
    *,
    model_name: str,
    pairs: list[dict[str, Any]],
    output_dir: str,
    initial_lora_path: str,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    beta: float,
    nll_weight: float,
    max_length: int,
    max_prompt_length: int,
    max_completion_length: int,
    optim: str,
    seed: int,
    device: int,
    timeout_per_1k: int,
    checkpoint_steps: int | None = 100,
    checkpoint_total_limit: int | None = 2,
    resume_from_checkpoint: bool = True,
    resume_optimizer_state: bool = False,
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return train_dpo(
        model_name_or_path=model_name,
        preference_dataset=pairs_to_dataset(pairs),
        output_dir=output_dir,
        initial_lora_path=initial_lora_path,
        model_type=detect_model_type(model_name),
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
        dpo_max_length=max_length,
        dpo_max_prompt_length=max_prompt_length,
        dpo_max_completion_length=max_completion_length,
        beta=beta,
        nll_weight=nll_weight,
        optim=optim,
        seed=seed,
        device=device,
        timeout_per_1k=timeout_per_1k,
        save_steps=checkpoint_steps,
        save_total_limit=checkpoint_total_limit,
        resume_from_checkpoint=resume_from_checkpoint,
        resume_optimizer_state=resume_optimizer_state,
    )
