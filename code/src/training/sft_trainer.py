"""Supervised fine-tuning with response-only loss and LoRA adapters."""

from __future__ import annotations

import logging
import os

from src.utils.runtime_env import configure_runtime_libraries

configure_runtime_libraries()


from src.training._common import run_training_subprocess  # noqa: E402

logger = logging.getLogger(__name__)


def train_sft(
    model_name_or_path: str,
    sft_dataset,
    output_dir: str,
    model_type: str = "qwen3",
    lora_r: int = 128,
    lora_alpha: int = 256,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    max_length: int = 2048,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    optim: str = "adamw_torch",
    weight_decay: float = 0.01,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "paired-actor-critic",
    gradient_checkpointing: bool = True,
    merge_lora: bool = False,
    device: int = 0,
    timeout_per_1k: int = 1800,
    save_steps: int | None = 100,
    save_total_limit: int | None = 2,
    resume_from_checkpoint: bool = True,
    training_fingerprint: str | None = None,
) -> str:
    """Train a LoRA adapter with causal LM loss on response tokens only."""

    _config = {
        "model_name_or_path": model_name_or_path,
        "output_dir": output_dir,
        "model_type": model_type,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_epochs": num_epochs,
        "max_length": max_length,
        "warmup_ratio": warmup_ratio,
        "max_grad_norm": max_grad_norm,
        "optim": optim,
        "weight_decay": weight_decay,
        "seed": seed,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "gradient_checkpointing": gradient_checkpointing,
        "merge_lora": merge_lora,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "resume_from_checkpoint": resume_from_checkpoint,
        "training_fingerprint": training_fingerprint,
    }
    _runner_script = os.path.join(os.path.dirname(__file__), "_sft_runner.py")
    return run_training_subprocess(
        runner_script=_runner_script,
        config=_config,
        dataset=sft_dataset,
        device=device,
        timeout_per_1k=timeout_per_1k,
        temp_prefix="sft_data_",
        dataset_subdir="sft_dataset",
        logger=logger,
    )
