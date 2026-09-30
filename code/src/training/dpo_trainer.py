"""
DPO training using trl library.

Implements the DPO loss from Eq. 6 of the ACC-Collab paper.
"""

from __future__ import annotations

import logging
import os

from src.utils.runtime_env import configure_runtime_libraries

configure_runtime_libraries()


from src.training._common import run_training_subprocess  # noqa: E402

logger = logging.getLogger(__name__)


def train_dpo(
    model_name_or_path: str,
    preference_dataset,
    output_dir: str,
    initial_lora_path: str | None = None,
    model_type: str = "gemma2",
    lora_r: int = 128,
    lora_alpha: int = 256,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    dpo_max_length: int = 4096,
    dpo_max_prompt_length: int = 3072,
    dpo_max_completion_length: int = 1024,
    warmup_ratio: float = 0.1,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    nll_weight: float = 1.0,
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
    save_steps: int | None = None,
    save_total_limit: int | None = 2,
    resume_from_checkpoint: bool = True,
    resume_optimizer_state: bool = True,
    training_fingerprint: str | None = None,
) -> str:
    """
    Train a model with DPO + LoRA.

    Args:
        model_name_or_path: Base model name or path.
        preference_dataset: HuggingFace Dataset with prompt/chosen/rejected.
        output_dir: Where to save the trained model.
        initial_lora_path: Existing LoRA adapter to continue from. When omitted, create a new
            trainable LoRA on the base model and use the adapter-disabled base as reference.
        model_type: Model architecture for LoRA target modules.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        learning_rate: Learning rate.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation.
        num_epochs: Number of training epochs.
        dpo_max_length: Max total DPO sequence length.
        dpo_max_prompt_length: Max prompt length before truncation.
        dpo_max_completion_length: Max chosen/rejected completion length.
        warmup_ratio: Warmup ratio.
        beta: DPO beta parameter controlling deviation from reference policy.
        loss_type: DPO loss type ("sigmoid", "hinge", "ipo", etc.).
        nll_weight: Weight for chosen-completion NLL regularization.
            ACC-Collab uses an NLL term alongside DPO. Set to 0.0 to disable
            that regularizer and use pure DPO loss only.
        max_grad_norm: Max gradient norm for clipping (important for FP16).
        optim: Optimizer type.
        weight_decay: Weight decay for optimizer.
        seed: Random seed.
        use_wandb: Whether to log to wandb.
        wandb_project: Wandb project name.
        gradient_checkpointing: Whether to use gradient checkpointing.
        merge_lora: Whether to merge the trained LoRA into a full model.
            Development runs keep this False so vLLM can load adapters directly.
        device: CUDA device index.
        timeout_per_1k: Base timeout in seconds per 1000 preference pairs.
            The total subprocess timeout is ``timeout_per_1k *
            ceil(n_pairs / 1000)``, minimum 1800 s.  Default 1800 s means
            a 200-pair job gets 1800 s, a 5000-pair job gets 9000 s.
        save_steps: Save trainer checkpoints every N optimizer steps. When
            unset or <=0, save only at epoch boundaries.
        save_total_limit: Maximum number of trainer checkpoints to keep.
        resume_from_checkpoint: Resume from the latest checkpoint-* under
            output_dir when present.
        resume_optimizer_state: Load optimizer state when resuming from a
            trainer checkpoint.
        training_fingerprint: Optional caller-computed identity for the exact pair file and
            hyperparameters. When omitted, the runner retains its legacy derived identity.

    Returns:
        Path to saved model.
    """

    if min(dpo_max_length, dpo_max_prompt_length, dpo_max_completion_length) <= 0:
        raise ValueError(
            "DPO lengths must be positive: "
            f"max={dpo_max_length}, prompt={dpo_max_prompt_length}, "
            f"completion={dpo_max_completion_length}"
        )
    if dpo_max_prompt_length + dpo_max_completion_length > dpo_max_length:
        raise ValueError(
            "DPO prompt/completion budgets exceed dpo_max_length: "
            f"{dpo_max_prompt_length} + {dpo_max_completion_length} > "
            f"{dpo_max_length}"
        )

    # Build subprocess config
    _config = {
        "model_name_or_path": model_name_or_path,
        "output_dir": output_dir,
        "initial_lora_path": initial_lora_path,
        "model_type": model_type,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_epochs": num_epochs,
        "dpo_max_length": dpo_max_length,
        "dpo_max_prompt_length": dpo_max_prompt_length,
        "dpo_max_completion_length": dpo_max_completion_length,
        "warmup_ratio": warmup_ratio,
        "beta": beta,
        "loss_type": loss_type,
        "nll_weight": nll_weight,
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
        "resume_optimizer_state": resume_optimizer_state,
        "training_fingerprint": training_fingerprint,
    }

    _runner_script = os.path.join(os.path.dirname(__file__), "_dpo_runner.py")
    return run_training_subprocess(
        runner_script=_runner_script,
        config=_config,
        dataset=preference_dataset,
        device=device,
        timeout_per_1k=timeout_per_1k,
        temp_prefix="dpo_",
        dataset_subdir="preference_dataset",
        logger=logger,
    )
