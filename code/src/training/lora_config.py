"""LoRA configuration for parameter-efficient fine-tuning."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Target modules shared by all supported text decoder architectures.
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

PHI3_TARGET_MODULES = [
    "qkv_proj", "o_proj", "gate_up_proj", "down_proj",
]

MODEL_TARGET_MODULES = {
    "llama3": DEFAULT_TARGET_MODULES,
    "mistral": DEFAULT_TARGET_MODULES,
    "mistral_v03": DEFAULT_TARGET_MODULES,
    "gemma2": DEFAULT_TARGET_MODULES,
    "qwen2.5": DEFAULT_TARGET_MODULES,
    "qwen3": DEFAULT_TARGET_MODULES,
    "qwen3_5": DEFAULT_TARGET_MODULES,
    # Phi-4-mini uses Transformers' Phi3 architecture, whose attention and
    # gated-MLP projections are fused rather than separate q/k/v and gate/up.
    "phi4": PHI3_TARGET_MODULES,
}


def get_lora_config(
    model_type: str = "llama3",
    r: int = 128,
    lora_alpha: int = 256,
    lora_dropout: float = 0.0,
):
    """
    Create a LoRA configuration.

    Args:
        model_type: Model architecture type.
        r: LoRA rank.
        lora_alpha: LoRA alpha (default: 2*r).
        lora_dropout: Dropout rate.

    Returns:
        peft.LoraConfig instance.
    """
    from peft import LoraConfig, TaskType

    target_modules = MODEL_TARGET_MODULES.get(model_type, DEFAULT_TARGET_MODULES)

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    logger.info(
        f"LoRA config: r={r}, alpha={lora_alpha}, "
        f"targets={target_modules}"
    )
    return config
