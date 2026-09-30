"""Training dtype selection helpers."""

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_training_dtype(model_name_or_path: str) -> tuple["torch.dtype", bool]:
    """Pick a training dtype for the given model.

    Returns (dtype, use_bf16). On bf16-capable hardware -> (bfloat16, True).
    Otherwise Gemma models -> (float32, False) with a warning; everything else
    -> (float16, False).
    """
    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(including_emulation=False)
    )
    if use_bf16:
        return torch.bfloat16, True
    elif "gemma" in model_name_or_path.lower():
        logger.warning(
            "Gemma models don't support fp16 (numerical instability). "
            "Falling back to float32 on this hardware."
        )
        return torch.float32, False
    else:
        return torch.float16, False
