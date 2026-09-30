"""
Model utility functions for detecting model types and architectures.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect_model_type(model_name: str) -> str:
    """
    Detect model architecture type from model name.

    Uses simple string matching on the model name/path to determine
    the architecture type. This is needed for selecting appropriate
    LoRA target modules and other model-specific configurations.

    Args:
        model_name: Model name or path (e.g., "google/gemma-2-2b-it").

    Returns:
        Model architecture type: "llama3", "mistral", "gemma2", "qwen2.5",
        "qwen3_5", "qwen3", or "phi4".
        Defaults to "llama3" if no match is found.

    Examples:
        >>> detect_model_type("meta-llama/Llama-3-8b")
        'llama3'
        >>> detect_model_type("mistralai/Mistral-7B")
        'mistral'
        >>> detect_model_type("google/gemma-2-2b-it")
        'gemma2'
        >>> detect_model_type("Qwen/Qwen2.5-7B-Instruct")
        'qwen2.5'
        >>> detect_model_type("Qwen/Qwen3-8B")
        'qwen3'
        >>> detect_model_type("microsoft/Phi-4-mini-instruct")
        'phi4'
    """
    name = model_name.lower()
    if "llama" in name:
        return "llama3"
    elif "mistral" in name:
        normalized = name.replace("-", "").replace("_", "")
        return "mistral_v03" if "v0.3" in normalized or "v03" in normalized else "mistral"
    elif "gemma" in name:
        return "gemma2"
    elif "qwen" in name:
        # Qwen2.5 must be checked before generic Qwen3
        # Match "qwen2.5" or "qwen-2.5" specifically, avoid false positives like "Qwen3-25B"
        normalized = name.replace(" ", "").replace("-", "").replace("_", "")
        if "qwen2.5" in normalized:
            return "qwen2.5"
        if "qwen3.5" in normalized or "qwen35" in normalized:
            return "qwen3_5"
        return "qwen3"
    elif "phi-4" in name or "phi_4" in name or "phi4" in name:
        return "phi4"
    else:
        # Default fallback
        logger.warning(
            f"Could not detect model type from '{model_name}', defaulting to 'llama3'"
        )
        return "llama3"
