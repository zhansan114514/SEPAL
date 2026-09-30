"""Unified seed fixing for reproducibility."""

from __future__ import annotations


def fix_seed(seed: int = 42) -> None:
    """Fix all random seeds to ensure reproducibility.

    Sets seeds for Python random, NumPy, PyTorch (CPU + CUDA),
    and enables deterministic mode for cuDNN.

    Args:
        seed: The random seed to use across all libraries.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode (reduces performance but guarantees reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
