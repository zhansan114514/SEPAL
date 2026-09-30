"""Offline runtime environment shared by ablation entry points."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from src.utils.runtime_env import configure_runtime_libraries


def offline_ablation_env(
    project_root: str | Path,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a process environment with an existing local HF cache."""
    env = dict(base or os.environ)
    cache_home = env.get("HF_HOME") or _discover_hf_home()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_ENDPOINT": env.get("HF_ENDPOINT", "https://hf-mirror.com"),
            "HF_HOME": cache_home,
            "HF_DATASETS_CACHE": env.get(
                "HF_DATASETS_CACHE",
                str(Path(cache_home) / "datasets"),
            ),
            "HF_HUB_CACHE": env.get(
                "HF_HUB_CACHE",
                str(Path(cache_home) / "hub"),
            ),
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
            "PYTHONPATH": str(Path(project_root)),
            "PYTHONUNBUFFERED": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": env.get(
                "VLLM_WORKER_MULTIPROC_METHOD",
                "spawn",
            ),
        }
    )
    configure_runtime_libraries(env, preload=False)
    return env


def configure_current_ablation_process(project_root: str | Path) -> None:
    """Apply the same offline environment before datasets or vLLM imports."""
    os.environ.update(offline_ablation_env(project_root))


def _discover_hf_home() -> str:
    candidates = (
        Path("DATA_ROOT/.cache/huggingface"),
        Path("/home/storage/cache/huggingface"),
        Path.home() / ".cache/huggingface",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return str(candidates[-1])
