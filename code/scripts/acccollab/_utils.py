"""Shared bootstrap and runtime helpers for isolated ACC-Collab entry points."""

from __future__ import annotations

# Direct entry points bootstrap the repository before importing project modules.
# ruff: noqa: E402

import argparse
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

# This machine is deliberately offline. Set these before datasets,
# transformers, or vLLM can attempt network access in this process or children.
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HOME": "DATA_ROOT/.cache/huggingface",
    "HF_DATASETS_CACHE": "DATA_ROOT/.cache/huggingface/datasets",
    "HF_HUB_CACHE": "DATA_ROOT/.cache/huggingface/hub",
}
for _name, _value in _OFFLINE_ENV.items():
    os.environ.setdefault(_name, _value)

from src.acccollab.config import ACCCollabConfig, load_acccollab_config
from src.utils.generation_audit import assess_generation_stats
from src.utils.runtime_env import configure_runtime_libraries
from src.utils.seeding import fix_seed

configure_runtime_libraries()

DEFAULT_CONFIG = "configs/acccollab/llama3_8b_instruct_mmlu_original.yaml"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_T = TypeVar("_T")


def setup_logging(*, seed: int | None = None, level: int = logging.INFO) -> None:
    """Configure logging and optionally fix all local random generators."""
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
    if seed is not None:
        fix_seed(int(seed))
        logging.getLogger(__name__).info("Random seed fixed to %d", seed)


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the strict YAML path and optional OmegaConf dot-list overrides."""
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict config override; repeat for multiple dot-list entries.",
    )


def load_config(args: argparse.Namespace) -> ACCCollabConfig:
    """Load the requested strict ACC-Collab configuration."""
    return load_acccollab_config(str(args.config), list(args.override or []))


def batched(items: Sequence[_T], batch_size: int) -> Iterator[tuple[int, list[_T]]]:
    """Yield deterministic ``(batch_index, list)`` chunks."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for start in range(0, len(items), batch_size):
        yield start // batch_size, list(items[start : start + batch_size])


def expected_batches(item_count: int, batch_size: int) -> int:
    """Return ceil(item_count / batch_size), including zero-item shards."""
    if item_count < 0 or batch_size < 1:
        raise ValueError("item_count must be non-negative and batch_size positive")
    return (item_count + batch_size - 1) // batch_size


def parse_device_list(raw: str | None) -> list[int]:
    """Parse unique non-negative physical CUDA ids from a comma-separated CLI value."""
    if raw is None:
        return []
    devices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not devices:
        raise ValueError("Device list must contain at least one GPU id")
    if len(set(devices)) != len(devices) or any(device < 0 for device in devices):
        raise ValueError(f"GPU ids must be unique and non-negative, got {devices}")
    return devices


def offline_subprocess_env() -> dict[str, str]:
    """Return a child environment with repository import and offline guarantees."""
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    for name, default in _OFFLINE_ENV.items():
        env.setdefault(name, default)
    configure_runtime_libraries(env, preload=False)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def generation_audit_summary(
    rows_or_stats,
    *,
    config: ACCCollabConfig,
) -> dict[str, Any]:
    """Apply the configured truncation assessment to raw per-batch deltas."""
    return assess_generation_stats(
        rows_or_stats,
        warn_rate=float(config.generation.truncation.warn_rate),
        fail_rate=float(config.generation.truncation.fail_rate),
        max_model_len=int(config.model.max_model_len),
    )
