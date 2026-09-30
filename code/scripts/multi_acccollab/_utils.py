"""Bootstrap and subprocess helpers for multi-role ACC-Collab scripts."""

from __future__ import annotations

# ruff: noqa: E402

import logging
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

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

from src.multi_acccollab.config import MultiACCCollabConfig, load_multi_acccollab_config
from src.utils.runtime_env import configure_runtime_libraries
from src.utils.seeding import fix_seed

configure_runtime_libraries()

DEFAULT_CONFIG = "configs/multi_acccollab/llama3_8b_instruct_mmlu_sft10k.yaml"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(seed: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    if seed is not None:
        fix_seed(int(seed))


def add_config_argument(parser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)


def load_config(path: str) -> MultiACCCollabConfig:
    return load_multi_acccollab_config(path)


def offline_subprocess_env() -> dict[str, str]:
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    for name, default in _OFFLINE_ENV.items():
        env.setdefault(name, default)
    configure_runtime_libraries(env, preload=False)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env
