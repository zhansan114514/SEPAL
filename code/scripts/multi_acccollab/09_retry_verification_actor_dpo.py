"""Recover one OOM-failed role Actor DPO run without invalidating its data."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ACCCOLLAB_SCRIPTS = PROJECT_ROOT / "scripts" / "acccollab"
if str(ACCCOLLAB_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(ACCCOLLAB_SCRIPTS))

from _preference_data import require_merged_preference_stage
from src.acccollab.config import load_acccollab_config
from src.acccollab.registry import (
    actor_training_initial_adapter,
    write_final_registry,
    write_iteration_registry,
)
from src.acccollab.training import train_role_dpo

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--gradient-accumulation-steps", required=True, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    original = load_acccollab_config(args.config)
    if original.prompt_role.name != "verification":
        raise ValueError(
            "Recovery entry point only accepts the verification role config, got "
            f"{original.prompt_role.name!r}"
        )
    iteration = original.method.alternating_iterations
    pair_path, _num_shards, _data_fingerprint = require_merged_preference_stage(
        original,
        agent="actor",
        iteration=iteration,
    )

    old_effective_batch = (
        original.training.dpo.batch_size
        * original.training.dpo.gradient_accumulation_steps
    )
    new_effective_batch = (
        args.batch_size * args.gradient_accumulation_steps
    )
    if new_effective_batch != old_effective_batch:
        raise ValueError(
            "Recovery must preserve effective batch size: "
            f"{new_effective_batch} != {old_effective_batch}"
        )

    recovered_dpo = replace(
        original.training.dpo,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    recovered = replace(
        original,
        training=replace(original.training, dpo=recovered_dpo),
    )
    initial_lora = actor_training_initial_adapter(original, iteration)
    completed = train_role_dpo(
        recovered,
        agent="actor",
        iteration=iteration,
        pair_path=pair_path,
        initial_lora_path=initial_lora,
        device=args.device,
    )

    # Registries authenticate the resulting adapters by path and content. Use the
    # formal role config here so downstream evaluation remains bound to the
    # original data and prompt contract.
    write_iteration_registry(original, iteration)
    write_final_registry(original)
    logger.info("Recovered verification Actor adapter: %s", completed)


if __name__ == "__main__":
    main()
