"""Shared training-stage helpers for the isolated ACC-Collab reproduction."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import logging

from _preference_data import require_merged_preference_stage
from _utils import add_config_arguments, load_config, setup_logging
from src.acccollab.config import ACCCollabConfig
from src.acccollab.registry import (
    actor_training_initial_adapter,
    critic_training_initial_adapter,
    final_registry_matches,
    iteration_registry_matches,
    write_final_registry,
    write_iteration_registry,
)
from src.acccollab.training import dpo_training_fingerprint, train_role_dpo
from src.utils.artifacts import completed_adapter_path

logger = logging.getLogger(__name__)


def training_parser(agent: str) -> argparse.ArgumentParser:
    """Build the numbered training script parser."""
    if agent not in {"actor", "critic"}:
        raise ValueError(f"agent must be actor or critic, got {agent!r}")
    parser = argparse.ArgumentParser(
        description=f"Train paper-original ACC-Collab {agent.title()} DPO adapter"
    )
    add_config_arguments(parser)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Only validate the completed adapter and write the iteration registry.",
    )
    return parser


def run_training_stage(
    config: ACCCollabConfig,
    *,
    agent: str,
    iteration: int,
    device: int | None,
    finalize_only: bool = False,
) -> str:
    """Train one role, safely reuse matching output, and finalize registries."""
    if agent not in {"actor", "critic"}:
        raise ValueError(f"agent must be actor or critic, got {agent!r}")
    config.validate_iteration(iteration)
    pair_path, _num_shards, _data_fingerprint = require_merged_preference_stage(
        config,
        agent=agent,
        iteration=iteration,
    )
    initial_lora = (
        critic_training_initial_adapter(config, iteration)
        if agent == "critic"
        else actor_training_initial_adapter(config, iteration)
    )
    fingerprint = dpo_training_fingerprint(
        config,
        agent=agent,
        iteration=iteration,
        pair_path=pair_path,
        initial_lora_path=initial_lora,
    )
    output_dir = config.paths.training_output(iteration, agent)
    completed = completed_adapter_path(output_dir, expected_fingerprint=fingerprint)
    if not completed and not finalize_only:
        selected_device = (
            int(device)
            if device is not None
            else int(config.runtime.training_devices[0])
        )
        completed = train_role_dpo(
            config,
            agent=agent,
            iteration=iteration,
            pair_path=pair_path,
            initial_lora_path=initial_lora,
            device=selected_device,
        )
    if not completed:
        raise RuntimeError(
            f"No completed {agent} adapter is available for iteration {iteration}: {output_dir}"
        )
    # Re-check the exact fingerprint after the runner returns. This catches a runner that
    # emitted an adapter but failed to propagate the caller's semantic identity.
    completed = completed_adapter_path(output_dir, expected_fingerprint=fingerprint)
    if not completed:
        raise RuntimeError(f"Completed adapter validation failed: {output_dir}")

    if agent == "actor":
        write_iteration_registry(config, iteration)
        if iteration == config.method.alternating_iterations:
            write_final_registry(config)
    logger.info(
        "Completed ACC-Collab %s iteration %d adapter: %s",
        agent,
        iteration,
        completed,
    )
    return completed


def registry_matches_iteration(config: ACCCollabConfig, iteration: int) -> bool:
    """Compatibility wrapper for exact current-adapter registry validation."""
    return iteration_registry_matches(config, iteration)


def final_registry_exists(config: ACCCollabConfig) -> bool:
    """Compatibility wrapper for exact final policy-state registry validation."""
    return final_registry_matches(config)


def main(agent: str) -> None:
    """CLI entry point for one numbered training wrapper."""
    args = training_parser(agent).parse_args()
    config = load_config(args)
    setup_logging(seed=config.run.seed)
    run_training_stage(
        config,
        agent=agent,
        iteration=int(args.iteration),
        device=args.device,
        finalize_only=bool(args.finalize_only),
    )
