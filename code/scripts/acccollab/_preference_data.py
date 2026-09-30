"""Resumable preference-data stages shared by original ACC-Collab Actor and Critic."""

from __future__ import annotations

# ``_utils`` bootstraps the repository and offline environment before project imports.
# ruff: noqa: E402

import argparse
import hashlib
import itertools
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from _utils import (
    add_config_arguments,
    batched,
    expected_batches,
    generation_audit_summary,
    load_config,
    setup_logging,
)
from src.acccollab.config import ACCCollabConfig
from src.acccollab.data import expand_preference_trials, load_split_samples, shard_samples
from src.acccollab.io import (
    count_jsonl,
    iter_jsonl,
    merge_sorted_jsonl,
    read_json,
    write_json,
)
from src.acccollab.pairs import summarize_pairs, validate_pair_count
from src.acccollab.policy import build_policy_bundle
from src.acccollab.prompts import specialize_sample
from src.acccollab.registry import PolicyState, actor_data_state, critic_data_state
from src.acccollab.stages import (
    policy_state_identity,
    preference_shard_fingerprint,
    preference_stage_fingerprint,
    validate_stage_success,
    write_stage_success,
)
from src.acccollab.trajectories import (
    TrajectorySettings,
    generate_actor_preference_batch,
    generate_critic_preference_batch,
)
from src.inference.vllm_server import inference_prompt_format_version
from src.utils.checkpoints import JsonlBatchCheckpoint
from src.utils.generation_audit import enforce_generation_assessment, subtract_generation_stats

logger = logging.getLogger(__name__)


def preference_stage_name(agent: str) -> str:
    """Return the durable stage name used in checkpoints and success markers."""
    _validate_agent(agent)
    return f"{agent}_dpo_data"


def preference_shard_dir(
    config: ACCCollabConfig,
    *,
    agent: str,
    iteration: int,
    shard_idx: int,
    num_shards: int,
) -> Path:
    """Return one materialized shard directory."""
    return (
        config.paths.data_dir(iteration, agent)
        / "shards"
        / f"shard-{shard_idx:03d}-of-{num_shards:03d}"
    )


def load_preference_samples(config: ACCCollabConfig) -> list[dict[str, Any]]:
    """Load the deterministic preference split selected by the strict config."""
    split = config.data.preference
    samples = load_split_samples(
        dataset_name=config.data.dataset,
        split=split.split,
        max_samples=split.samples,
        strategy=split.strategy,
        seed=(
            int(split.sampling_seed)
            if split.sampling_seed is not None
            else int(config.run.seed)
        ),
        mmlu_load_mode=config.data.mmlu_load_mode,
        expected_samples=split.expected_samples,
        benchmark_data_dir=config.data.benchmark_data_dir,
    )
    role = config.prompt_role
    if role.name != "original":
        samples = [
            specialize_sample(
                sample,
                role_name=role.name,
                actor_instruction=role.actor_instruction,
                critic_instruction=role.critic_instruction,
                implementation_version=role.implementation_version,
            )
            for sample in samples
        ]
    return expand_preference_trials(
        samples,
        trials=int(config.data.preference_trials),
    )


def preference_policy_state(
    config: ACCCollabConfig,
    *,
    agent: str,
    iteration: int,
) -> PolicyState:
    """Resolve Algorithm 1's exact policy state for one data stage."""
    _validate_agent(agent)
    if agent == "critic":
        return critic_data_state(config, iteration)
    return actor_data_state(config, iteration)


def expected_preference_stage_fingerprint(
    config: ACCCollabConfig,
    *,
    agent: str,
    iteration: int,
    num_shards: int,
) -> tuple[PolicyState, str]:
    """Resolve the state and semantic fingerprint expected by a data stage."""
    state = preference_policy_state(config, agent=agent, iteration=iteration)
    fingerprint = preference_stage_fingerprint(
        config,
        agent=agent,
        iteration=iteration,
        state=state,
        num_shards=num_shards,
    )
    return state, fingerprint


def validate_merged_preference_stage(
    config: ACCCollabConfig,
    *,
    agent: str,
    iteration: int,
    num_shards: int,
) -> dict[str, Any] | None:
    """Validate a complete merged data stage against current semantic inputs."""
    _state, fingerprint = expected_preference_stage_fingerprint(
        config,
        agent=agent,
        iteration=iteration,
        num_shards=num_shards,
    )
    return validate_stage_success(
        config.paths.data_dir(iteration, agent) / "_SUCCESS",
        expected_stage=preference_stage_name(agent),
        expected_fingerprint=fingerprint,
    )


def require_merged_preference_stage(
    config: ACCCollabConfig,
    *,
    agent: str,
    iteration: int,
) -> tuple[Path, int, str]:
    """Require the current merged data marker and return pair path/sharding identity."""
    _validate_agent(agent)
    config.validate_iteration(iteration)
    output_dir = config.paths.data_dir(iteration, agent)
    marker_path = output_dir / "_SUCCESS"
    try:
        raw_marker = read_json(marker_path)
        num_shards = int(dict(raw_marker.get("metadata") or {}).get("num_shards"))
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Missing or malformed merged {preference_stage_name(agent)} marker: {marker_path}"
        ) from exc
    _state, fingerprint = expected_preference_stage_fingerprint(
        config,
        agent=agent,
        iteration=iteration,
        num_shards=num_shards,
    )
    marker = validate_stage_success(
        marker_path,
        expected_stage=preference_stage_name(agent),
        expected_fingerprint=fingerprint,
    )
    if marker is None:
        raise RuntimeError(
            f"Merged {preference_stage_name(agent)} data is stale or corrupted: {output_dir}"
        )
    pair_path = output_dir / "pairs.jsonl"
    return pair_path, num_shards, fingerprint


def build_trajectory_settings(config: ACCCollabConfig) -> TrajectorySettings:
    """Project the paper-original trajectory settings from the strict config."""
    return TrajectorySettings(
        deliberation_rounds=config.method.deliberation_rounds,
        rollouts=config.reward.rollouts,
        epsilon=config.reward.epsilon,
        actor_max_tokens=config.tokens.actor,
        critic_max_tokens=config.tokens.critic,
        temperature=config.generation.train_temperature,
        top_p=config.generation.top_p,
        actor_thinking=config.generation.thinking.train,
        critic_thinking=config.generation.thinking.train,
    )


def generate_preference_shard(
    config: ACCCollabConfig,
    *,
    agent: str,
    iteration: int,
    device: int,
    shard_idx: int,
    num_shards: int,
) -> Path:
    """Generate or resume one deterministic preference-data shard."""
    _validate_shard(shard_idx, num_shards)
    config.validate_iteration(iteration)
    samples = load_preference_samples(config)
    shard = shard_samples(samples, shard_idx=shard_idx, num_shards=num_shards)
    state, stage_fingerprint = expected_preference_stage_fingerprint(
        config,
        agent=agent,
        iteration=iteration,
        num_shards=num_shards,
    )
    batch_size = int(config.runtime.batch.preference_generation)
    shard_fingerprint = preference_shard_fingerprint(
        stage_fingerprint=stage_fingerprint,
        shard_idx=shard_idx,
        num_shards=num_shards,
        sample_ids=[str(sample["sample_id"]) for sample in shard],
        batch_size=batch_size,
    )
    stage_name = preference_stage_name(agent)
    shard_dir = preference_shard_dir(
        config,
        agent=agent,
        iteration=iteration,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )
    shard_success = validate_stage_success(
        shard_dir / "_SUCCESS",
        expected_stage=f"{stage_name}_shard",
        expected_fingerprint=shard_fingerprint,
    )
    if shard_success is not None:
        logger.info(
            "Reusing completed %s shard %d/%d (%d samples)",
            stage_name,
            shard_idx,
            num_shards,
            len(shard),
        )
        return shard_dir

    checkpoint = JsonlBatchCheckpoint(
        output_dir=config.paths.data_dir(iteration, agent),
        stage=stage_name,
        shard_idx=shard_idx,
        num_shards=num_shards,
        fingerprint=shard_fingerprint,
    )
    batch_count = expected_batches(len(shard), batch_size)
    if not checkpoint.is_complete(batch_count):
        provenance = {
            "pipeline": "acccollab_original",
            "stage": stage_name,
            "iteration": iteration,
            "stage_fingerprint": stage_fingerprint,
            "shard_fingerprint": shard_fingerprint,
            "policy_state": policy_state_identity(state),
        }
        with build_policy_bundle(
            config,
            actor_adapter=state.actor_adapter,
            critic_adapter=state.critic_adapter,
            device=device,
        ) as policies:
            generator = (
                generate_critic_preference_batch
                if agent == "critic"
                else generate_actor_preference_batch
            )
            settings = build_trajectory_settings(config)
            for batch_index, batch in batched(shard, batch_size):
                if checkpoint.is_completed(batch_index):
                    logger.info(
                        "Reusing %s checkpoint batch %d/%d for shard %d/%d",
                        stage_name,
                        batch_index,
                        batch_count,
                        shard_idx,
                        num_shards,
                    )
                    continue
                before = policies.generation_stats()
                trajectories, pairs = generator(
                    actor_policy=policies.actor,
                    critic_policy=policies.critic,
                    samples=batch,
                    dataset_name=config.data.dataset,
                    iteration=iteration,
                    settings=settings,
                    seed=_batch_seed(
                        config.run.seed,
                        stage=stage_name,
                        iteration=iteration,
                        shard_idx=shard_idx,
                        batch_index=batch_index,
                    ),
                    policy_provenance=provenance,
                )
                after = policies.generation_stats()
                trajectories.sort(key=_trajectory_key)
                pairs.sort(key=_pair_key)
                sample_indices = [int(sample["acccollab_sample_index"]) for sample in batch]
                audit = {
                    "schema_version": 1,
                    "pipeline": "acccollab_original",
                    "stage": stage_name,
                    "iteration": iteration,
                    "shard_idx": shard_idx,
                    "num_shards": num_shards,
                    "batch_index": batch_index,
                    "first_sample_index": min(sample_indices),
                    "last_sample_index": max(sample_indices),
                    **subtract_generation_stats(after, before),
                }
                checkpoint.commit(
                    batch_index,
                    {
                        "pairs": pairs,
                        "trajectories": trajectories,
                        "generation_audit": [audit],
                    },
                )
                logger.info(
                    "Generated %s shard %d/%d batch %d/%d: samples=%d pairs=%d",
                    stage_name,
                    shard_idx,
                    num_shards,
                    batch_index + 1,
                    batch_count,
                    len(batch),
                    len(pairs),
                )

    checkpoint.validate_complete(batch_count)
    shard_dir.mkdir(parents=True, exist_ok=True)
    pair_path = shard_dir / "pairs.jsonl"
    trajectory_path = shard_dir / "trajectories.jsonl"
    audit_path = shard_dir / "generation_audit.jsonl"
    pair_count = checkpoint.materialize("pairs", pair_path, expected_batches=batch_count)
    trajectory_count = checkpoint.materialize(
        "trajectories",
        trajectory_path,
        expected_batches=batch_count,
    )
    checkpoint.materialize(
        "generation_audit",
        audit_path,
        expected_batches=batch_count,
    )
    _validate_trajectory_coverage(trajectory_path, shard)
    _validate_pairs(
        pair_path,
        agent=agent,
        iteration=iteration,
        allowed_sample_indices={int(sample["acccollab_sample_index"]) for sample in shard},
    )
    if trajectory_count != len(shard):
        raise RuntimeError(
            f"{stage_name} shard trajectory count mismatch: {trajectory_count} != {len(shard)}"
        )
    if pair_count != count_jsonl(pair_path):
        raise RuntimeError(f"{stage_name} shard pair materialization count changed unexpectedly")

    audit_summary = generation_audit_summary(iter_jsonl(audit_path), config=config)
    pair_summary = summarize_pairs(iter_jsonl(pair_path))
    selection_summary = summarize_trajectory_selections(iter_jsonl(trajectory_path))
    _validate_selection_summary(
        selection_summary,
        expected_rounds=len(shard) * config.pair_rounds,
        expected_pairs=pair_count,
        stage_name=f"{stage_name} shard {shard_idx}/{num_shards}",
    )
    metrics = _preference_metrics(
        config,
        agent=agent,
        iteration=iteration,
        scope="shard",
        fingerprint=shard_fingerprint,
        stage_fingerprint=stage_fingerprint,
        sample_count=len(shard),
        unique_sample_count=_unique_source_sample_count(shard),
        pair_summary=pair_summary,
        selection_summary=selection_summary,
        audit_summary=audit_summary,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )
    metrics_path = shard_dir / "metrics.json"
    write_json(metrics_path, metrics)
    enforce_generation_assessment(
        audit_summary,
        fail_on_excess=config.generation.truncation.fail_on_excess,
    )
    write_stage_success(
        shard_dir / "_SUCCESS",
        stage=f"{stage_name}_shard",
        fingerprint=shard_fingerprint,
        artifacts={
            "pairs": pair_path,
            "trajectories": trajectory_path,
            "generation_audit": audit_path,
            "metrics": metrics_path,
        },
        metadata={
            "agent": agent,
            "iteration": iteration,
            "shard_idx": shard_idx,
            "num_shards": num_shards,
            "samples": len(shard),
            "unique_samples": _unique_source_sample_count(shard),
            "preference_trials": config.data.preference_trials,
            "pairs": pair_count,
            "stage_fingerprint": stage_fingerprint,
        },
    )
    return shard_dir


def merge_preference_shards(
    config: ACCCollabConfig,
    *,
    agent: str,
    iteration: int,
    num_shards: int,
) -> Path:
    """Validate and stream-merge all shards, then apply full-stage quality gates."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    config.validate_iteration(iteration)
    samples = load_preference_samples(config)
    state, stage_fingerprint = expected_preference_stage_fingerprint(
        config,
        agent=agent,
        iteration=iteration,
        num_shards=num_shards,
    )
    stage_name = preference_stage_name(agent)
    output_dir = config.paths.data_dir(iteration, agent)
    existing = validate_stage_success(
        output_dir / "_SUCCESS",
        expected_stage=stage_name,
        expected_fingerprint=stage_fingerprint,
    )
    if existing is not None:
        logger.info("Reusing completed merged %s stage", stage_name)
        return output_dir

    pair_paths: list[Path] = []
    trajectory_paths: list[Path] = []
    audit_paths: list[Path] = []
    shard_metrics: list[dict[str, Any]] = []
    batch_size = int(config.runtime.batch.preference_generation)
    for shard_idx in range(num_shards):
        expected_shard = shard_samples(
            samples,
            shard_idx=shard_idx,
            num_shards=num_shards,
        )
        shard_fingerprint = preference_shard_fingerprint(
            stage_fingerprint=stage_fingerprint,
            shard_idx=shard_idx,
            num_shards=num_shards,
            sample_ids=[str(sample["sample_id"]) for sample in expected_shard],
            batch_size=batch_size,
        )
        shard_dir = preference_shard_dir(
            config,
            agent=agent,
            iteration=iteration,
            shard_idx=shard_idx,
            num_shards=num_shards,
        )
        marker = validate_stage_success(
            shard_dir / "_SUCCESS",
            expected_stage=f"{stage_name}_shard",
            expected_fingerprint=shard_fingerprint,
        )
        if marker is None:
            raise RuntimeError(
                f"Cannot merge {stage_name}; shard {shard_idx}/{num_shards} is missing, "
                "stale, or corrupted"
            )
        pair_paths.append(shard_dir / "pairs.jsonl")
        trajectory_paths.append(shard_dir / "trajectories.jsonl")
        audit_paths.append(shard_dir / "generation_audit.jsonl")
        metadata = dict(marker.get("metadata") or {})
        shard_metrics.append(
            {
                "shard_idx": shard_idx,
                "samples": int(metadata.get("samples", len(expected_shard))),
                "pairs": int(metadata.get("pairs", 0)),
                "fingerprint": shard_fingerprint,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / "pairs.jsonl"
    trajectory_path = output_dir / "trajectories.jsonl"
    audit_path = output_dir / "generation_audit.jsonl"
    pair_count = merge_sorted_jsonl(pair_paths, pair_path, key=_pair_key)
    trajectory_count = merge_sorted_jsonl(
        trajectory_paths,
        trajectory_path,
        key=_trajectory_key,
    )
    merge_sorted_jsonl(audit_paths, audit_path, key=_audit_key)

    _validate_trajectory_coverage(trajectory_path, samples)
    _validate_pairs(
        pair_path,
        agent=agent,
        iteration=iteration,
        allowed_sample_indices={int(sample["acccollab_sample_index"]) for sample in samples},
    )
    if trajectory_count != len(samples):
        raise RuntimeError(
            f"{stage_name} merged trajectory count mismatch: {trajectory_count} != {len(samples)}"
        )
    pair_summary = summarize_pairs(iter_jsonl(pair_path))
    if int(pair_summary["total_pairs"]) != pair_count:
        raise RuntimeError(f"{stage_name} merged pair count changed during validation")
    selection_summary = summarize_trajectory_selections(iter_jsonl(trajectory_path))
    _validate_selection_summary(
        selection_summary,
        expected_rounds=len(samples) * config.pair_rounds,
        expected_pairs=pair_count,
        stage_name=f"{stage_name} merged",
    )
    audit_summary = generation_audit_summary(iter_jsonl(audit_path), config=config)
    metrics = _preference_metrics(
        config,
        agent=agent,
        iteration=iteration,
        scope="merged",
        fingerprint=stage_fingerprint,
        stage_fingerprint=stage_fingerprint,
        sample_count=len(samples),
        unique_sample_count=_unique_source_sample_count(samples),
        pair_summary=pair_summary,
        selection_summary=selection_summary,
        audit_summary=audit_summary,
        shard_idx=None,
        num_shards=num_shards,
    )
    metrics.update(
        policy_state=policy_state_identity(state),
        shards=shard_metrics,
        quality_gate={
            "minimum_pairs": config.reward.min_pairs_per_stage,
            "actual_pairs": pair_count,
            "passed": pair_count >= config.reward.min_pairs_per_stage,
        },
    )
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)

    validate_pair_count(
        pair_count,
        minimum=config.reward.min_pairs_per_stage,
        agent=agent,
    )
    enforce_generation_assessment(
        audit_summary,
        fail_on_excess=config.generation.truncation.fail_on_excess,
    )
    write_stage_success(
        output_dir / "_SUCCESS",
        stage=stage_name,
        fingerprint=stage_fingerprint,
        artifacts={
            "pairs": pair_path,
            "trajectories": trajectory_path,
            "generation_audit": audit_path,
            "metrics": metrics_path,
        },
        metadata={
            "agent": agent,
            "iteration": iteration,
            "num_shards": num_shards,
            "samples": len(samples),
            "unique_samples": _unique_source_sample_count(samples),
            "preference_trials": config.data.preference_trials,
            "pairs": pair_count,
            "policy_state": policy_state_identity(state),
        },
    )
    logger.info(
        "Merged %s: samples=%d pairs=%d shards=%d",
        stage_name,
        len(samples),
        pair_count,
        num_shards,
    )
    return output_dir


def run_preference_stage(
    config: ACCCollabConfig,
    *,
    agent: str,
    iteration: int,
    device: int | None,
    shard_idx: int,
    num_shards: int,
    merge_shards: int | None,
) -> Path:
    """Execute a worker/merge invocation with single-GPU auto-materialization."""
    _validate_agent(agent)
    config.validate_iteration(iteration)
    if merge_shards is not None:
        return merge_preference_shards(
            config,
            agent=agent,
            iteration=iteration,
            num_shards=int(merge_shards),
        )

    _validate_shard(shard_idx, num_shards)
    selected_device = (
        int(device)
        if device is not None
        else int(config.runtime.generation_devices[0])
    )
    _state, stage_fingerprint = expected_preference_stage_fingerprint(
        config,
        agent=agent,
        iteration=iteration,
        num_shards=num_shards,
    )
    output_dir = config.paths.data_dir(iteration, agent)
    if (
        validate_stage_success(
            output_dir / "_SUCCESS",
            expected_stage=preference_stage_name(agent),
            expected_fingerprint=stage_fingerprint,
        )
        is not None
    ):
        logger.info("The complete %s stage already matches the current config", agent)
        return output_dir

    generate_preference_shard(
        config,
        agent=agent,
        iteration=iteration,
        device=selected_device,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )
    if num_shards == 1 and shard_idx == 0:
        return merge_preference_shards(
            config,
            agent=agent,
            iteration=iteration,
            num_shards=1,
        )
    return preference_shard_dir(
        config,
        agent=agent,
        iteration=iteration,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )


def build_parser(agent: str) -> argparse.ArgumentParser:
    """Build the shared CLI for an Actor or Critic preference-data entry point."""
    _validate_agent(agent)
    parser = argparse.ArgumentParser(
        description=f"Build paper-original ACC-Collab {agent.title()} DPO data"
    )
    add_config_arguments(parser)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--merge-shards",
        type=int,
        default=None,
        metavar="N",
        help="Validate and merge N completed shards without loading a model.",
    )
    return parser


def main(agent: str) -> None:
    """CLI entry point used by the numbered Actor/Critic wrappers."""
    args = build_parser(agent).parse_args()
    config = load_config(args)
    setup_logging(seed=config.run.seed)
    run_preference_stage(
        config,
        agent=agent,
        iteration=int(args.iteration),
        device=args.device,
        shard_idx=int(args.shard_idx),
        num_shards=int(args.num_shards),
        merge_shards=args.merge_shards,
    )


def _preference_metrics(
    config: ACCCollabConfig,
    *,
    agent: str,
    iteration: int,
    scope: str,
    fingerprint: str,
    stage_fingerprint: str,
    sample_count: int,
    unique_sample_count: int,
    pair_summary: Mapping[str, Any],
    selection_summary: Mapping[str, Any],
    audit_summary: Mapping[str, Any],
    shard_idx: int | None,
    num_shards: int,
) -> dict[str, Any]:
    total_pairs = int(pair_summary.get("total_pairs", 0))
    maximum_pairs = sample_count * config.pair_rounds
    return {
        "schema_version": 1,
        "pipeline": "acccollab_original",
        "stage": preference_stage_name(agent),
        "scope": scope,
        "agent": agent,
        "iteration": iteration,
        "fingerprint": fingerprint,
        "stage_fingerprint": stage_fingerprint,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "samples": sample_count,
        "trajectory_samples": sample_count,
        "unique_samples": unique_sample_count,
        "preference_trials": config.data.preference_trials,
        "deliberation_rounds": config.method.deliberation_rounds,
        "pair_rounds": config.pair_rounds,
        "maximum_pairs": maximum_pairs,
        "pair_yield": total_pairs / maximum_pairs if maximum_pairs else 0.0,
        "pairs": dict(pair_summary),
        "selection": dict(selection_summary),
        "generation_audit": dict(audit_summary),
        "protocol": {
            "agents": ["actor", "critic"],
            "reward_estimator": "one_step_mc",
            "reward_rollouts": config.reward.rollouts,
            "pair_rule": "paper_eq5_if_elif",
            "guided_positive_branch_precedence": True,
            "natural_trajectory_spine": True,
            "drops_identical_chosen_rejected": True,
            "preference_trials": config.data.preference_trials,
            "inference_prompt_format_version": inference_prompt_format_version(
                config.model.type,
                config.model.name,
            ),
        },
    }


def _unique_source_sample_count(samples: Sequence[Mapping[str, Any]]) -> int:
    return len(
        {
            str(sample.get("source_sample_id") or sample.get("sample_id") or "")
            for sample in samples
        }
    )


def summarize_trajectory_selections(
    trajectories: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize all t=1..4 Eq. 5 decisions, including defensive drops."""
    total_rounds = 0
    selected_rounds = 0
    threshold_passed_rounds = 0
    drop_reasons: Counter[str] = Counter()
    for trajectory in trajectories:
        for round_record in list(trajectory.get("rounds") or []):
            if "eq5_selection" not in round_record:
                continue
            selection = round_record.get("eq5_selection")
            if not isinstance(selection, Mapping):
                raise RuntimeError("Trajectory eq5_selection must be a mapping")
            total_rounds += 1
            selected = selection.get("selected")
            if selected is True:
                selected_rounds += 1
            elif selected is False:
                reason = str(selection.get("drop_reason") or "unknown")
                drop_reasons[reason] += 1
            else:
                raise RuntimeError("Trajectory eq5_selection.selected must be boolean")
            if selection.get("eq5_threshold_passed") is True:
                threshold_passed_rounds += 1

    dropped_rounds = total_rounds - selected_rounds
    return {
        "total_rounds": total_rounds,
        "selected_rounds": selected_rounds,
        "dropped_rounds": dropped_rounds,
        "threshold_passed_rounds": threshold_passed_rounds,
        "drop_reason_counts": dict(sorted(drop_reasons.items())),
    }


def _validate_selection_summary(
    summary: Mapping[str, Any],
    *,
    expected_rounds: int,
    expected_pairs: int,
    stage_name: str,
) -> None:
    total = int(summary.get("total_rounds", -1))
    selected = int(summary.get("selected_rounds", -1))
    dropped = int(summary.get("dropped_rounds", -1))
    if total != expected_rounds:
        raise RuntimeError(
            f"{stage_name} Eq. 5 round coverage mismatch: {total} != {expected_rounds}"
        )
    if selected != expected_pairs:
        raise RuntimeError(
            f"{stage_name} selected-round/pair mismatch: {selected} != {expected_pairs}"
        )
    if selected + dropped != total:
        raise RuntimeError(
            f"{stage_name} Eq. 5 selection accounting mismatch: "
            f"{selected} + {dropped} != {total}"
        )


def _validate_trajectory_coverage(
    path: str | Path,
    expected_samples: Sequence[Mapping[str, Any]],
) -> None:
    sentinel = object()
    rows = iter_jsonl(path)
    for position, pair in enumerate(
        itertools.zip_longest(expected_samples, rows, fillvalue=sentinel)
    ):
        expected, row = pair
        if expected is sentinel:
            raise RuntimeError(f"Unexpected extra trajectory at {path}:{position + 1}")
        if row is sentinel:
            raise RuntimeError(
                f"Missing trajectory for sample {expected['sample_id']!r} in {path}"
            )
        assert isinstance(expected, Mapping)
        assert isinstance(row, Mapping)
        expected_id = str(expected["sample_id"])
        expected_index = int(expected["acccollab_sample_index"])
        actual_sample = dict(row.get("sample") or {})
        actual_id = str(row.get("sample_id") or actual_sample.get("sample_id") or "")
        actual_index = int(actual_sample.get("acccollab_sample_index", -1))
        if actual_id != expected_id or actual_index != expected_index:
            raise RuntimeError(
                "Trajectory coverage/order mismatch at position "
                f"{position}: expected=({expected_index}, {expected_id!r}), "
                f"actual=({actual_index}, {actual_id!r})"
            )


def _validate_pairs(
    path: str | Path,
    *,
    agent: str,
    iteration: int,
    allowed_sample_indices: set[int],
) -> None:
    previous: tuple[int, int] | None = None
    for line_number, pair in enumerate(iter_jsonl(path), start=1):
        metadata = dict(pair.get("metadata") or {})
        key = _pair_key(pair)
        if previous is not None and key <= previous:
            raise RuntimeError(
                f"Pair order/uniqueness violation at {path}:{line_number}: {key} <= {previous}"
            )
        previous = key
        if metadata.get("agent") != agent or int(metadata.get("iteration", -1)) != iteration:
            raise RuntimeError(f"Pair provenance mismatch at {path}:{line_number}")
        if key[0] not in allowed_sample_indices:
            raise RuntimeError(f"Pair references an unexpected sample at {path}:{line_number}")
        if key[1] not in range(1, 5):
            raise RuntimeError(f"Pair round must be 1..4 at {path}:{line_number}")
        for field in ("prompt", "chosen", "rejected"):
            if not str(pair.get(field) or "").strip():
                raise RuntimeError(f"Pair has empty {field} at {path}:{line_number}")
        if str(pair["chosen"]).strip() == str(pair["rejected"]).strip():
            raise RuntimeError(
                f"Pair has identical chosen/rejected completions at {path}:{line_number}"
            )


def _pair_key(row: Mapping[str, Any]) -> tuple[int, int]:
    metadata = dict(row.get("metadata") or {})
    return int(metadata.get("sample_index", -1)), int(metadata.get("round", -1))


def _trajectory_key(row: Mapping[str, Any]) -> int:
    sample = dict(row.get("sample") or {})
    return int(sample.get("acccollab_sample_index", -1))


def _audit_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("first_sample_index", -1)),
        int(row.get("shard_idx", -1)),
        int(row.get("batch_index", -1)),
    )


def _batch_seed(
    base_seed: int,
    *,
    stage: str,
    iteration: int,
    shard_idx: int,
    batch_index: int,
) -> int:
    encoded = f"{base_seed}:{stage}:{iteration}:{shard_idx}:{batch_index}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big") & 0x7FFFFFFF


def _validate_agent(agent: str) -> None:
    if agent not in {"actor", "critic"}:
        raise ValueError(f"agent must be actor or critic, got {agent!r}")


def _validate_shard(shard_idx: int, num_shards: int) -> None:
    if num_shards < 1 or shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(
            f"Invalid shard placement: shard_idx={shard_idx}, num_shards={num_shards}"
        )
