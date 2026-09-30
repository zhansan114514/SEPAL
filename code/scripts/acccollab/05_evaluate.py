"""Five-round final-Actor-only evaluation for paper-original ACC-Collab."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import itertools
import logging
from collections.abc import Mapping, Sequence
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
from src.acccollab.data import load_split_samples, shard_samples
from src.acccollab.evaluation import (
    EvaluationSettings,
    aggregate_trial_metrics,
    compare_with_paper,
    generate_evaluation_batch,
    score_evaluation_records,
)
from src.acccollab.io import iter_jsonl, merge_sorted_jsonl, write_json
from src.acccollab.policy import build_policy_bundle
from src.acccollab.prompts import specialize_sample
from src.acccollab.registry import evaluation_state
from src.acccollab.stages import (
    evaluation_shard_fingerprint,
    evaluation_stage_fingerprint,
    policy_state_identity,
    validate_stage_success,
    write_stage_success,
)
from src.utils.checkpoints import JsonlBatchCheckpoint
from src.utils.generation_audit import enforce_generation_assessment, subtract_generation_stats

logger = logging.getLogger(__name__)


def eval_shard_dir(config, *, shard_idx: int, num_shards: int) -> Path:
    """Return one materialized evaluation shard directory."""
    return (
        config.paths.eval_dir
        / "shards"
        / f"shard-{shard_idx:03d}-of-{num_shards:03d}"
    )


def load_eval_samples(config) -> list[dict[str, Any]]:
    """Load the exact configured evaluation split and stable sample order."""
    split = config.data.eval
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
    if role.name == "original":
        return samples
    return [
        specialize_sample(
            sample,
            role_name=role.name,
            actor_instruction=role.actor_instruction,
            critic_instruction=role.critic_instruction,
            implementation_version=role.implementation_version,
        )
        for sample in samples
    ]


def evaluation_settings(config) -> EvaluationSettings:
    """Project strict config values into the five-round evaluation protocol."""
    return EvaluationSettings(
        deliberation_rounds=config.method.deliberation_rounds,
        actor_max_tokens=config.tokens.actor,
        critic_max_tokens=config.tokens.critic,
        temperature=config.generation.eval_temperature,
        top_p=config.generation.top_p,
        actor_thinking=config.generation.thinking.eval,
        critic_thinking=config.generation.thinking.eval,
    )


def expected_evaluation_identity(config, *, num_shards: int):
    """Return final policy state and evaluation-stage fingerprint."""
    state = evaluation_state(config)
    fingerprint = evaluation_stage_fingerprint(
        config,
        state=state,
        num_shards=num_shards,
    )
    return state, fingerprint


def validate_merged_evaluation(config, *, num_shards: int) -> dict[str, Any] | None:
    """Validate the complete evaluation marker for a particular shard layout."""
    _state, fingerprint = expected_evaluation_identity(config, num_shards=num_shards)
    return validate_stage_success(
        config.paths.eval_dir / "_SUCCESS",
        expected_stage="evaluate",
        expected_fingerprint=fingerprint,
    )


def generate_evaluation_shard(
    config,
    *,
    device: int,
    shard_idx: int,
    num_shards: int,
) -> Path:
    """Generate or resume one evaluation shard."""
    _validate_shard(shard_idx, num_shards)
    samples = load_eval_samples(config)
    shard = shard_samples(samples, shard_idx=shard_idx, num_shards=num_shards)
    state, stage_fingerprint = expected_evaluation_identity(config, num_shards=num_shards)
    batch_size = int(config.runtime.batch.evaluation)
    sample_ids = [str(sample["sample_id"]) for sample in shard]
    shard_fingerprint = evaluation_shard_fingerprint(
        stage_fingerprint=stage_fingerprint,
        shard_idx=shard_idx,
        num_shards=num_shards,
        sample_ids=sample_ids,
        batch_size=batch_size,
        trials=config.evaluation.trials,
    )
    shard_dir = eval_shard_dir(config, shard_idx=shard_idx, num_shards=num_shards)
    if (
        validate_stage_success(
            shard_dir / "_SUCCESS",
            expected_stage="evaluate_shard",
            expected_fingerprint=shard_fingerprint,
        )
        is not None
    ):
        logger.info("Reusing completed evaluation shard %d/%d", shard_idx, num_shards)
        return shard_dir

    batches_per_trial = expected_batches(len(shard), batch_size)
    total_batches = config.evaluation.trials * batches_per_trial
    checkpoint = JsonlBatchCheckpoint(
        output_dir=config.paths.eval_dir,
        stage="evaluate",
        shard_idx=shard_idx,
        num_shards=num_shards,
        fingerprint=shard_fingerprint,
    )
    if not checkpoint.is_complete(total_batches):
        settings = evaluation_settings(config)
        provenance = {
            "pipeline": "acccollab_original",
            "stage": "evaluate",
            "stage_fingerprint": stage_fingerprint,
            "shard_fingerprint": shard_fingerprint,
            "policy_state": policy_state_identity(state),
            "decision_rule": "single_final_actor_answer",
            "uses_majority_vote": False,
            "uses_judge_fallback": False,
        }
        with build_policy_bundle(
            config,
            actor_adapter=state.actor_adapter,
            critic_adapter=state.critic_adapter,
            device=device,
        ) as policies:
            for trial_index in range(config.evaluation.trials):
                for batch_index, batch in batched(shard, batch_size):
                    checkpoint_index = trial_index * batches_per_trial + batch_index
                    if checkpoint.is_completed(checkpoint_index):
                        logger.info(
                            "Reusing evaluation checkpoint trial=%d batch=%d shard=%d/%d",
                            trial_index,
                            batch_index,
                            shard_idx,
                            num_shards,
                        )
                        continue
                    before = policies.generation_stats()
                    records = generate_evaluation_batch(
                        actor_policy=policies.actor,
                        critic_policy=policies.critic,
                        samples=batch,
                        dataset_name=config.data.dataset,
                        trial_index=trial_index,
                        settings=settings,
                        seed=_batch_seed(
                            config.run.seed,
                            trial_index=trial_index,
                            shard_idx=shard_idx,
                            batch_index=batch_index,
                        ),
                        policy_provenance=provenance,
                    )
                    after = policies.generation_stats()
                    records.sort(key=_evaluation_key)
                    sample_indices = [int(sample["acccollab_sample_index"]) for sample in batch]
                    audit = {
                        "schema_version": 1,
                        "pipeline": "acccollab_original",
                        "stage": "evaluate",
                        "trial": trial_index,
                        "shard_idx": shard_idx,
                        "num_shards": num_shards,
                        "batch_index": batch_index,
                        "checkpoint_index": checkpoint_index,
                        "first_sample_index": min(sample_indices),
                        "last_sample_index": max(sample_indices),
                        **subtract_generation_stats(after, before),
                    }
                    checkpoint.commit(
                        checkpoint_index,
                        {"records": records, "generation_audit": [audit]},
                    )
                    logger.info(
                        "Evaluated shard %d/%d trial %d/%d batch %d/%d",
                        shard_idx,
                        num_shards,
                        trial_index + 1,
                        config.evaluation.trials,
                        batch_index + 1,
                        batches_per_trial,
                    )

    checkpoint.validate_complete(total_batches)
    shard_dir.mkdir(parents=True, exist_ok=True)
    records_path = shard_dir / "records.jsonl"
    audit_path = shard_dir / "generation_audit.jsonl"
    checkpoint.materialize("records", records_path, expected_batches=total_batches)
    checkpoint.materialize("generation_audit", audit_path, expected_batches=total_batches)
    shard_trial_metrics = _score_trial_groups(
        records_path,
        expected_sample_ids=sample_ids,
        trials=config.evaluation.trials,
    )
    audit_summary = generation_audit_summary(iter_jsonl(audit_path), config=config)
    metrics = {
        "schema_version": 1,
        "pipeline": "acccollab_original",
        "stage": "evaluate_shard",
        "scope": "shard",
        "fingerprint": shard_fingerprint,
        "stage_fingerprint": stage_fingerprint,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "samples": len(shard),
        "trials": config.evaluation.trials,
        "trial_metrics": shard_trial_metrics,
        "generation_audit": audit_summary,
        "policy_state": policy_state_identity(state),
        "decision_rule": "single_final_actor_answer",
        "uses_majority_vote": False,
        "uses_judge_fallback": False,
    }
    metrics_path = shard_dir / "metrics.json"
    write_json(metrics_path, metrics)
    enforce_generation_assessment(
        audit_summary,
        fail_on_excess=config.generation.truncation.fail_on_excess,
    )
    write_stage_success(
        shard_dir / "_SUCCESS",
        stage="evaluate_shard",
        fingerprint=shard_fingerprint,
        artifacts={
            "records": records_path,
            "generation_audit": audit_path,
            "metrics": metrics_path,
        },
        metadata={
            "shard_idx": shard_idx,
            "num_shards": num_shards,
            "samples": len(shard),
            "trials": config.evaluation.trials,
            "stage_fingerprint": stage_fingerprint,
        },
    )
    return shard_dir


def merge_evaluation_shards(config, *, num_shards: int) -> Path:
    """Merge all evaluation shards and write trial, aggregate, and paper metrics."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    samples = load_eval_samples(config)
    state, stage_fingerprint = expected_evaluation_identity(config, num_shards=num_shards)
    output_dir = config.paths.eval_dir
    existing = validate_stage_success(
        output_dir / "_SUCCESS",
        expected_stage="evaluate",
        expected_fingerprint=stage_fingerprint,
    )
    if existing is not None:
        logger.info("Reusing completed evaluation stage")
        return output_dir

    record_paths: list[Path] = []
    audit_paths: list[Path] = []
    shard_metadata: list[dict[str, Any]] = []
    batch_size = int(config.runtime.batch.evaluation)
    for shard_idx in range(num_shards):
        expected_shard = shard_samples(
            samples,
            shard_idx=shard_idx,
            num_shards=num_shards,
        )
        shard_fingerprint = evaluation_shard_fingerprint(
            stage_fingerprint=stage_fingerprint,
            shard_idx=shard_idx,
            num_shards=num_shards,
            sample_ids=[str(sample["sample_id"]) for sample in expected_shard],
            batch_size=batch_size,
            trials=config.evaluation.trials,
        )
        shard_dir = eval_shard_dir(config, shard_idx=shard_idx, num_shards=num_shards)
        marker = validate_stage_success(
            shard_dir / "_SUCCESS",
            expected_stage="evaluate_shard",
            expected_fingerprint=shard_fingerprint,
        )
        if marker is None:
            raise RuntimeError(
                f"Cannot merge evaluation; shard {shard_idx}/{num_shards} is missing, "
                "stale, or corrupted"
            )
        record_paths.append(shard_dir / "records.jsonl")
        audit_paths.append(shard_dir / "generation_audit.jsonl")
        shard_metadata.append(dict(marker.get("metadata") or {}))

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    audit_path = output_dir / "generation_audit.jsonl"
    merge_sorted_jsonl(record_paths, records_path, key=_evaluation_key)
    merge_sorted_jsonl(audit_paths, audit_path, key=_audit_key)
    expected_ids = [str(sample["sample_id"]) for sample in samples]
    trial_metrics = _score_trial_groups(
        records_path,
        expected_sample_ids=expected_ids,
        trials=config.evaluation.trials,
    )
    aggregate = aggregate_trial_metrics(trial_metrics)
    comparison = compare_with_paper(
        aggregate,
        dataset_name=config.data.dataset,
        model_type=config.model.type,
        alternating_iterations=config.method.alternating_iterations,
    )
    if comparison is not None:
        comparison["config"] = {
            "dataset": config.data.dataset,
            "model": config.model.name,
            "evaluation_samples": len(samples),
            "trials": config.evaluation.trials,
            "policy_output_dir": config.evaluation.policy_output_dir,
        }
    audit_summary = generation_audit_summary(iter_jsonl(audit_path), config=config)

    trial_paths: list[Path] = []
    for trial_index, metric in enumerate(trial_metrics):
        trial_path = output_dir / f"trial_{trial_index:02d}_metrics.json"
        write_json(trial_path, metric)
        trial_paths.append(trial_path)
    aggregate_path = output_dir / "aggregate_metrics.json"
    comparison_path = output_dir / "paper_comparison.json"
    metrics_path = output_dir / "metrics.json"
    write_json(aggregate_path, aggregate)
    if comparison is not None:
        write_json(comparison_path, comparison)
    else:
        comparison_path.unlink(missing_ok=True)
    metrics = {
        "schema_version": 1,
        "pipeline": "acccollab_original",
        "stage": "evaluate",
        "scope": "merged",
        "fingerprint": stage_fingerprint,
        "num_shards": num_shards,
        "samples": len(samples),
        "trials": config.evaluation.trials,
        "policy_state": policy_state_identity(state),
        "decision_rule": "single_final_actor_answer",
        "uses_majority_vote": False,
        "uses_judge_fallback": False,
        "headline_metric": {
            "name": "round_4_single_actor_accuracy",
            "value": aggregate["final_accuracy"],
        },
        "aggregate": aggregate,
        "paper_comparison": comparison,
        "generation_audit": audit_summary,
        "shards": shard_metadata,
    }
    write_json(metrics_path, metrics)
    enforce_generation_assessment(
        audit_summary,
        fail_on_excess=config.generation.truncation.fail_on_excess,
    )
    artifacts = {
        "records": records_path,
        "generation_audit": audit_path,
        "aggregate_metrics": aggregate_path,
        "metrics": metrics_path,
    }
    if comparison is not None:
        artifacts["paper_comparison"] = comparison_path
    artifacts.update(
        {f"trial_{index:02d}_metrics": path for index, path in enumerate(trial_paths)}
    )
    write_stage_success(
        output_dir / "_SUCCESS",
        stage="evaluate",
        fingerprint=stage_fingerprint,
        artifacts=artifacts,
        metadata={
            "num_shards": num_shards,
            "samples": len(samples),
            "trials": config.evaluation.trials,
            "headline_accuracy": aggregate["final_accuracy"]["mean"],
        },
    )
    if comparison is None:
        logger.info(
            "Evaluation complete: samples=%d trials=%d final_accuracy=%.6f "
            "(no matching paper reference)",
            len(samples),
            config.evaluation.trials,
            float(aggregate["final_accuracy"]["mean"]),
        )
    else:
        logger.info(
            "Evaluation complete: samples=%d trials=%d "
            "final_accuracy=%.6f paper_delta=%+.6f",
            len(samples),
            config.evaluation.trials,
            float(aggregate["final_accuracy"]["mean"]),
            float(comparison["absolute_delta"]),
        )
    return output_dir


def run_evaluation_stage(
    config,
    *,
    device: int | None,
    shard_idx: int,
    num_shards: int,
    merge_shards: int | None,
) -> Path:
    """Execute an evaluation worker or merge invocation."""
    if merge_shards is not None:
        return merge_evaluation_shards(config, num_shards=int(merge_shards))
    _validate_shard(shard_idx, num_shards)
    _state, fingerprint = expected_evaluation_identity(config, num_shards=num_shards)
    if validate_stage_success(
        config.paths.eval_dir / "_SUCCESS",
        expected_stage="evaluate",
        expected_fingerprint=fingerprint,
    ) is not None:
        logger.info("The complete evaluation stage already matches the current config")
        return config.paths.eval_dir
    selected_device = (
        int(device) if device is not None else int(config.runtime.generation_devices[0])
    )
    generate_evaluation_shard(
        config,
        device=selected_device,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )
    if num_shards == 1 and shard_idx == 0:
        return merge_evaluation_shards(config, num_shards=1)
    return eval_shard_dir(config, shard_idx=shard_idx, num_shards=num_shards)


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI."""
    parser = argparse.ArgumentParser(description="Evaluate paper-original ACC-Collab")
    add_config_arguments(parser)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--merge-shards",
        type=int,
        default=None,
        metavar="N",
        help="Validate and merge N completed evaluation shards without a GPU.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    config = load_config(args)
    setup_logging(seed=config.run.seed)
    run_evaluation_stage(
        config,
        device=args.device,
        shard_idx=int(args.shard_idx),
        num_shards=int(args.num_shards),
        merge_shards=args.merge_shards,
    )


def _score_trial_groups(
    path: str | Path,
    *,
    expected_sample_ids: Sequence[str],
    trials: int,
) -> list[dict[str, Any]]:
    """Score a trial-major JSONL stream without buffering all trials at once."""
    result: dict[int, dict[str, Any]] = {}
    iterator = iter_jsonl(path)
    for trial_any, group in itertools.groupby(iterator, key=lambda row: int(row.get("trial", -1))):
        trial = int(trial_any)
        if trial in result or trial < 0 or trial >= trials:
            raise RuntimeError(f"Unexpected or duplicate evaluation trial {trial} in {path}")
        result[trial] = score_evaluation_records(
            group,
            deliberation_rounds=5,
            expected_sample_ids=expected_sample_ids,
        )
    # An empty shard still has a well-defined zero-sample score for every trial.
    if not expected_sample_ids:
        for trial in range(trials):
            result.setdefault(
                trial,
                score_evaluation_records(
                    (),
                    deliberation_rounds=5,
                    expected_sample_ids=(),
                ),
            )
    missing = sorted(set(range(trials)) - set(result))
    if missing:
        raise RuntimeError(f"Evaluation records are missing trials {missing} in {path}")
    return [result[index] for index in range(trials)]


def _evaluation_key(row: Mapping[str, Any]) -> tuple[int, int]:
    sample = dict(row.get("sample") or {})
    return int(row.get("trial", -1)), int(sample.get("acccollab_sample_index", -1))


def _audit_key(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("trial", -1)),
        int(row.get("first_sample_index", -1)),
        int(row.get("shard_idx", -1)),
        int(row.get("batch_index", -1)),
    )


def _batch_seed(
    base_seed: int,
    *,
    trial_index: int,
    shard_idx: int,
    batch_index: int,
) -> int:
    encoded = f"{base_seed}:evaluate:{trial_index}:{shard_idx}:{batch_index}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big") & 0x7FFFFFFF


def _validate_shard(shard_idx: int, num_shards: int) -> None:
    if num_shards < 1 or shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(
            f"Invalid shard placement: shard_idx={shard_idx}, num_shards={num_shards}"
        )


if __name__ == "__main__":
    main()
