"""Generation and scoring for authenticated intermediate-policy ablations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

from src.acccollab.config import ACCCollabConfig, load_acccollab_config
from src.acccollab.data import load_split_samples, shard_samples
from src.acccollab.evaluation import EvaluationSettings, generate_evaluation_batch
from src.acccollab.generation import generate_actor_records, make_actor_record
from src.acccollab.io import iter_jsonl, merge_sorted_jsonl, write_json, write_jsonl
from src.acccollab.policy import build_policy_bundle
from src.acccollab.prompts import build_initial_actor_prompt, specialize_sample
from src.ablations.config import (
    ABLATION_MANIFEST_VERSION,
    POLICY_VARIANT_DATASETS,
    POLICY_VARIANTS,
    ROLE_NAMES,
    load_ablation_manifest,
    resolve_manifest_path,
)
from src.evaluation.answer_resolution import answers_match, normalize_task_answer
from src.multi_acccollab.majority import resolve_majority
from src.utils.artifacts import file_sha256, stable_fingerprint
from src.utils.checkpoints import JsonlBatchCheckpoint

POLICY_LATTICE_VERSION = "multi_acccollab_policy_lattice_v1"
# Expanding the pre-registered evaluation datasets does not change any prompt,
# model, seed, sampling, or scoring behavior below.  Preserve the v1 generation
# source identity so durable SFT-only checkpoints remain reusable after this
# orchestration-only scope correction.
POLICY_LATTICE_V1_SOURCE_SHA256 = (
    "2226e636f79fcf8a288f4e8cd1fd48d67e1f07dd90b4c4e3ab31b0a24bf35d41"
)
ANSWER_EXTRACTOR_PATH = Path(__file__).resolve().parents[1] / "parsing/answer_extractor.py"
POLICY_SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).resolve().parents[1] / "acccollab/evaluation.py",
    Path(__file__).resolve().parents[1] / "acccollab/generation.py",
    Path(__file__).resolve().parents[1] / "acccollab/policy.py",
    Path(__file__).resolve().parents[1] / "acccollab/prompts.py",
    Path(__file__).resolve().parents[1] / "inference/vllm_server.py",
    ANSWER_EXTRACTOR_PATH,
)


def load_variant_context(
    manifest_path: str | Path,
    *,
    dataset_name: str,
    variant: str,
    role_name: str,
    project_root: str | Path,
) -> dict[str, Any]:
    """Resolve one role's dataset config and authenticated policy combination."""
    manifest = load_ablation_manifest(manifest_path)
    if variant not in POLICY_VARIANTS:
        raise ValueError(f"Unknown policy-lattice variant {variant!r}")
    if role_name not in ROLE_NAMES:
        raise ValueError(f"Unknown role {role_name!r}")
    datasets = dict(manifest["datasets"])
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset {dataset_name!r}")
    if dataset_name not in POLICY_VARIANT_DATASETS[variant]:
        raise ValueError(
            f"{variant} is not pre-registered for dataset {dataset_name!r}"
        )

    dataset = dict(datasets[dataset_name])
    role_dataset = dict(dict(dataset["roles"])[role_name])
    config_record = dict(role_dataset["config"])
    config_path = resolve_manifest_path(config_record["path"], project_root=project_root)
    if file_sha256(config_path) != config_record["sha256"]:
        raise RuntimeError(f"Resolved role config is stale or corrupted: {config_path}")
    config = load_acccollab_config(str(config_path))

    role_policies = dict(dict(manifest["source"])["role_policies"])
    policy = dict(role_policies[role_name])
    actor_adapter = resolve_manifest_path(
        policy["sft_actor_adapter"],
        project_root=project_root,
    )
    critic_adapter = None
    if variant == "sft_trained_critic":
        critic_adapter = resolve_manifest_path(
            policy["trained_critic_adapter"],
            project_root=project_root,
        )
    rounds = 1 if variant == "sft_only" else 5
    return {
        "manifest": manifest,
        "config": config,
        "actor_adapter": str(actor_adapter),
        "critic_adapter": str(critic_adapter) if critic_adapter is not None else None,
        "rounds": rounds,
        "expected_samples": int(dataset["expected_samples"]),
        "logical_shards": int(dataset["logical_shards"]),
    }


def load_eval_samples(config: ACCCollabConfig) -> list[dict[str, Any]]:
    """Load the exact benchmark split with the source experiment's role prompts."""
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


def generate_variant_shard(
    manifest_path: str | Path,
    *,
    dataset_name: str,
    variant: str,
    role_name: str,
    output_dir: str | Path,
    device: int,
    shard_idx: int,
    num_shards: int,
    project_root: str | Path,
) -> Path:
    """Generate one resumable logical shard for one intermediate policy state."""
    context = load_variant_context(
        manifest_path,
        dataset_name=dataset_name,
        variant=variant,
        role_name=role_name,
        project_root=project_root,
    )
    if num_shards != int(context["logical_shards"]):
        raise ValueError(
            f"{dataset_name} must retain {context['logical_shards']} logical shards; "
            f"got {num_shards}"
        )
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"Invalid logical shard {shard_idx}/{num_shards}")
    config: ACCCollabConfig = context["config"]
    samples = load_eval_samples(config)
    if len(samples) != int(context["expected_samples"]):
        raise RuntimeError(
            f"{dataset_name} coverage changed: {len(samples)} != {context['expected_samples']}"
        )
    shard = shard_samples(samples, shard_idx=shard_idx, num_shards=num_shards)
    root = Path(output_dir)
    fingerprint = _shard_fingerprint(
        context,
        dataset_name=dataset_name,
        variant=variant,
        role_name=role_name,
        shard_idx=shard_idx,
        num_shards=num_shards,
        samples=shard,
    )
    checkpoint = JsonlBatchCheckpoint(
        output_dir=root,
        stage="evaluate_ablation",
        shard_idx=shard_idx,
        num_shards=num_shards,
        fingerprint=fingerprint,
    )
    batch_size = int(config.runtime.batch.evaluation)
    batches = list(_batched(shard, batch_size))
    if not checkpoint.is_complete(len(batches)):
        provenance = {
            "pipeline": "multi_acccollab_ablation",
            "implementation_version": POLICY_LATTICE_VERSION,
            "manifest_fingerprint": context["manifest"]["fingerprint"],
            "variant": variant,
            "role": role_name,
            "actor_adapter": context["actor_adapter"],
            "critic_adapter": context["critic_adapter"] or "base_model",
            "logical_shard": shard_idx,
            "logical_shards": num_shards,
        }
        with build_policy_bundle(
            config,
            actor_adapter=context["actor_adapter"],
            critic_adapter=context["critic_adapter"],
            device=device,
        ) as policies:
            for batch_index, batch in batches:
                if checkpoint.is_completed(batch_index):
                    continue
                seed = batch_seed(
                    config.run.seed,
                    trial_index=0,
                    shard_idx=shard_idx,
                    batch_index=batch_index,
                )
                if int(context["rounds"]) == 1:
                    records = _generate_actor_only_batch(
                        config,
                        actor_policy=policies.actor,
                        samples=batch,
                        seed=seed,
                        provenance=provenance,
                    )
                else:
                    settings = EvaluationSettings(
                        deliberation_rounds=5,
                        actor_max_tokens=config.tokens.actor,
                        critic_max_tokens=config.tokens.critic,
                        temperature=config.generation.eval_temperature,
                        top_p=config.generation.top_p,
                        actor_thinking=config.generation.thinking.eval,
                        critic_thinking=config.generation.thinking.eval,
                    )
                    records = generate_evaluation_batch(
                        actor_policy=policies.actor,
                        critic_policy=policies.critic,
                        samples=batch,
                        dataset_name=config.data.dataset,
                        trial_index=0,
                        settings=settings,
                        seed=seed,
                        policy_provenance=provenance,
                    )
                    for record in records:
                        record["ablation_variant"] = variant
                        record["ablation_role"] = role_name
                records.sort(key=record_key)
                checkpoint.commit(batch_index, {"records": records})

    checkpoint.validate_complete(len(batches))
    shard_dir = root / f"shards/shard-{shard_idx:03d}-of-{num_shards:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    records_path = shard_dir / "records.jsonl"
    checkpoint.materialize("records", records_path, expected_batches=len(batches))
    metrics = score_role_records(iter_jsonl(records_path), expected_samples=len(shard))
    metrics.update(
        {
            "schema_version": 1,
            "pipeline": "multi_acccollab_ablation",
            "implementation_version": POLICY_LATTICE_VERSION,
            "scope": "shard",
            "variant": variant,
            "role": role_name,
            "dataset": dataset_name,
            "fingerprint": fingerprint,
            "shard_idx": shard_idx,
            "num_shards": num_shards,
        }
    )
    metrics_path = shard_dir / "metrics.json"
    write_json(metrics_path, metrics)
    write_json(
        shard_dir / "_SUCCESS",
        {
            "schema_version": 1,
            "pipeline": "multi_acccollab_ablation",
            "stage": "evaluate_ablation_shard",
            "status": "complete",
            "fingerprint": fingerprint,
            "artifacts": {
                "records": {"path": str(records_path), "sha256": file_sha256(records_path)},
                "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
            },
        },
    )
    return shard_dir


def merge_variant_shards(
    manifest_path: str | Path,
    *,
    dataset_name: str,
    variant: str,
    role_name: str,
    output_dir: str | Path,
    num_shards: int,
    project_root: str | Path,
) -> Path:
    """Validate and merge all logical shards for one role/variant."""
    context = load_variant_context(
        manifest_path,
        dataset_name=dataset_name,
        variant=variant,
        role_name=role_name,
        project_root=project_root,
    )
    if num_shards != int(context["logical_shards"]):
        raise ValueError(f"Wrong logical shard count for {dataset_name}: {num_shards}")
    root = Path(output_dir)
    samples = load_eval_samples(context["config"])
    record_paths: list[Path] = []
    shard_fingerprints: list[str] = []
    for shard_idx in range(num_shards):
        shard_samples_expected = shard_samples(
            samples,
            shard_idx=shard_idx,
            num_shards=num_shards,
        )
        fingerprint = _shard_fingerprint(
            context,
            dataset_name=dataset_name,
            variant=variant,
            role_name=role_name,
            shard_idx=shard_idx,
            num_shards=num_shards,
            samples=shard_samples_expected,
        )
        shard_dir = root / f"shards/shard-{shard_idx:03d}-of-{num_shards:03d}"
        marker_path = shard_dir / "_SUCCESS"
        marker = _read_success(marker_path)
        if (
            marker.get("stage") != "evaluate_ablation_shard"
            or marker.get("fingerprint") != fingerprint
        ):
            raise RuntimeError(f"Missing or stale ablation shard: {shard_dir}")
        records_path = shard_dir / "records.jsonl"
        if (
            dict(dict(marker.get("artifacts") or {}).get("records") or {}).get("sha256")
            != file_sha256(records_path)
        ):
            raise RuntimeError(f"Ablation shard records changed after completion: {records_path}")
        record_paths.append(records_path)
        shard_fingerprints.append(fingerprint)

    merged_fingerprint = stable_fingerprint(
        {
            "implementation_version": POLICY_LATTICE_VERSION,
            "manifest_fingerprint": context["manifest"]["fingerprint"],
            "dataset": dataset_name,
            "variant": variant,
            "role": role_name,
            "num_shards": num_shards,
            "shard_fingerprints": shard_fingerprints,
        }
    )
    success_path = root / "_SUCCESS"
    if success_path.is_file():
        marker = _read_success(success_path)
        if marker.get("fingerprint") == merged_fingerprint:
            return root
    root.mkdir(parents=True, exist_ok=True)
    records_path = root / "records.jsonl"
    merge_sorted_jsonl(record_paths, records_path, key=record_key)
    metrics = score_role_records(
        iter_jsonl(records_path),
        expected_samples=int(context["expected_samples"]),
    )
    metrics.update(
        {
            "schema_version": 1,
            "pipeline": "multi_acccollab_ablation",
            "implementation_version": POLICY_LATTICE_VERSION,
            "scope": "role",
            "variant": variant,
            "role": role_name,
            "dataset": dataset_name,
            "fingerprint": merged_fingerprint,
            "num_shards": num_shards,
        }
    )
    metrics_path = root / "metrics.json"
    write_json(metrics_path, metrics)
    write_json(
        success_path,
        {
            "schema_version": 1,
            "pipeline": "multi_acccollab_ablation",
            "stage": "evaluate_ablation",
            "status": "complete",
            "fingerprint": merged_fingerprint,
            "artifacts": {
                "records": {"path": str(records_path), "sha256": file_sha256(records_path)},
                "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
            },
        },
    )
    return root


def aggregate_role_records(
    role_paths: Mapping[str, str | Path],
    *,
    output_dir: str | Path,
    dataset_name: str,
    variant: str,
    expected_samples: int,
    reparse_completions: bool = False,
) -> Path:
    """Aggregate aligned roles and report majority performance for every round."""
    if tuple(role_paths) != ROLE_NAMES:
        raise ValueError(f"role_paths must be ordered exactly as {ROLE_NAMES}")
    paths = {role: Path(path) for role, path in role_paths.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    fingerprint = stable_fingerprint(
        {
            "implementation_version": POLICY_LATTICE_VERSION,
            "dataset": dataset_name,
            "variant": variant,
            "reparse_completions": bool(reparse_completions),
            "answer_extractor_sha256": (
                file_sha256(ANSWER_EXTRACTOR_PATH)
                if reparse_completions
                else None
            ),
            "role_records": {
                role: {"path": str(path), "sha256": file_sha256(path)}
                for role, path in paths.items()
            },
        }
    )
    root = Path(output_dir)
    success_path = root / "_SUCCESS"
    if success_path.is_file() and _read_success(success_path).get("fingerprint") == fingerprint:
        return root

    iterators = [iter_jsonl(paths[role]) for role in ROLE_NAMES]
    round_counts: list[dict[str, Any]] | None = None
    pair_counts = {
        f"{left}+{right}": {"agree": 0, "agree_correct": 0}
        for index, left in enumerate(ROLE_NAMES)
        for right in ROLE_NAMES[index + 1 :]
    }
    sample_count = 0

    def decisions() -> Iterable[dict[str, Any]]:
        nonlocal round_counts, sample_count
        for row_index, aligned in enumerate(zip_longest(*iterators), start=1):
            if any(record is None for record in aligned):
                raise RuntimeError("Ablation role record files have different lengths")
            records = {
                role: dict(record) for role, record in zip(ROLE_NAMES, aligned)
            }
            keys = {record_key(record) for record in records.values()}
            if len(keys) != 1:
                raise RuntimeError(f"Role alignment mismatch at row {row_index}: {keys}")
            signatures = {_sample_signature(record) for record in records.values()}
            if len(signatures) != 1:
                raise RuntimeError(f"Role sample mismatch at row {row_index}")
            sample = dict(records["direct"].get("sample") or {})
            task_type = str(sample.get("task_type") or "multiple_choice")
            gold = sample.get("answer")
            rounds_by_role = {
                role: list(record.get("rounds") or []) for role, record in records.items()
            }
            round_lengths = {len(rounds) for rounds in rounds_by_role.values()}
            if len(round_lengths) != 1:
                raise RuntimeError(f"Role round-count mismatch at row {row_index}")
            num_rounds = next(iter(round_lengths))
            if round_counts is None:
                round_counts = [_empty_round_counts() for _ in range(num_rounds)]
            if len(round_counts) != num_rounds:
                raise RuntimeError("Ablation record round count changed within one file")
            sample_count += 1
            decisions_by_round = []
            for round_index in range(num_rounds):
                completions = {
                    role: _completion_for_scoring(
                        rounds_by_role[role],
                        round_index,
                        sample=sample,
                        reparse=reparse_completions,
                    )
                    for role in ROLE_NAMES
                }
                votes = {
                    role: completion.get("answer")
                    for role, completion in completions.items()
                }
                decision = resolve_majority(
                    votes,
                    task_type=task_type,
                    fallback_role="direct",
                )
                correct = answers_match(decision["answer"], gold, task_type)
                _update_round_counts(
                    round_counts[round_index],
                    completions=completions,
                    decision=decision,
                    correct=correct,
                )
                decisions_by_round.append(
                    {
                        "round": round_index,
                        "role_votes": dict(decision["normalized_votes"]),
                        "decision": {
                            key: value
                            for key, value in decision.items()
                            if key != "normalized_votes"
                        },
                        "correct": correct,
                    }
                )
            final_completions = {
                role: _completion_for_scoring(
                    rounds_by_role[role],
                    num_rounds - 1,
                    sample=sample,
                    reparse=reparse_completions,
                )
                for role in ROLE_NAMES
            }
            normalized = {
                role: normalize_task_answer(completion.get("answer"), task_type)
                for role, completion in final_completions.items()
            }
            for pair_name, counts in pair_counts.items():
                left, right = pair_name.split("+")
                agrees = (
                    normalized[left] is not None
                    and normalized[left] == normalized[right]
                )
                counts["agree"] += int(agrees)
                counts["agree_correct"] += int(
                    agrees and answers_match(normalized[left], gold, task_type)
                )
            yield {
                "schema_version": 1,
                "pipeline": "multi_acccollab_ablation",
                "implementation_version": POLICY_LATTICE_VERSION,
                "dataset": dataset_name,
                "variant": variant,
                "sample_id": str(records["direct"].get("sample_id") or ""),
                "sample_index": int(sample.get("acccollab_sample_index", -1)),
                "task_type": task_type,
                "gold_answer": gold,
                "rounds": decisions_by_round,
            }

    root.mkdir(parents=True, exist_ok=True)
    decisions_path = root / "records.jsonl"
    write_jsonl(decisions_path, decisions())
    if sample_count != expected_samples or round_counts is None:
        raise RuntimeError(
            f"Ablation aggregate coverage mismatch: {sample_count} != {expected_samples}"
        )
    per_round = [
        _finalize_round_counts(index, counts)
        for index, counts in enumerate(round_counts)
    ]
    metrics = {
        "schema_version": 1,
        "pipeline": "multi_acccollab_ablation",
        "implementation_version": POLICY_LATTICE_VERSION,
        "source_sha256": _policy_source_hashes(),
        "dataset": dataset_name,
        "variant": variant,
        "fingerprint": fingerprint,
        "samples": sample_count,
        "roles": list(ROLE_NAMES),
        "decision_rule": "round_majority_then_fixed_direct",
        "uses_judge": False,
        "rescored_from_raw_response": bool(reparse_completions),
        "answer_extractor_sha256": (
            file_sha256(ANSWER_EXTRACTOR_PATH)
            if reparse_completions
            else None
        ),
        "per_round": per_round,
        "headline": per_round[-1],
        "pair_agreement": {
            pair: {
                "coverage": _ratio(counts["agree"], sample_count),
                "conditional_accuracy": _ratio(
                    counts["agree_correct"],
                    counts["agree"],
                ),
                **counts,
            }
            for pair, counts in pair_counts.items()
        },
        "role_records": {
            role: {"path": str(path), "sha256": file_sha256(path)}
            for role, path in paths.items()
        },
    }
    metrics_path = root / "metrics.json"
    write_json(metrics_path, metrics)
    write_json(
        success_path,
        {
            "schema_version": 1,
            "pipeline": "multi_acccollab_ablation",
            "stage": "aggregate_ablation",
            "status": "complete",
            "fingerprint": fingerprint,
            "artifacts": {
                "records": {"path": str(decisions_path), "sha256": file_sha256(decisions_path)},
                "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
            },
        },
    )
    return root


def score_role_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_samples: int,
) -> dict[str, Any]:
    """Score a single role for all recorded rounds."""
    counts: list[dict[str, int]] | None = None
    seen: set[str] = set()
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        if not sample_id or sample_id in seen:
            raise ValueError(f"Missing or duplicate ablation sample id: {sample_id!r}")
        seen.add(sample_id)
        rounds = list(record.get("rounds") or [])
        if not rounds:
            raise ValueError(f"Ablation record has no rounds: {sample_id}")
        if counts is None:
            counts = [
                {"correct": 0, "parsed": 0, "truncated": 0}
                for _ in rounds
            ]
        if len(counts) != len(rounds):
            raise ValueError("Ablation round count changed within a role file")
        for round_index, round_record in enumerate(rounds):
            if int(round_record.get("round", -1)) != round_index:
                raise ValueError(f"Non-contiguous rounds for {sample_id}")
            completion = _completion_at(rounds, round_index)
            counts[round_index]["correct"] += int(bool(completion.get("correct")))
            counts[round_index]["parsed"] += int(bool(completion.get("parsed")))
            counts[round_index]["truncated"] += int(bool(completion.get("truncated")))
    if len(seen) != expected_samples or counts is None:
        raise RuntimeError(
            f"Role ablation coverage mismatch: {len(seen)} != {expected_samples}"
        )
    per_round = [
        {
            "round": index,
            "samples": expected_samples,
            "accuracy": _ratio(item["correct"], expected_samples),
            "parse_rate": _ratio(item["parsed"], expected_samples),
            "truncation_rate": _ratio(item["truncated"], expected_samples),
            **item,
        }
        for index, item in enumerate(counts)
    ]
    return {"samples": expected_samples, "per_round": per_round, "final": per_round[-1]}


def batch_seed(
    base_seed: int,
    *,
    trial_index: int,
    shard_idx: int,
    batch_index: int,
) -> int:
    """Match the completed main experiment's evaluation seed derivation."""
    encoded = f"{base_seed}:evaluate:{trial_index}:{shard_idx}:{batch_index}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big") & 0x7FFFFFFF


def record_key(row: Mapping[str, Any]) -> tuple[int, int]:
    sample = dict(row.get("sample") or {})
    return int(row.get("trial", -1)), int(sample.get("acccollab_sample_index", -1))


def _generate_actor_only_batch(
    config: ACCCollabConfig,
    *,
    actor_policy: Any,
    samples: Sequence[dict[str, Any]],
    seed: int,
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prompts = [
        build_initial_actor_prompt(sample, config.data.dataset)
        for sample in samples
    ]
    completions = generate_actor_records(
        actor_policy,
        prompts,
        list(samples),
        max_tokens=config.tokens.actor,
        temperature=config.generation.eval_temperature,
        top_p=config.generation.top_p,
        enable_thinking=config.generation.thinking.eval,
        seed=seed,
    )
    return [
        {
            "schema_version": 1,
            "pipeline": "acccollab_original",
            "trial": 0,
            "sample_id": str(sample["sample_id"]),
            "sample": sample,
            "settings": {
                "deliberation_rounds": 1,
                "actor_max_tokens": config.tokens.actor,
                "temperature": config.generation.eval_temperature,
                "top_p": config.generation.top_p,
                "actor_thinking": config.generation.thinking.eval,
            },
            "policy_provenance": dict(provenance),
            "ablation_variant": "sft_only",
            "ablation_role": config.prompt_role.name,
            "decision_rule": "single_round0_actor_answer",
            "uses_majority_vote": False,
            "uses_judge_fallback": False,
            "rounds": [
                {
                    "round": 0,
                    "actor": {
                        "prompt": prompts[index],
                        "completion": completions[index],
                    },
                    "critic": None,
                }
            ],
        }
        for index, sample in enumerate(samples)
    ]


def _shard_fingerprint(
    context: Mapping[str, Any],
    *,
    dataset_name: str,
    variant: str,
    role_name: str,
    shard_idx: int,
    num_shards: int,
    samples: Sequence[Mapping[str, Any]],
) -> str:
    config: ACCCollabConfig = context["config"]
    policy = dict(dict(context["manifest"]["source"])["role_policies"])[role_name]
    payload = {
        "implementation_version": POLICY_LATTICE_VERSION,
        "source_sha256": _policy_source_hashes(),
        "manifest_version": ABLATION_MANIFEST_VERSION,
        "manifest_fingerprint": context["manifest"]["fingerprint"],
        "dataset": dataset_name,
        "variant": variant,
        "role": role_name,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "batch_size": config.runtime.batch.evaluation,
        "run_seed": config.run.seed,
        "generation": asdict(config.generation),
        "tokens": asdict(config.tokens),
        "actor_identity": policy["sft_actor_identity"],
        "critic_identity": (
            policy["trained_critic_identity"]
            if variant == "sft_trained_critic"
            else "base_model"
        ),
        "sample_ids": [str(sample.get("sample_id") or "") for sample in samples],
    }
    return stable_fingerprint(payload)


def _completion_at(
    rounds: Sequence[Mapping[str, Any]],
    round_index: int,
) -> dict[str, Any]:
    round_record = dict(rounds[round_index])
    actor = dict(round_record.get("actor") or {})
    return dict(actor.get("completion") or {})


def _completion_for_scoring(
    rounds: Sequence[Mapping[str, Any]],
    round_index: int,
    *,
    sample: Mapping[str, Any],
    reparse: bool,
) -> dict[str, Any]:
    completion = _completion_at(rounds, round_index)
    if not reparse:
        return completion
    raw_response = str(
        completion.get("raw_response")
        or completion.get("response")
        or ""
    )
    return make_actor_record(raw_response, dict(sample))


def _sample_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    sample = dict(record.get("sample") or {})
    return (
        str(sample.get("sample_id") or ""),
        str(sample.get("task_type") or ""),
        str(sample.get("question") or ""),
        tuple(str(choice) for choice in list(sample.get("choices") or [])),
        str(sample.get("passage") or ""),
        str(sample.get("answer") or ""),
    )


def _empty_round_counts() -> dict[str, Any]:
    return {
        "samples": 0,
        "correct": 0,
        "parsed": 0,
        "majority": 0,
        "majority_correct": 0,
        "fallback": 0,
        "fallback_correct": 0,
        "unanimous": 0,
        "oracle_any_correct": 0,
        "roles": {
            role: {"correct": 0, "parsed": 0, "truncated": 0}
            for role in ROLE_NAMES
        },
    }


def _update_round_counts(
    counts: dict[str, Any],
    *,
    completions: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    correct: bool,
) -> None:
    counts["samples"] += 1
    counts["correct"] += int(correct)
    counts["parsed"] += int(decision.get("answer") is not None)
    if bool(decision["majority_reached"]):
        counts["majority"] += 1
        counts["majority_correct"] += int(correct)
    else:
        counts["fallback"] += 1
        counts["fallback_correct"] += int(correct)
    counts["unanimous"] += int(decision["source"] == "unanimous")
    counts["oracle_any_correct"] += int(
        any(bool(completion.get("correct")) for completion in completions.values())
    )
    for role, completion in completions.items():
        role_counts = counts["roles"][role]
        role_counts["correct"] += int(bool(completion.get("correct")))
        role_counts["parsed"] += int(bool(completion.get("parsed")))
        role_counts["truncated"] += int(bool(completion.get("truncated")))


def _finalize_round_counts(round_index: int, counts: Mapping[str, Any]) -> dict[str, Any]:
    total = int(counts["samples"])
    majority = int(counts["majority"])
    fallback = int(counts["fallback"])
    return {
        "round": round_index,
        "samples": total,
        "accuracy": _ratio(int(counts["correct"]), total),
        "parse_rate": _ratio(int(counts["parsed"]), total),
        "majority_coverage": _ratio(majority, total),
        "majority_conditional_accuracy": _ratio(
            int(counts["majority_correct"]),
            majority,
        ),
        "fallback_rate": _ratio(fallback, total),
        "fallback_accuracy": _ratio(int(counts["fallback_correct"]), fallback),
        "unanimous_rate": _ratio(int(counts["unanimous"]), total),
        "oracle_any_role_accuracy": _ratio(
            int(counts["oracle_any_correct"]),
            total,
        ),
        "per_role": {
            role: {
                "accuracy": _ratio(int(counts["roles"][role]["correct"]), total),
                "parse_rate": _ratio(int(counts["roles"][role]["parsed"]), total),
                "truncation_rate": _ratio(
                    int(counts["roles"][role]["truncated"]),
                    total,
                ),
            }
            for role in ROLE_NAMES
        },
    }


def _batched(
    samples: Sequence[dict[str, Any]],
    batch_size: int,
) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(samples), batch_size):
        yield start // batch_size, list(samples[start : start + batch_size])


def _read_success(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    from src.acccollab.io import read_json

    payload = read_json(path)
    return payload if payload.get("status") == "complete" else {}


def _policy_source_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    hashes = {
        str(path.resolve().relative_to(project_root)): file_sha256(path)
        for path in POLICY_SOURCE_PATHS
    }
    hashes[str(Path(__file__).resolve().relative_to(project_root))] = (
        POLICY_LATTICE_V1_SOURCE_SHA256
    )
    return hashes


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
