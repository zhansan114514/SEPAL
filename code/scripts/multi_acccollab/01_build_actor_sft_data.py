"""Build role-specialized Actor SFT data from 10,000 MMLU train questions."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from _utils import add_config_argument, load_config, setup_logging
from src.acccollab.data import load_split_samples, shard_samples
from src.acccollab.generation import make_actor_record
from src.acccollab.io import iter_jsonl, merge_sorted_jsonl, write_json
from src.acccollab.prompts import (
    build_initial_actor_prompt,
    prompt_version_for_role,
    specialize_sample,
)
from src.multi_acccollab.config import MultiACCCollabConfig, config_snapshot
from src.multi_acccollab.sft import select_matched_sft_rows, validate_balanced_sft_rows
from src.utils.artifacts import file_sha256, stable_fingerprint
from src.utils.checkpoints import JsonlBatchCheckpoint
from src.utils.generation_audit import (
    assess_generation_stats,
    enforce_generation_assessment,
    subtract_generation_stats,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-shards", type=int, default=None)
    return parser


def load_sft_samples(config: MultiACCCollabConfig) -> list[dict[str, Any]]:
    base = config.base_config()
    source = config.sft.data
    return load_split_samples(
        dataset_name=base.data.dataset,
        split=source.split,
        max_samples=source.samples,
        strategy=source.strategy,
        seed=config.run.seed,
        mmlu_load_mode=base.data.mmlu_load_mode,
        expected_samples=source.expected_samples,
    )


def data_fingerprint(
    config: MultiACCCollabConfig,
    *,
    samples: list[dict[str, Any]],
    shard_idx: int,
    num_shards: int,
) -> str:
    return stable_fingerprint(
        {
            "pipeline": "multi_acccollab",
            "stage": "actor_sft_data",
            "config": config_snapshot(config),
            "shard_idx": shard_idx,
            "num_shards": num_shards,
            "sample_ids": [sample["sample_id"] for sample in samples],
            "sample_content_fingerprint": stable_fingerprint({"samples": samples}),
            "prompt_versions": {
                role.name: prompt_version_for_role(role.name) for role in config.roles
            },
        }
    )


def generate_shard(
    config: MultiACCCollabConfig,
    *,
    device: int,
    shard_idx: int,
    num_shards: int,
) -> Path:
    all_samples = load_sft_samples(config)
    samples = shard_samples(all_samples, shard_idx=shard_idx, num_shards=num_shards)
    output_dir = config.paths.sft_data_dir
    fingerprint = data_fingerprint(
        config, samples=samples, shard_idx=shard_idx, num_shards=num_shards
    )
    checkpoint = JsonlBatchCheckpoint(
        output_dir=output_dir,
        stage="multi_actor_sft_data",
        shard_idx=shard_idx,
        num_shards=num_shards,
        fingerprint=fingerprint,
    )
    batch_size = config.runtime.sft_generation_batch_size
    batch_count = (len(samples) + batch_size - 1) // batch_size
    if not checkpoint.is_complete(batch_count):
        from src.inference.vllm_server import build_inference_engine

        base = config.base_config()
        engine = build_inference_engine(base.inference_args(device))
        try:
            for batch_index, start in enumerate(range(0, len(samples), batch_size)):
                if checkpoint.is_completed(batch_index):
                    continue
                batch = samples[start : start + batch_size]
                before = engine.generation_stats()
                candidates: list[dict[str, Any]] = []
                for temperature in config.sft.generation.temperatures:
                    prompts: list[str] = []
                    coordinates: list[tuple[dict[str, Any], Any]] = []
                    for sample in batch:
                        for role in config.roles:
                            specialized = specialize_sample(
                                sample,
                                role_name=role.name,
                                actor_instruction=role.actor_instruction,
                                critic_instruction=role.critic_instruction,
                            )
                            prompts.append(
                                build_initial_actor_prompt(specialized, base.data.dataset)
                            )
                            coordinates.append((specialized, role))
                    outputs = engine.generate(
                        prompts,
                        max_tokens=config.sft.generation.max_tokens,
                        temperature=float(temperature),
                        top_p=config.sft.generation.top_p,
                        enable_thinking=config.sft.generation.enable_thinking,
                        seed=[
                            _request_seed(
                                config.run.seed,
                                sample_index=int(sample["acccollab_sample_index"]),
                                role_name=role.name,
                                temperature=float(temperature),
                            )
                            for sample, role in coordinates
                        ],
                    )
                    if len(outputs) != len(coordinates):
                        raise RuntimeError(
                            f"SFT generation returned {len(outputs)} outputs for "
                            f"{len(coordinates)} prompts"
                        )
                    for prompt, raw, (sample, role) in zip(prompts, outputs, coordinates):
                        completion = make_actor_record(str(raw), sample)
                        row = {
                            "sample_index": int(sample["acccollab_sample_index"]),
                            "sample_id": str(sample["sample_id"]),
                            "role": role.name,
                            "temperature": float(temperature),
                            "prompt_version": prompt_version_for_role(role.name),
                            "prompt": prompt,
                            **completion,
                        }
                        candidates.append(row)
                sft_rows = select_matched_sft_rows(config, candidates)
                candidates.sort(key=_row_key)
                sft_rows.sort(key=_row_key)
                audit = {
                    "schema_version": 1,
                    "pipeline": "multi_acccollab",
                    "stage": "actor_sft_data",
                    "shard_idx": shard_idx,
                    "num_shards": num_shards,
                    "batch_index": batch_index,
                    **subtract_generation_stats(engine.generation_stats(), before),
                }
                checkpoint.commit(
                    batch_index,
                    {"candidates": candidates, "sft_rows": sft_rows, "audit": [audit]},
                )
                logger.info(
                    "SFT shard %d/%d batch %d/%d complete",
                    shard_idx,
                    num_shards,
                    batch_index + 1,
                    batch_count,
                )
        finally:
            engine.cleanup()

    checkpoint.validate_complete(batch_count)
    shard_dir = output_dir / "shards" / f"shard-{shard_idx:03d}-of-{num_shards:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.materialize(
        "candidates", shard_dir / "candidates.jsonl", expected_batches=batch_count
    )
    checkpoint.materialize(
        "sft_rows", shard_dir / "sft_rows.jsonl", expected_batches=batch_count
    )
    checkpoint.materialize(
        "audit", shard_dir / "generation_audit.jsonl", expected_batches=batch_count
    )
    write_json(
        shard_dir / "_SUCCESS",
        {
            "schema_version": 1,
            "pipeline": "multi_acccollab",
            "status": "complete",
            "stage": "actor_sft_data_shard",
            "fingerprint": fingerprint,
            "samples": len(samples),
        },
    )
    return shard_dir


def merge_shards(config: MultiACCCollabConfig, *, num_shards: int) -> Path:
    samples = load_sft_samples(config)
    output_dir = config.paths.sft_data_dir
    candidate_paths: list[Path] = []
    row_paths: list[Path] = []
    audit_paths: list[Path] = []
    for shard_idx in range(num_shards):
        shard = shard_samples(samples, shard_idx=shard_idx, num_shards=num_shards)
        fingerprint = data_fingerprint(
            config, samples=shard, shard_idx=shard_idx, num_shards=num_shards
        )
        shard_dir = output_dir / "shards" / f"shard-{shard_idx:03d}-of-{num_shards:03d}"
        marker = _read_mapping(shard_dir / "_SUCCESS")
        if marker.get("fingerprint") != fingerprint or marker.get("status") != "complete":
            raise RuntimeError(f"Missing or stale SFT shard: {shard_dir}")
        candidate_paths.append(shard_dir / "candidates.jsonl")
        row_paths.append(shard_dir / "sft_rows.jsonl")
        audit_paths.append(shard_dir / "generation_audit.jsonl")

    candidates_path = output_dir / "candidates.jsonl"
    rows_path = output_dir / "sft_rows.jsonl"
    audit_path = output_dir / "generation_audit.jsonl"
    candidate_count = merge_sorted_jsonl(candidate_paths, candidates_path, key=_row_key)
    row_count = merge_sorted_jsonl(row_paths, rows_path, key=_row_key)
    merge_sorted_jsonl(audit_paths, audit_path, key=_audit_key)
    expected = len(samples) * len(config.roles) * len(config.sft.generation.temperatures)
    if candidate_count != expected:
        raise RuntimeError(f"SFT candidate coverage mismatch: {candidate_count} != {expected}")
    _validate_candidate_coverage(config, samples, candidates_path)
    counts = Counter(str(row["role"]) for row in iter_jsonl(rows_path))
    matched_samples = validate_balanced_sft_rows(config, list(iter_jsonl(rows_path)))
    for role in config.roles:
        if counts[role.name] < config.sft.training.min_examples_per_role:
            raise RuntimeError(
                f"Role {role.name} has only {counts[role.name]} valid SFT rows"
            )
    base = config.base_config()
    audit_summary = assess_generation_stats(
        iter_jsonl(audit_path),
        warn_rate=base.generation.truncation.warn_rate,
        fail_rate=base.generation.truncation.fail_rate,
        max_model_len=base.model.max_model_len,
    )
    metrics = {
        "schema_version": 1,
        "pipeline": "multi_acccollab",
        "stage": "actor_sft_data",
        "source_split": "mmlu_train_auxiliary_train",
        "source_samples": len(samples),
        "candidates": candidate_count,
        "sft_rows": row_count,
        "sft_rows_per_role": dict(counts),
        "matched_unique_source_samples": matched_samples,
        "rows_per_source_sample_per_role": 1,
        "role_data_balanced": True,
        "generation_audit": audit_summary,
    }
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    enforce_generation_assessment(
        audit_summary,
        fail_on_excess=base.generation.truncation.fail_on_excess,
    )
    stage_fingerprint = stable_fingerprint(
        {
            "config": config_snapshot(config),
            "config_fingerprint": stable_fingerprint(config_snapshot(config)),
            "num_shards": num_shards,
            "candidates_sha256": file_sha256(candidates_path),
            "rows_sha256": file_sha256(rows_path),
        }
    )
    write_json(
        output_dir / "_SUCCESS",
        {
            "schema_version": 1,
            "pipeline": "multi_acccollab",
            "status": "complete",
            "stage": "actor_sft_data",
            "fingerprint": stage_fingerprint,
            "config_fingerprint": stable_fingerprint(config_snapshot(config)),
            "num_shards": num_shards,
            "artifacts": {
                "candidates": {
                    "path": str(candidates_path),
                    "sha256": file_sha256(candidates_path),
                },
                "sft_rows": {"path": str(rows_path), "sha256": file_sha256(rows_path)},
                "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
            },
        },
    )
    return output_dir


def _validate_candidate_coverage(
    config: MultiACCCollabConfig,
    samples: list[dict[str, Any]],
    path: Path,
) -> None:
    expected = {
        (int(sample["acccollab_sample_index"]), role.name, float(temperature))
        for sample in samples
        for role in config.roles
        for temperature in config.sft.generation.temperatures
    }
    actual: set[tuple[int, str, float]] = set()
    for row in iter_jsonl(path):
        key = (int(row["sample_index"]), str(row["role"]), float(row["temperature"]))
        if key in actual:
            raise RuntimeError(f"Duplicate SFT candidate key: {key}")
        actual.add(key)
    if actual != expected:
        raise RuntimeError(
            f"SFT candidate keys differ: missing={list(expected - actual)[:10]}, "
            f"extra={list(actual - expected)[:10]}"
        )


def _row_key(row: Mapping[str, Any]) -> tuple[int, str, float]:
    return int(row["sample_index"]), str(row["role"]), float(row["temperature"])


def _audit_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["shard_idx"]), int(row["batch_index"])


def _read_mapping(path: Path) -> dict[str, Any]:
    from src.acccollab.io import read_json

    try:
        return read_json(path)
    except (OSError, ValueError):
        return {}


def _request_seed(
    base_seed: int,
    *,
    sample_index: int,
    role_name: str,
    temperature: float,
) -> int:
    payload = f"{base_seed}:sft:{sample_index}:{role_name}:{temperature:.8f}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    setup_logging(config.run.seed)
    if args.merge_shards is not None:
        merge_shards(config, num_shards=int(args.merge_shards))
        return
    device = (
        int(args.device)
        if args.device is not None
        else config.runtime.sft_generation_devices[0]
    )
    generate_shard(
        config,
        device=device,
        shard_idx=int(args.shard_idx),
        num_shards=int(args.num_shards),
    )
    if int(args.num_shards) == 1:
        merge_shards(config, num_shards=1)


if __name__ == "__main__":
    main()
