"""Sharded Direct, untrained Debate, and SoM evaluation on one dataset."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ACCCOLLAB_SCRIPTS = PROJECT_ROOT / "scripts" / "acccollab"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACCCOLLAB_SCRIPTS))

from _utils import add_config_arguments, batched, expected_batches, load_config, setup_logging
from src.acccollab.data import load_split_samples, shard_samples
from src.acccollab.io import iter_jsonl, merge_sorted_jsonl, write_json, write_jsonl
from src.baselines.protocols import (
    BASELINE_PROTOCOL_VERSION,
    BaselineSettings,
    generate_actor_critic_debate_batch,
    generate_direct_batch,
    generate_som_batch,
    iter_direct_records_from_debate,
    paper_som_comparison,
    score_actor_critic_debate,
    score_direct,
    score_som,
)
from src.inference.vllm_server import build_inference_engine
from src.utils.checkpoints import JsonlBatchCheckpoint, checkpoint_fingerprint
from src.utils.generation_audit import subtract_generation_stats

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--method", choices=("direct", "debate", "som"), required=True)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args)
    setup_logging(seed=config.run.seed)
    _validate_args(args)
    samples = _load_samples(config, max_samples=args.max_samples)
    settings = BaselineSettings(
        rounds=5,
        max_tokens=int(config.tokens.actor),
        temperature=float(config.generation.eval_temperature),
        top_p=float(config.generation.top_p),
        enable_thinking=config.generation.thinking.eval,
    )
    output_dir = Path(args.output_dir)
    batch_size = int(args.batch_size or config.runtime.batch.evaluation)
    stage_fingerprint = _stage_fingerprint(
        config=config,
        args=args,
        settings=settings,
        samples=samples,
        batch_size=batch_size,
    )
    if args.merge_only:
        _merge(
            config=config,
            args=args,
            samples=samples,
            output_dir=output_dir,
            stage_fingerprint=stage_fingerprint,
        )
        return
    _generate_shard(
        config=config,
        args=args,
        samples=samples,
        settings=settings,
        output_dir=output_dir,
        batch_size=batch_size,
        stage_fingerprint=stage_fingerprint,
    )


def _load_samples(config, *, max_samples: int | None) -> list[dict[str, Any]]:
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
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    return samples


def _generate_shard(
    *,
    config,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    settings: BaselineSettings,
    output_dir: Path,
    batch_size: int,
    stage_fingerprint: str,
) -> None:
    shard = shard_samples(samples, shard_idx=args.shard_idx, num_shards=args.num_shards)
    shard_fp = checkpoint_fingerprint(
        {
            "stage_fingerprint": stage_fingerprint,
            "shard_idx": args.shard_idx,
            "num_shards": args.num_shards,
            "sample_ids": [str(sample["sample_id"]) for sample in shard],
        }
    )
    shard_dir = _shard_dir(output_dir, args.shard_idx, args.num_shards)
    success_path = shard_dir / "_SUCCESS.json"
    if _valid_success(success_path, fingerprint=shard_fp):
        logger.info("Reusing completed baseline shard %d/%d", args.shard_idx, args.num_shards)
        return

    total_batches = expected_batches(len(shard), batch_size)
    checkpoint = JsonlBatchCheckpoint(
        output_dir=output_dir,
        stage=f"baseline_{_method_tag(args)}",
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
        fingerprint=shard_fp,
    )
    with build_inference_engine(config.inference_args(args.device)) as engine:
        for batch_index, batch in batched(shard, batch_size):
            if checkpoint.is_completed(batch_index):
                logger.info("Reusing batch %d/%d", batch_index + 1, total_batches)
                continue
            before = engine.generation_stats()
            seed = _batch_seed(
                int(config.run.seed),
                shard_idx=args.shard_idx,
                batch_index=batch_index,
            )
            if args.method == "direct":
                records = generate_direct_batch(
                    policy=engine,
                    samples=batch,
                    dataset_name=config.data.dataset,
                    settings=settings,
                    seed=seed,
                )
            elif args.method == "debate":
                records = generate_actor_critic_debate_batch(
                    policy=engine,
                    samples=batch,
                    dataset_name=config.data.dataset,
                    settings=settings,
                    seed=seed,
                )
            else:
                records = generate_som_batch(
                    policy=engine,
                    samples=batch,
                    dataset_name=config.data.dataset,
                    agents=int(args.agents),
                    settings=settings,
                    seed=seed,
                )
            records.sort(key=_record_key)
            audit = {
                "schema_version": 1,
                "pipeline": "inference_baseline",
                "method": _method_tag(args),
                "batch_index": batch_index,
                "shard_idx": args.shard_idx,
                "num_shards": args.num_shards,
                **subtract_generation_stats(engine.generation_stats(), before),
            }
            checkpoint.commit(batch_index, {"records": records, "generation_audit": [audit]})
            logger.info(
                "Completed %s shard %d/%d batch %d/%d",
                _method_tag(args),
                args.shard_idx,
                args.num_shards,
                batch_index + 1,
                total_batches,
            )
    checkpoint.validate_complete(total_batches)
    shard_dir.mkdir(parents=True, exist_ok=True)
    records_path = shard_dir / "records.jsonl"
    audit_path = shard_dir / "generation_audit.jsonl"
    checkpoint.materialize("records", records_path, expected_batches=total_batches)
    checkpoint.materialize("generation_audit", audit_path, expected_batches=total_batches)
    metrics = _score(args, iter_jsonl(records_path))
    write_json(shard_dir / "metrics.json", metrics)
    write_json(
        success_path,
        {
            "schema_version": 1,
            "stage": "baseline_shard",
            "fingerprint": shard_fp,
            "stage_fingerprint": stage_fingerprint,
            "method": _method_tag(args),
            "samples": len(shard),
        },
    )


def _merge(
    *,
    config,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    output_dir: Path,
    stage_fingerprint: str,
) -> None:
    shard_paths = []
    for shard_idx in range(args.num_shards):
        shard = shard_samples(samples, shard_idx=shard_idx, num_shards=args.num_shards)
        shard_fp = checkpoint_fingerprint(
            {
                "stage_fingerprint": stage_fingerprint,
                "shard_idx": shard_idx,
                "num_shards": args.num_shards,
                "sample_ids": [str(sample["sample_id"]) for sample in shard],
            }
        )
        shard_dir = _shard_dir(output_dir, shard_idx, args.num_shards)
        if not _valid_success(shard_dir / "_SUCCESS.json", fingerprint=shard_fp):
            raise RuntimeError(f"Baseline shard is missing or stale: {shard_dir}")
        shard_paths.append(shard_dir / "records.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    merged = merge_sorted_jsonl(shard_paths, records_path, key=_record_key)
    if merged != len(samples):
        raise RuntimeError(f"Merged sample coverage mismatch: {merged} != {len(samples)}")
    metrics = _score(args, iter_jsonl(records_path))
    metrics.update(
        dataset=config.data.dataset,
        model_name=config.model.name,
        model_type=config.model.type,
        trials=1,
        stage_fingerprint=stage_fingerprint,
    )
    if args.method == "som":
        comparison = paper_som_comparison(
            float(dict(metrics["final"])["accuracy"]),
            model_type=config.model.type,
            dataset_name=config.data.dataset,
            agents=int(args.agents),
        )
        if comparison is not None:
            metrics["paper_comparison"] = comparison
    write_json(output_dir / "aggregate_metrics.json", metrics)
    if args.method == "debate":
        direct_dir = output_dir.parent / "direct"
        direct_dir.mkdir(parents=True, exist_ok=True)
        direct_records_path = direct_dir / "records.jsonl"
        write_jsonl(
            direct_records_path,
            iter_direct_records_from_debate(iter_jsonl(records_path)),
        )
        direct_metrics = score_direct(iter_jsonl(direct_records_path))
        direct_metrics.update(
            dataset=config.data.dataset,
            model_name=config.model.name,
            model_type=config.model.type,
            trials=1,
            stage_fingerprint=stage_fingerprint,
            reused_from=str(records_path),
        )
        write_json(direct_dir / "aggregate_metrics.json", direct_metrics)
        write_json(
            direct_dir / "_SUCCESS.json",
            {
                "schema_version": 1,
                "stage": "baseline_complete",
                "fingerprint": stage_fingerprint,
                "method": "direct",
                "samples": len(samples),
                "reused_from": "actor_critic_debate_round_0",
            },
        )
    write_json(
        output_dir / "_SUCCESS.json",
        {
            "schema_version": 1,
            "stage": "baseline_complete",
            "fingerprint": stage_fingerprint,
            "method": _method_tag(args),
            "samples": len(samples),
            "trials": 1,
        },
    )
    logger.info(
        "Merged %s: samples=%d accuracy=%.6f",
        _method_tag(args),
        len(samples),
        _headline_accuracy(metrics),
    )


def _score(args: argparse.Namespace, records) -> dict[str, Any]:
    if args.method == "direct":
        return score_direct(records)
    if args.method == "debate":
        return score_actor_critic_debate(records)
    return score_som(records, agents=int(args.agents))


def _headline_accuracy(metrics: dict[str, Any]) -> float:
    if "final" in metrics:
        return float(dict(metrics["final"])["accuracy"])
    return float(metrics["accuracy"])


def _stage_fingerprint(
    *,
    config,
    args: argparse.Namespace,
    settings: BaselineSettings,
    samples: list[dict[str, Any]],
    batch_size: int,
) -> str:
    sample_digest = hashlib.sha256(
        "\n".join(str(sample["sample_id"]) for sample in samples).encode("utf-8")
    ).hexdigest()
    return checkpoint_fingerprint(
        {
            "protocol_version": BASELINE_PROTOCOL_VERSION,
            "model_name": config.model.name,
            "model_type": config.model.type,
            "dataset": config.data.dataset,
            "split": config.data.eval.split,
            "method": _method_tag(args),
            "settings": settings.__dict__,
            "seed": int(config.run.seed),
            "trials": 1,
            "num_shards": int(args.num_shards),
            "batch_size": batch_size,
            "samples": len(samples),
            "sample_digest": sample_digest,
        }
    )


def _method_tag(args: argparse.Namespace) -> str:
    return f"som_{int(args.agents)}x" if args.method == "som" else str(args.method)


def _record_key(record: dict[str, Any]) -> tuple[int, str]:
    sample = dict(record.get("sample") or {})
    return (
        int(sample.get("acccollab_sample_index", 0)),
        str(record.get("sample_id") or ""),
    )


def _shard_dir(output_dir: Path, shard_idx: int, num_shards: int) -> Path:
    return output_dir / "shards" / f"shard-{shard_idx:03d}-of-{num_shards:03d}"


def _valid_success(path: Path, *, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        import json

        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("fingerprint") == fingerprint


def _batch_seed(seed: int, *, shard_idx: int, batch_index: int) -> int:
    return int(seed) + int(shard_idx) * 10_000_019 + int(batch_index) * 1_009


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_shards < 1 or not 0 <= args.shard_idx < args.num_shards:
        raise ValueError("Invalid shard placement")
    if args.method == "som" and args.agents not in {2, 4}:
        raise ValueError("SoM agents must be 2 or 4")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max_samples must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch_size must be positive")


if __name__ == "__main__":
    main()
