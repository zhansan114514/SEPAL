"""Run a real-GPU smoke across all intermediate policy combinations."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.runtime import configure_current_ablation_process

configure_current_ablation_process(PROJECT_ROOT)

from src.acccollab.evaluation import EvaluationSettings, generate_evaluation_batch
from src.acccollab.generation import generate_actor_records
from src.acccollab.io import write_json
from src.acccollab.policy import build_policy_bundle
from src.acccollab.prompts import build_initial_actor_prompt
from src.ablations.config import POLICY_VARIANTS
from src.ablations.policy_lattice import (
    batch_seed,
    load_eval_samples,
    load_variant_context,
    score_role_records,
)
from src.utils.artifacts import stable_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--role", default="direct")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    summaries: dict[str, Any] = {}
    for variant in POLICY_VARIANTS:
        context = load_variant_context(
            args.manifest,
            dataset_name="mmlu",
            variant=variant,
            role_name=args.role,
            project_root=PROJECT_ROOT,
        )
        config = context["config"]
        samples = load_eval_samples(config)[: args.samples]
        seed = batch_seed(
            config.run.seed,
            trial_index=0,
            shard_idx=0,
            batch_index=0,
        )
        provenance = {
            "pipeline": "multi_acccollab_ablation_smoke",
            "manifest_fingerprint": context["manifest"]["fingerprint"],
            "variant": variant,
            "role": args.role,
        }
        with build_policy_bundle(
            config,
            actor_adapter=context["actor_adapter"],
            critic_adapter=context["critic_adapter"],
            device=args.device,
        ) as policies:
            if variant == "sft_only":
                prompts = [
                    build_initial_actor_prompt(sample, config.data.dataset)
                    for sample in samples
                ]
                completions = generate_actor_records(
                    policies.actor,
                    prompts,
                    samples,
                    max_tokens=config.tokens.actor,
                    temperature=config.generation.eval_temperature,
                    top_p=config.generation.top_p,
                    enable_thinking=config.generation.thinking.eval,
                    seed=seed,
                )
                records = [
                    {
                        "trial": 0,
                        "sample_id": str(sample["sample_id"]),
                        "sample": sample,
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
                    samples=samples,
                    dataset_name=config.data.dataset,
                    trial_index=0,
                    settings=settings,
                    seed=seed,
                    policy_provenance=provenance,
                )
        summary = score_role_records(records, expected_samples=len(samples))
        expected_rounds = 1 if variant == "sft_only" else 5
        if len(summary["per_round"]) != expected_rounds:
            raise RuntimeError(f"{variant} produced the wrong number of rounds")
        summaries[variant] = summary

    payload = {
        "schema_version": 1,
        "pipeline": "multi_acccollab_ablation_smoke",
        "status": "complete",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "device": args.device,
        "role": args.role,
        "samples": args.samples,
        "variants": summaries,
    }
    payload["fingerprint"] = stable_fingerprint(payload)
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
