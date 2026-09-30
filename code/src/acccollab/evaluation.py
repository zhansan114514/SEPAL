"""Five-round, final-Actor-only evaluation for the original ACC-Collab protocol."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.acccollab.generation import (
    GeneratingPolicy,
    generate_actor_records,
    generate_text_records,
)
from src.acccollab.prompts import (
    build_actor_deliberation_prompt,
    build_critic_prompt,
    build_initial_actor_prompt,
    prompt_version_for_sample,
)

PAPER_RESULTS = {
    "llama3": {
        "boolq": {"ACC-Collab": (0.887, 0.005), "ACC-Collab+": (0.894, 0.003)},
        "mmlu": {"ACC-Collab": (0.644, 0.010), "ACC-Collab+": (0.683, 0.012)},
        "bbh": {"ACC-Collab": (0.593, 0.006), "ACC-Collab+": (0.574, 0.003)},
        "sciq": {"ACC-Collab": (0.952, 0.000), "ACC-Collab+": (0.948, 0.003)},
        "arc": {"ACC-Collab": (0.881, 0.004), "ACC-Collab+": (0.869, 0.002)},
    },
    "mistral": {
        "boolq": {"ACC-Collab": (0.877, 0.002), "ACC-Collab+": (0.893, 0.002)},
        "mmlu": {"ACC-Collab": (0.610, 0.005), "ACC-Collab+": (0.672, 0.004)},
        "bbh": {"ACC-Collab": (0.519, 0.009), "ACC-Collab+": (0.601, 0.004)},
        "sciq": {"ACC-Collab": (0.902, 0.005), "ACC-Collab+": (0.905, 0.002)},
        "arc": {"ACC-Collab": (0.843, 0.003), "ACC-Collab+": (0.856, 0.003)},
    },
    "gemma2": {
        "boolq": {"ACC-Collab": (0.840, 0.005), "ACC-Collab+": (0.845, 0.005)},
        "mmlu": {"ACC-Collab": (0.510, 0.016), "ACC-Collab+": (0.555, 0.003)},
        "bbh": {"ACC-Collab": (0.513, 0.006), "ACC-Collab+": (0.475, 0.008)},
        "sciq": {"ACC-Collab": (0.918, 0.003), "ACC-Collab+": (0.909, 0.003)},
        "arc": {"ACC-Collab": (0.852, 0.003), "ACC-Collab+": (0.849, 0.002)},
    },
}


@dataclass(frozen=True)
class EvaluationSettings:
    """Natural deliberation settings used identically for all trials."""

    deliberation_rounds: int
    actor_max_tokens: int
    critic_max_tokens: int
    temperature: float
    top_p: float
    actor_thinking: bool | None = False
    critic_thinking: bool | None = False

    def validate(self) -> None:
        if self.deliberation_rounds != 5:
            raise ValueError("Paper-original ACC-Collab evaluation requires rounds t=0..4")
        if min(self.actor_max_tokens, self.critic_max_tokens) <= 0:
            raise ValueError("Evaluation token budgets must be positive")


def generate_evaluation_batch(
    *,
    actor_policy: GeneratingPolicy,
    critic_policy: GeneratingPolicy,
    samples: Sequence[dict[str, Any]],
    dataset_name: str,
    trial_index: int,
    settings: EvaluationSettings,
    seed: int | None = None,
    policy_provenance: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run five natural Actor→Critic rounds with no vote or Judge fallback."""
    settings.validate()
    samples = [dict(sample) for sample in samples]
    if not samples:
        return []

    actor_prompts = [build_initial_actor_prompt(sample, dataset_name) for sample in samples]
    actors = generate_actor_records(
        actor_policy,
        actor_prompts,
        samples,
        max_tokens=settings.actor_max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        enable_thinking=settings.actor_thinking,
        seed=_offset_seed(seed, 0, 0),
    )
    critic_prompts = [
        build_critic_prompt(sample, dataset_name, str(actor["response"]))
        for sample, actor in zip(samples, actors)
    ]
    critics = generate_text_records(
        critic_policy,
        critic_prompts,
        max_tokens=settings.critic_max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        enable_thinking=settings.critic_thinking,
        seed=_offset_seed(seed, 0, 1),
    )

    records = [
        {
            "schema_version": 1,
            "pipeline": "acccollab_original",
            "trial": int(trial_index),
            "sample_id": str(sample["sample_id"]),
            "sample": sample,
            "prompt_version": prompt_version_for_sample(sample),
            "settings": asdict(settings),
            "policy_provenance": dict(policy_provenance or {}),
            "decision_rule": "single_final_actor_answer",
            "uses_majority_vote": False,
            "uses_judge_fallback": False,
            "rounds": [
                {
                    "round": 0,
                    "actor": {"prompt": actor_prompts[index], "completion": actors[index]},
                    "critic": {"prompt": critic_prompts[index], "completion": critics[index]},
                }
            ],
        }
        for index, sample in enumerate(samples)
    ]
    previous_actors = [str(record["response"]) for record in actors]
    previous_critics = [str(record["response"]) for record in critics]

    for round_index in range(1, settings.deliberation_rounds):
        actor_prompts = [
            build_actor_deliberation_prompt(
                sample,
                dataset_name,
                actor_response,
                critic_response,
            )
            for sample, actor_response, critic_response in zip(
                samples,
                previous_actors,
                previous_critics,
            )
        ]
        actors = generate_actor_records(
            actor_policy,
            actor_prompts,
            samples,
            max_tokens=settings.actor_max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            enable_thinking=settings.actor_thinking,
            seed=_offset_seed(seed, round_index, 0),
        )
        actor_texts = [str(record["response"]) for record in actors]
        critic_prompts = [
            build_critic_prompt(sample, dataset_name, actor_response)
            for sample, actor_response in zip(samples, actor_texts)
        ]
        critics = generate_text_records(
            critic_policy,
            critic_prompts,
            max_tokens=settings.critic_max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            enable_thinking=settings.critic_thinking,
            seed=_offset_seed(seed, round_index, 1),
        )
        for index, record in enumerate(records):
            record["rounds"].append(
                {
                    "round": round_index,
                    "actor": {"prompt": actor_prompts[index], "completion": actors[index]},
                    "critic": {"prompt": critic_prompts[index], "completion": critics[index]},
                }
            )
        previous_actors = actor_texts
        previous_critics = [str(record["response"]) for record in critics]

    return records


def score_evaluation_records(
    records: Iterable[Mapping[str, Any]],
    *,
    deliberation_rounds: int = 5,
    expected_sample_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score every Actor round while defining the headline metric from round four only."""
    if deliberation_rounds != 5:
        raise ValueError("Original ACC-Collab metrics require final Actor round index 4")
    round_counts = [
        {"correct": 0, "parsed": 0, "truncated": 0}
        for _ in range(deliberation_rounds)
    ]
    seen: set[str] = set()
    total = 0
    for record in records:
        _validate_record_protocol(record)
        sample_id = str(record.get("sample_id") or "")
        if not sample_id:
            raise ValueError("Evaluation record is missing sample_id")
        if sample_id in seen:
            raise ValueError(f"Duplicate evaluation sample_id: {sample_id}")
        seen.add(sample_id)
        total += 1
        rounds = list(record.get("rounds") or [])
        indices = [int(round_record.get("round", -1)) for round_record in rounds]
        if indices != list(range(deliberation_rounds)):
            raise ValueError(
                f"Sample {sample_id} must contain rounds 0..4 exactly; found {indices}"
            )
        for round_index, round_record in enumerate(rounds):
            actor = dict(round_record.get("actor") or {})
            completion = dict(actor.get("completion") or {})
            round_counts[round_index]["correct"] += int(bool(completion.get("correct")))
            round_counts[round_index]["parsed"] += int(bool(completion.get("parsed")))
            round_counts[round_index]["truncated"] += int(bool(completion.get("truncated")))

    if expected_sample_ids is not None:
        expected = [str(sample_id) for sample_id in expected_sample_ids]
        if len(set(expected)) != len(expected):
            raise ValueError("expected_sample_ids contains duplicates")
        expected_set = set(expected)
        missing = [sample_id for sample_id in expected if sample_id not in seen]
        extra = sorted(seen - expected_set)
        if missing or extra:
            raise ValueError(
                "Evaluation sample coverage mismatch: "
                f"missing={missing[:10]} ({len(missing)} total), "
                f"extra={extra[:10]} ({len(extra)} total)"
            )
    per_round = []
    for round_index, counts in enumerate(round_counts):
        per_round.append(
            {
                "round": round_index,
                "samples": total,
                "correct": counts["correct"],
                "parsed": counts["parsed"],
                "truncated": counts["truncated"],
                "accuracy": counts["correct"] / total if total else 0.0,
                "parse_rate": counts["parsed"] / total if total else 0.0,
                "truncation_rate": counts["truncated"] / total if total else 0.0,
            }
        )
    final = dict(per_round[-1])
    final["source"] = "round_4_single_actor"
    return {
        "schema_version": 1,
        "pipeline": "acccollab_original",
        "decision_rule": "single_final_actor_answer",
        "uses_majority_vote": False,
        "uses_judge_fallback": False,
        "samples": total,
        "per_round": per_round,
        "final": final,
    }


def aggregate_trial_metrics(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate independent trials with sample standard deviation and standard error."""
    if not trials:
        raise ValueError("At least one trial metric is required")
    sample_counts: set[int] = set()
    for trial in trials:
        if trial.get("decision_rule") != "single_final_actor_answer":
            raise ValueError("Every trial must use the single final Actor decision rule")
        if trial.get("uses_majority_vote") is not False:
            raise ValueError("ACC-Collab trial metrics must not use majority vote")
        if trial.get("uses_judge_fallback") is not False:
            raise ValueError("ACC-Collab trial metrics must not use Judge fallback")
        per_round = list(trial.get("per_round") or [])
        indices = [int(round_metric.get("round", -1)) for round_metric in per_round]
        if indices != list(range(5)):
            raise ValueError("Every ACC-Collab trial must contain round metrics 0..4")
        final = dict(trial.get("final") or {})
        if int(final.get("round", -1)) != 4 or final.get("source") != "round_4_single_actor":
            raise ValueError("Every trial headline must come from the single Actor at round 4")
        sample_counts.add(int(trial.get("samples", -1)))
    if len(sample_counts) != 1 or next(iter(sample_counts)) < 0:
        raise ValueError("Every evaluation trial must cover the same non-negative sample count")
    accuracies = [float(dict(trial["final"])["accuracy"]) for trial in trials]
    parse_rates = [float(dict(trial["final"])["parse_rate"]) for trial in trials]
    truncation_rates = [
        float(dict(trial["final"])["truncation_rate"]) for trial in trials
    ]
    aggregate = {
        "schema_version": 1,
        "pipeline": "acccollab_original",
        "trials": len(trials),
        "decision_rule": "single_final_actor_answer",
        "final_accuracy": _summary_stats(accuracies),
        "final_parse_rate": _summary_stats(parse_rates),
        "final_truncation_rate": _summary_stats(truncation_rates),
        "trial_values": [
            {
                "trial": index,
                "accuracy": accuracies[index],
                "parse_rate": parse_rates[index],
                "truncation_rate": truncation_rates[index],
            }
            for index in range(len(trials))
        ],
        "per_round_accuracy": [],
    }
    for round_index in range(5):
        values = [
            float(list(trial["per_round"])[round_index]["accuracy"]) for trial in trials
        ]
        aggregate["per_round_accuracy"].append(
            {"round": round_index, **_summary_stats(values)}
        )
    return aggregate


def compare_with_paper(
    aggregate_metrics: Mapping[str, Any],
    *,
    dataset_name: str,
    model_type: str,
    alternating_iterations: int,
) -> dict[str, Any] | None:
    """Compare with the matching Table 1 row, or return ``None`` if unsupported."""
    if alternating_iterations not in {1, 2}:
        raise ValueError("alternating_iterations must be 1 or 2")
    dataset_key = str(dataset_name).strip().lower()
    raw_model_key = str(model_type).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "llama3": "llama3",
        "llama38b": "llama3",
        "llama38binstruct": "llama3",
        "mistral": "mistral",
        "mistral7b": "mistral",
        "mistral7binstruct": "mistral",
        "gemma2": "gemma2",
        "gemma22b": "gemma2",
        "gemma22binstruct": "gemma2",
    }
    model_key = aliases.get(raw_model_key)
    if model_key is None:
        return None
    dataset_results = PAPER_RESULTS[model_key].get(dataset_key)
    if dataset_results is None:
        return None

    method = "ACC-Collab" if alternating_iterations == 1 else "ACC-Collab+"
    paper_mean, half_width = dataset_results[method]
    measured = float(dict(aggregate_metrics["final_accuracy"])["mean"])
    return {
        "dataset": dataset_key,
        "model_type": model_key,
        "method": method,
        "paper": {
            "accuracy": paper_mean,
            "reported_ci95_half_width": half_width,
            "alternating_iterations": alternating_iterations,
            "ci95_low": paper_mean - half_width,
            "ci95_high": paper_mean + half_width,
            "note": "The paper reports ± as a 95% confidence interval, not a standard deviation.",
        },
        "measured_accuracy": measured,
        "absolute_delta": measured - paper_mean,
        "relative_delta_percent": (
            (measured - paper_mean) / paper_mean * 100.0 if paper_mean else None
        ),
        "above_paper_mean": measured > paper_mean,
        "within_paper_reported_ci95": (
            paper_mean - half_width <= measured <= paper_mean + half_width
        ),
        "reference_note": (
            "Table 1 is a same-dataset benchmark reference. A policy trained on another "
            "dataset is a cross-domain evaluation and is not a like-for-like paper reproduction."
        ),
    }


def _summary_stats(values: Sequence[float]) -> dict[str, float | int]:
    count = len(values)
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if count > 1 else 0.0
    standard_error = sample_std / math.sqrt(count) if count else 0.0
    return {
        "count": count,
        "mean": mean,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "ci95_normal_half_width": 1.96 * standard_error,
    }


def _validate_record_protocol(record: Mapping[str, Any]) -> None:
    """Reject records that could silently introduce voting or Judge-based evaluation."""
    if record.get("pipeline") != "acccollab_original":
        raise ValueError("Evaluation record does not belong to acccollab_original")
    sample = dict(record.get("sample") or {})
    if record.get("prompt_version") != prompt_version_for_sample(sample):
        raise ValueError("Evaluation record prompt_version does not match this implementation")
    if record.get("decision_rule") != "single_final_actor_answer":
        raise ValueError("Evaluation record must use the single final Actor decision rule")
    if record.get("uses_majority_vote") is not False:
        raise ValueError("Original ACC-Collab evaluation forbids majority vote")
    if record.get("uses_judge_fallback") is not False:
        raise ValueError("Original ACC-Collab evaluation forbids Judge fallback")


def _offset_seed(seed: int | None, round_index: int, offset: int) -> int | None:
    return None if seed is None else int(seed) + int(round_index) * 100 + int(offset)
