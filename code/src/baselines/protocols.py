"""Direct, untrained Actor/Critic Debate, and Society-of-Minds baselines."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.acccollab.evaluation import EvaluationSettings, generate_evaluation_batch
from src.acccollab.generation import (
    GeneratingPolicy,
    generate_actor_records,
    make_actor_record,
)
from src.acccollab.prompts import build_initial_actor_prompt
from src.evaluation.answer_resolution import answers_match, normalize_task_answer

BASELINE_PROTOCOL_VERSION = "acccollab_inference_baselines_v1"
PAPER_SOM_RESULTS = {
    "llama3": {
        2: {"boolq": 0.812, "mmlu": 0.620, "bbh": 0.508, "sciq": 0.925, "arc": 0.874},
        4: {"boolq": 0.811, "mmlu": 0.635, "bbh": 0.514, "sciq": 0.923, "arc": 0.874},
    },
    "gemma2": {
        2: {"boolq": 0.750, "mmlu": 0.580, "bbh": 0.454, "sciq": 0.903, "arc": 0.841},
        4: {"boolq": 0.759, "mmlu": 0.578, "bbh": 0.449, "sciq": 0.903, "arc": 0.843},
    },
}


@dataclass(frozen=True)
class BaselineSettings:
    """Shared paper-aligned generation settings."""

    rounds: int = 5
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    enable_thinking: bool | None = False

    def validate(self) -> None:
        if self.rounds != 5:
            raise ValueError("Paper-aligned baseline evaluation requires rounds 0..4")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid generation sampling settings")


def generate_direct_batch(
    *,
    policy: GeneratingPolicy,
    samples: Sequence[dict[str, Any]],
    dataset_name: str,
    settings: BaselineSettings,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate one off-the-shelf model answer per sample."""
    settings.validate()
    samples = [dict(sample) for sample in samples]
    prompts = [build_initial_actor_prompt(sample, dataset_name) for sample in samples]
    completions = generate_actor_records(
        policy,
        prompts,
        samples,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        enable_thinking=settings.enable_thinking,
        seed=_request_seeds(samples, seed=seed, round_index=0, agent_index=0),
    )
    return [
        {
            "schema_version": 1,
            "pipeline": "inference_baseline",
            "method": "direct",
            "protocol_version": BASELINE_PROTOCOL_VERSION,
            "sample_id": str(sample["sample_id"]),
            "sample": sample,
            "settings": asdict(settings),
            "decision_rule": "single_model_single_shot",
            "prompt": prompts[index],
            "completion": completions[index],
        }
        for index, sample in enumerate(samples)
    ]


def generate_actor_critic_debate_batch(
    *,
    policy: GeneratingPolicy,
    samples: Sequence[dict[str, Any]],
    dataset_name: str,
    settings: BaselineSettings,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run an untrained shared-base Actor/Critic alternation for five rounds."""
    settings.validate()
    evaluation_settings = EvaluationSettings(
        deliberation_rounds=settings.rounds,
        actor_max_tokens=settings.max_tokens,
        critic_max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        actor_thinking=settings.enable_thinking,
        critic_thinking=settings.enable_thinking,
    )
    records = generate_evaluation_batch(
        actor_policy=policy,
        critic_policy=policy,
        samples=[dict(sample) for sample in samples],
        dataset_name=dataset_name,
        trial_index=0,
        settings=evaluation_settings,
        seed=seed,
        policy_provenance={
            "source": "shared_untrained_base_model",
            "actor_adapter": None,
            "critic_adapter": None,
        },
    )
    for record in records:
        record.update(
            pipeline="inference_baseline",
            method="actor_critic_debate",
            protocol_version=BASELINE_PROTOCOL_VERSION,
            decision_rule="single_final_actor_answer",
            uses_majority_vote=False,
            uses_judge_fallback=False,
        )
    return records


def direct_records_from_debate(
    debate_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize Direct records from the identical round-0 Actor generations."""
    return list(iter_direct_records_from_debate(debate_records))


def iter_direct_records_from_debate(
    debate_records: Iterable[Mapping[str, Any]],
) -> Iterable[dict[str, Any]]:
    """Yield Direct records without retaining a full benchmark in memory."""
    for raw in debate_records:
        record = dict(raw)
        rounds = list(record.get("rounds") or [])
        if len(rounds) != 5 or int(dict(rounds[0]).get("round", -1)) != 0:
            raise ValueError("Debate record must contain rounds 0..4")
        actor = dict(dict(rounds[0]).get("actor") or {})
        sample = dict(record.get("sample") or {})
        yield {
            "schema_version": 1,
            "pipeline": "inference_baseline",
            "method": "direct",
            "protocol_version": BASELINE_PROTOCOL_VERSION,
            "sample_id": str(record.get("sample_id") or ""),
            "sample": sample,
            "settings": dict(record.get("settings") or {}),
            "decision_rule": "single_model_single_shot",
            "prompt": actor.get("prompt"),
            "completion": _completion_for_scoring(
                dict(actor.get("completion") or {}),
                sample,
            ),
            "reused_from": "actor_critic_debate_round_0",
        }


def generate_som_batch(
    *,
    policy: GeneratingPolicy,
    samples: Sequence[dict[str, Any]],
    dataset_name: str,
    agents: int,
    settings: BaselineSettings,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run symmetric Society-of-Minds peer debate with one shared base model."""
    settings.validate()
    if agents not in {2, 4}:
        raise ValueError("Paper Table 1 reports SoM only for 2 or 4 agents")
    samples = [dict(sample) for sample in samples]
    if not samples:
        return []

    records = [
        {
            "schema_version": 1,
            "pipeline": "inference_baseline",
            "method": f"som_{agents}x",
            "protocol_version": BASELINE_PROTOCOL_VERSION,
            "sample_id": str(sample["sample_id"]),
            "sample": sample,
            "settings": asdict(settings),
            "agents": agents,
            "decision_rule": "mean_final_agent_accuracy",
            "uses_majority_vote": False,
            "uses_judge_fallback": False,
            "rounds": [],
        }
        for sample in samples
    ]
    previous_responses: list[list[str]] | None = None

    for round_index in range(settings.rounds):
        prompts: list[str] = []
        repeated_samples: list[dict[str, Any]] = []
        seeds: list[int] = []
        for sample_index, sample in enumerate(samples):
            for agent_index in range(agents):
                if round_index == 0:
                    prompt = build_initial_actor_prompt(sample, dataset_name)
                else:
                    assert previous_responses is not None
                    prompt = build_peer_debate_prompt(
                        sample,
                        dataset_name,
                        previous_responses[sample_index],
                    )
                prompts.append(prompt)
                repeated_samples.append(sample)
                seeds.append(
                    _request_seed(
                        sample,
                        seed=seed,
                        round_index=round_index,
                        agent_index=agent_index,
                    )
                )
        completions = generate_actor_records(
            policy,
            prompts,
            repeated_samples,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            enable_thinking=settings.enable_thinking,
            seed=seeds,
        )
        previous_responses = []
        cursor = 0
        for record in records:
            round_agents = []
            response_row = []
            shared_prompt = prompts[cursor]
            for agent_index in range(agents):
                completion = completions[cursor]
                round_agents.append(
                    {
                        "agent": agent_index,
                        "completion": completion,
                    }
                )
                response_row.append(str(completion.get("response") or ""))
                cursor += 1
            record["rounds"].append(
                {
                    "round": round_index,
                    "prompt": shared_prompt,
                    "agents": round_agents,
                }
            )
            previous_responses.append(response_row)
        if cursor != len(completions):
            raise RuntimeError("SoM completion reshape did not consume the full batch")
    return records


def build_peer_debate_prompt(
    sample: Mapping[str, Any],
    dataset_name: str,
    responses: Sequence[str],
) -> str:
    """Build the released SoM-style prompt using every previous-round response."""
    if not responses:
        raise ValueError("A peer-debate prompt requires previous responses")
    task_type = str(sample.get("task_type") or "multiple_choice")
    question = str(sample.get("question") or "").strip()
    if task_type == "yes_no":
        kind = "yes-no"
    elif sample.get("choices"):
        kind = "multiple choice"
    else:
        kind = "question-answering"
    response_block = "\n".join(
        f"Person {index} said: {response}" for index, response in enumerate(responses)
    )
    prompt = (
        f"Several people have provided responses to a {kind} question. "
        f"Below are their responses:\n{response_block}\n\n"
        "You should take these responses into consideration when providing your own answer. "
    )
    if task_type == "yes_no":
        prompt += (
            "Give an extremely brief justification and provide a final answer of either "
            "Yes or No."
        )
    elif sample.get("choices"):
        prompt += (
            'Give an extremely brief justification and state the final option letter by saying '
            '"Final Answer:".'
        )
    else:
        prompt += 'Give a brief justification and state the result by saying "Final Answer:".'
    prompt += f"\nQuestion: {question}"
    passage = str(sample.get("passage") or "").strip()
    if passage:
        prompt += f"\nPassage: {passage}"
    choices = list(sample.get("choices") or [])
    labels = list(sample.get("choice_labels") or [])
    if choices:
        if len(labels) != len(choices):
            labels = [chr(ord("A") + index) for index in range(len(choices))]
        prompt += "\nOptions:\n" + "\n".join(
            f"{label}: {choice}" for label, choice in zip(labels, choices)
        )
    return prompt


def score_direct(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Score Direct with the single generated completion as the headline."""
    counts = {"samples": 0, "correct": 0, "parsed": 0, "truncated": 0}
    seen: set[str] = set()
    for raw in records:
        record = dict(raw)
        _validate_identity(record, expected_method="direct", seen=seen)
        _add_completion(
            counts,
            dict(record.get("completion") or {}),
            dict(record.get("sample") or {}),
        )
    return {
        "schema_version": 1,
        "pipeline": "inference_baseline",
        "method": "direct",
        "protocol_version": BASELINE_PROTOCOL_VERSION,
        "decision_rule": "single_model_single_shot",
        **_rates(counts),
    }


def score_actor_critic_debate(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score every Actor round and use round 4 as the Debate headline."""
    per_round_counts = [
        {"samples": 0, "correct": 0, "parsed": 0, "truncated": 0} for _ in range(5)
    ]
    seen: set[str] = set()
    for raw in records:
        record = dict(raw)
        _validate_identity(record, expected_method="actor_critic_debate", seen=seen)
        sample = dict(record.get("sample") or {})
        rounds = list(record.get("rounds") or [])
        if [int(dict(item).get("round", -1)) for item in rounds] != list(range(5)):
            raise ValueError("Actor/Critic Debate record must contain rounds 0..4")
        for round_index, round_record in enumerate(rounds):
            actor = dict(dict(round_record).get("actor") or {})
            _add_completion(
                per_round_counts[round_index],
                dict(actor.get("completion") or {}),
                sample,
            )
    per_round = [
        {"round": round_index, **_rates(counts)}
        for round_index, counts in enumerate(per_round_counts)
    ]
    return {
        "schema_version": 1,
        "pipeline": "inference_baseline",
        "method": "actor_critic_debate",
        "protocol_version": BASELINE_PROTOCOL_VERSION,
        "decision_rule": "single_final_actor_answer",
        "per_round": per_round,
        "final": {**per_round[-1], "source": "round_4_single_actor"},
    }


def score_som(
    records: Iterable[Mapping[str, Any]],
    *,
    agents: int,
) -> dict[str, Any]:
    """Score SoM like the released code: mean accuracy across all final agents."""
    if agents not in {2, 4}:
        raise ValueError("SoM scoring supports 2 or 4 agents")
    agent_counts = [
        {"samples": 0, "correct": 0, "parsed": 0, "truncated": 0} for _ in range(5)
    ]
    vote_counts = [
        {"samples": 0, "correct": 0, "resolved": 0} for _ in range(5)
    ]
    seen: set[str] = set()
    expected_method = f"som_{agents}x"
    for raw in records:
        record = dict(raw)
        _validate_identity(record, expected_method=expected_method, seen=seen)
        sample = dict(record.get("sample") or {})
        task_type = str(sample.get("task_type") or "multiple_choice")
        rounds = list(record.get("rounds") or [])
        if [int(dict(item).get("round", -1)) for item in rounds] != list(range(5)):
            raise ValueError("SoM record must contain rounds 0..4")
        for round_index, round_record in enumerate(rounds):
            round_agents = list(dict(round_record).get("agents") or [])
            if len(round_agents) != agents:
                raise ValueError(f"SoM round must contain exactly {agents} agents")
            answers: list[str] = []
            for agent in round_agents:
                completion = _add_completion(
                    agent_counts[round_index],
                    dict(dict(agent).get("completion") or {}),
                    sample,
                )
                normalized = normalize_task_answer(completion.get("answer"), task_type)
                if normalized is not None:
                    answers.append(normalized)
            vote_counts[round_index]["samples"] += 1
            prediction = _unique_plurality(answers)
            if prediction is not None:
                vote_counts[round_index]["resolved"] += 1
                vote_counts[round_index]["correct"] += int(
                    answers_match(prediction, sample.get("answer"), task_type)
                )
    per_round = []
    for round_index in range(5):
        agent_metrics = _rates(agent_counts[round_index])
        votes = vote_counts[round_index]
        sample_count = votes["samples"]
        per_round.append(
            {
                "round": round_index,
                "agent_decisions": agent_metrics["samples"],
                "mean_agent_accuracy": agent_metrics["accuracy"],
                "mean_agent_parse_rate": agent_metrics["parse_rate"],
                "mean_agent_truncation_rate": agent_metrics["truncation_rate"],
                "samples": sample_count,
                "plurality_vote_accuracy": (
                    votes["correct"] / sample_count if sample_count else 0.0
                ),
                "plurality_vote_coverage": (
                    votes["resolved"] / sample_count if sample_count else 0.0
                ),
            }
        )
    return {
        "schema_version": 1,
        "pipeline": "inference_baseline",
        "method": expected_method,
        "protocol_version": BASELINE_PROTOCOL_VERSION,
        "agents": agents,
        "decision_rule": "mean_final_agent_accuracy",
        "per_round": per_round,
        "final": {
            **per_round[-1],
            "accuracy": per_round[-1]["mean_agent_accuracy"],
            "source": "round_4_mean_agent_accuracy",
        },
    }


def paper_som_comparison(
    measured_accuracy: float,
    *,
    model_type: str,
    dataset_name: str,
    agents: int,
) -> dict[str, Any] | None:
    """Return the corresponding paper Table 1 SoM comparison when available."""
    normalized_model = str(model_type).lower().replace("-", "").replace("_", "")
    aliases = {
        "llama3": "llama3",
        "llama38b": "llama3",
        "llama38binstruct": "llama3",
        "gemma2": "gemma2",
        "gemma22b": "gemma2",
        "gemma22binstruct": "gemma2",
    }
    model = aliases.get(normalized_model)
    dataset = str(dataset_name).lower()
    paper = PAPER_SOM_RESULTS.get(model or "", {}).get(agents, {}).get(dataset)
    if paper is None:
        return None
    return {
        "model_type": model,
        "dataset": dataset,
        "method": f"SoM {agents}x",
        "paper_accuracy": paper,
        "measured_accuracy": measured_accuracy,
        "absolute_delta": measured_accuracy - paper,
    }


def _validate_identity(
    record: Mapping[str, Any],
    *,
    expected_method: str,
    seen: set[str],
) -> None:
    if record.get("pipeline") != "inference_baseline":
        raise ValueError("Record does not belong to the inference baseline pipeline")
    if record.get("method") != expected_method:
        raise ValueError(
            f"Expected method {expected_method!r}, got {record.get('method')!r}"
        )
    if record.get("protocol_version") != BASELINE_PROTOCOL_VERSION:
        raise ValueError("Baseline protocol version mismatch")
    sample_id = str(record.get("sample_id") or "")
    if not sample_id or sample_id in seen:
        raise ValueError(f"Missing or duplicate sample_id: {sample_id!r}")
    seen.add(sample_id)


def _add_completion(
    counts: dict[str, int],
    completion: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    completion = _completion_for_scoring(completion, sample)
    counts["samples"] += 1
    counts["correct"] += int(bool(completion.get("correct")))
    counts["parsed"] += int(bool(completion.get("parsed")))
    counts["truncated"] += int(bool(completion.get("truncated")))
    return completion


def _completion_for_scoring(
    completion: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Reparse raw text so final metrics always use the current unified parser."""
    original = dict(completion)
    raw_response = original.get("raw_response")
    if raw_response is None:
        raw_response = original.get("response") or ""
    rescored = make_actor_record(str(raw_response), dict(sample))
    # JSON round-tripping strips GenerationText's in-memory metadata. Preserve
    # the generation-time truncation audit while refreshing answer fields.
    for key in (
        "truncated",
        "truncation_reason",
        "generation",
        "empty_response_retries",
        "empty_response_exhausted",
    ):
        if key in original:
            rescored[key] = original[key]
    return rescored


def _rates(counts: Mapping[str, int]) -> dict[str, Any]:
    samples = int(counts["samples"])
    return {
        "samples": samples,
        "correct": int(counts["correct"]),
        "parsed": int(counts["parsed"]),
        "truncated": int(counts["truncated"]),
        "accuracy": int(counts["correct"]) / samples if samples else 0.0,
        "parse_rate": int(counts["parsed"]) / samples if samples else 0.0,
        "truncation_rate": int(counts["truncated"]) / samples if samples else 0.0,
    }


def _unique_plurality(answers: Sequence[str]) -> str | None:
    if not answers:
        return None
    counts = Counter(answers)
    highest = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == highest]
    return winners[0] if len(winners) == 1 else None


def _request_seeds(
    samples: Sequence[Mapping[str, Any]],
    *,
    seed: int | None,
    round_index: int,
    agent_index: int,
) -> list[int] | None:
    if seed is None:
        return None
    return [
        _request_seed(
            sample,
            seed=seed,
            round_index=round_index,
            agent_index=agent_index,
        )
        for sample in samples
    ]


def _request_seed(
    sample: Mapping[str, Any],
    *,
    seed: int | None,
    round_index: int,
    agent_index: int,
) -> int:
    base = 0 if seed is None else int(seed)
    sample_index = int(sample.get("acccollab_sample_index", 0))
    return base + round_index * 1_000_003 + agent_index * 100_003 + sample_index
