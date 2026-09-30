"""Generation record helpers for the paired pipeline."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.evaluation.answer_resolution import answers_match, normalize_task_answer
from src.parsing.answer_extractor import extract_answer
from src.parsing.think_blocks import clean_model_response, has_unclosed_think
from src.prompts.judge_prompts import build_judge_prompt
from src.prompts.prompt_builder import build_actor_prompt, build_critic_prompt, build_problem_text
from src.prompts.summary_prompts import build_peer_summary_prompt
from src.utils.generation_audit import generation_metadata


def parse_answer(response: str, task_type: str) -> tuple[str | None, str, float]:
    extracted = extract_answer(response, task_type)
    return extracted.answer, extracted.source, extracted.confidence


def is_correct(answer: str | None, sample: dict[str, Any]) -> bool:
    return answers_match(answer, sample.get("answer"), sample.get("task_type", "multiple_choice"))


def make_response_record(
    *,
    raw_response: str,
    task_type: str,
    sample: dict[str, Any],
) -> dict[str, Any]:
    metadata = generation_metadata(raw_response)
    think_truncated = has_unclosed_think(raw_response)
    truncated = bool(metadata["length_truncated"] or think_truncated)
    cleaned = clean_model_response(raw_response)
    answer, source, confidence = parse_answer(cleaned, task_type)
    return {
        "raw_response": raw_response,
        "response": cleaned,
        "answer": answer,
        "answer_source": source,
        "parse_confidence": confidence,
        "correct": is_correct(answer, sample),
        "truncated": truncated,
        "truncation_reason": (
            "max_tokens" if metadata["length_truncated"]
            else "unclosed_think" if think_truncated
            else None
        ),
        "generation": metadata,
    }


def build_initial_actor_prompts(
    actor_names: list[str],
    sample: dict[str, Any],
    dataset_name: str,
) -> dict[str, str]:
    return {
        actor_name: build_actor_prompt(actor_name, sample, dataset_name)
        for actor_name in actor_names
    }


def build_summary_prompts(
    actor_names: list[str],
    sample: dict[str, Any],
    dataset_name: str,
    actor_records: dict[str, dict[str, Any]],
) -> dict[str, str]:
    problem_text = build_problem_text(sample, dataset_name)
    prompts = {}
    for target in actor_names:
        peers = [
            (name, actor_records[name]["response"])
            for name in actor_names
            if name != target and name in actor_records
        ]
        prompts[target] = build_peer_summary_prompt(problem_text, target, peers)
    return prompts


def build_critic_prompts(
    actor_names: list[str],
    critic_names: dict[str, str],
    sample: dict[str, Any],
    dataset_name: str,
    actor_records: dict[str, dict[str, Any]],
    summaries: dict[str, str],
) -> dict[str, str]:
    prompts = {}
    for actor_name in actor_names:
        prompts[actor_name] = build_critic_prompt(
            critic_names[actor_name],
            sample,
            dataset_name,
            actor_name,
            actor_records[actor_name]["response"],
            summaries.get(actor_name, ""),
        )
    return prompts


def build_revision_prompts(
    actor_names: list[str],
    sample: dict[str, Any],
    dataset_name: str,
    actor_records: dict[str, dict[str, Any]],
    critic_feedbacks: dict[str, str],
) -> dict[str, str]:
    return {
        actor_name: build_actor_prompt(
            actor_name,
            sample,
            dataset_name,
            previous_response=actor_records[actor_name]["response"],
            critic_feedback=critic_feedbacks.get(actor_name, ""),
        )
        for actor_name in actor_names
    }


def majority_answer(
    answers: list[str | None],
    task_type: str,
) -> tuple[str | None, int]:
    normalized = []
    for answer in answers:
        item = normalize_task_answer(answer, task_type)
        if item:
            normalized.append(item)
    if not normalized:
        return None, 0
    answer, count = Counter(normalized).most_common(1)[0]
    return answer, count


def resolve_final_decision(
    answers: list[str | None],
    task_type: str,
) -> tuple[str, str | None, str | None, int]:
    """Pick the final answer by majority, deferring to the judge on disagreement.

    Returns ``(source, final_answer, majority_answer, majority_count)``. When
    at least two Actors agree (``majority_count >= 2``) the majority answer is
    final and ``source == "majority"``; otherwise the answers all differ (or are
    missing) and ``source == "judge"`` with ``final_answer=None`` pending the
    judge. With three Actors, no majority is equivalent to all three differing.
    """
    majority, count = majority_answer(answers, task_type)
    if count >= 2:
        return "majority", majority, majority, count
    return "judge", None, majority, count


def build_final_judge_prompt(
    sample: dict[str, Any],
    dataset_name: str,
    final_actor_records: dict[str, dict[str, Any]],
    *,
    tokenizer=None,
    max_input_tokens: int | None = None,
) -> str:
    candidates = [
        (name, record["response"])
        for name, record in final_actor_records.items()
    ]
    return build_judge_prompt(
        build_problem_text(sample, dataset_name),
        candidates,
        tokenizer=tokenizer,
        max_input_tokens=max_input_tokens,
    )
