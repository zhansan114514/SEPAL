"""Generation, cleaning, and answer parsing for original ACC-Collab."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from src.evaluation.answer_resolution import answers_match, normalize_task_answer
from src.parsing.answer_extractor import extract_answer
from src.parsing.think_blocks import clean_model_response, has_unclosed_think
from src.utils.generation_audit import generation_metadata

logger = logging.getLogger(__name__)
EMPTY_RESPONSE_MAX_RETRIES = 2


class GeneratingPolicy(Protocol):
    """Minimal role-policy protocol used by trajectory and rollout code."""

    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool | None,
        seed: int | Sequence[int] | None = None,
    ) -> list[str]: ...


def make_text_record(raw_response: str) -> dict[str, Any]:
    """Clean one generated completion while retaining truncation provenance."""
    raw_text = str(raw_response or "")
    metadata = generation_metadata(raw_text)
    think_truncated = has_unclosed_think(raw_text)
    length_truncated = bool(metadata["length_truncated"])
    return {
        "raw_response": raw_text,
        "response": clean_model_response(raw_text),
        "truncated": bool(length_truncated or think_truncated),
        "truncation_reason": (
            "max_tokens"
            if length_truncated
            else "unclosed_think"
            if think_truncated
            else None
        ),
        "generation": metadata,
    }


def make_actor_record(raw_response: str, sample: dict[str, Any]) -> dict[str, Any]:
    """Build one Actor completion record with parsed answer and correctness."""
    record = make_text_record(raw_response)
    task_type = str(sample.get("task_type") or "multiple_choice")
    extracted = extract_answer(record["response"], task_type)
    normalized = normalize_task_answer(extracted.answer, task_type)
    record.update(
        answer=normalized if normalized is not None else extracted.answer,
        answer_source=extracted.source,
        parse_confidence=extracted.confidence,
        parsed=extracted.answer is not None,
        correct=answers_match(extracted.answer, sample.get("answer"), task_type),
    )
    return record


def generate_text_records(
    policy: GeneratingPolicy,
    prompts: list[str],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool | None,
    seed: int | Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Generate and clean one free-form record per prompt."""
    return _generate_records_with_empty_retries(
        policy,
        prompts,
        record_factory=lambda output, _index: make_text_record(output),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
        seed=seed,
    )


def generate_actor_records(
    policy: GeneratingPolicy,
    prompts: list[str],
    samples: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool | None,
    seed: int | Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Generate, parse, and score one Actor response per sample/prompt."""
    if len(prompts) != len(samples):
        raise ValueError(f"prompts/samples length mismatch: {len(prompts)} != {len(samples)}")
    return _generate_records_with_empty_retries(
        policy,
        prompts,
        record_factory=lambda output, index: make_actor_record(output, samples[index]),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
        seed=seed,
    )


def guidance_answer(sample: dict[str, Any]) -> str:
    """Return the display label used in a guided-positive prompt."""
    task_type = str(sample.get("task_type") or "multiple_choice")
    answer = normalize_task_answer(sample.get("answer"), task_type)
    if task_type == "yes_no":
        if answer == "YES":
            return "Yes"
        if answer == "NO":
            return "No"
    return str(answer if answer is not None else sample.get("answer") or "").strip()


def wrong_guidance_answer(
    sample: dict[str, Any],
    *,
    seed: int | None = None,
) -> str:
    """Sample a stable guided-negative label different from gold (``!y``).

    The released implementation uses ``random.choice`` over wrong labels.  A
    digest of the run/round seed and stable sample identity provides the same
    diversity while remaining deterministic across checkpoint resumes.
    """
    task_type = str(sample.get("task_type") or "multiple_choice")
    gold = guidance_answer(sample)
    if task_type == "yes_no":
        return "No" if gold.upper() in {"YES", "Y"} else "Yes"

    choices = list(sample.get("choices") or [])
    labels = list(sample.get("choice_labels") or [])
    if len(labels) != len(choices):
        labels = [chr(ord("A") + index) for index in range(len(choices))]
    wrong_labels = [
        str(label).strip()
        for label in labels
        if str(label).strip() and str(label).strip().upper() != gold.upper()
    ]
    if wrong_labels:
        sample_identity = str(
            sample.get("sample_id")
            or sample.get("source_index")
            or sample.get("question")
            or "unknown"
        )
        payload = f"{seed!r}:{sample_identity}".encode("utf-8")
        index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return wrong_labels[index % len(wrong_labels)]
    raise ValueError(
        f"Cannot construct guided-negative target for sample {sample.get('sample_id')!r}"
    )


def _validate_output_count(prompts: list[str], outputs: list[str]) -> None:
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"Generation returned {len(outputs)} completions for {len(prompts)} prompts"
        )


def _generate_records_with_empty_retries(
    policy: GeneratingPolicy,
    prompts: list[str],
    *,
    record_factory: Callable[[str, int], dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool | None,
    seed: int | Sequence[int] | None,
) -> list[dict[str, Any]]:
    """Retry only empty completions while preserving the original batch order."""
    if not prompts:
        return []
    _validate_seed_count(seed, len(prompts))
    outputs = policy.generate(
        prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
        seed=seed,
    )
    _validate_output_count(prompts, outputs)
    records = [
        record_factory(str(output or ""), index) for index, output in enumerate(outputs)
    ]
    retry_counts = [0] * len(records)
    pending = _empty_record_indices(records)

    for attempt in range(1, EMPTY_RESPONSE_MAX_RETRIES + 1):
        if not pending:
            break
        logger.warning(
            "Retrying %d empty generation response(s), attempt %d/%d",
            len(pending),
            attempt,
            EMPTY_RESPONSE_MAX_RETRIES,
        )
        retry_prompts = [prompts[index] for index in pending]
        retry_seeds = _derive_empty_retry_seeds(seed, pending, attempt)
        retry_outputs = policy.generate(
            retry_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
            seed=retry_seeds,
        )
        _validate_output_count(retry_prompts, retry_outputs)
        for original_index, output in zip(pending, retry_outputs):
            records[original_index] = record_factory(
                str(output or ""),
                original_index,
            )
            retry_counts[original_index] = attempt
        pending = _empty_record_indices(records)

    exhausted = set(pending)
    if exhausted:
        logger.error(
            "%d generation response(s) remained empty after %d retries; "
            "their downstream preference pairs will be discarded",
            len(exhausted),
            EMPTY_RESPONSE_MAX_RETRIES,
        )
    for index, record in enumerate(records):
        record["empty_response_retries"] = retry_counts[index]
        record["empty_response_exhausted"] = index in exhausted
    return records


def _empty_record_indices(records: Sequence[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, record in enumerate(records)
        if not str(record.get("response") or "").strip()
    ]


def _validate_seed_count(
    seed: int | Sequence[int] | None,
    prompt_count: int,
) -> None:
    if _is_seed_sequence(seed) and len(seed) != prompt_count:
        raise ValueError(
            f"Per-request seed count mismatch: {len(seed)} != {prompt_count}"
        )


def _derive_empty_retry_seeds(
    seed: int | Sequence[int] | None,
    pending_indices: Sequence[int],
    attempt: int,
) -> list[int] | None:
    if seed is None:
        return None
    retry_seeds = []
    for index in pending_indices:
        base_seed = int(seed[index]) if _is_seed_sequence(seed) else int(seed)
        payload = f"{base_seed}:{index}:empty-response-retry:{attempt}".encode("utf-8")
        retry_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        retry_seeds.append(retry_seed & ((1 << 63) - 1))
    return retry_seeds


def _is_seed_sequence(seed: int | Sequence[int] | None) -> bool:
    return isinstance(seed, Sequence) and not isinstance(seed, (str, bytes, bytearray))
