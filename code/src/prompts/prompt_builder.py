"""Prompt construction helpers for the paired pipeline."""

from __future__ import annotations

from typing import Any

from src.prompts.actor_prompts import (
    build_actor_initial_prompt,
    build_actor_revision_prompt,
)
from src.prompts.critic_prompts import build_paired_critic_prompt


def build_problem_text(sample: dict[str, Any], dataset_name: str = "") -> str:
    """Render task context without imposing a model-specific format."""
    question = str(sample.get("question", "")).strip()
    passage = str(sample.get("passage", "")).strip()
    choices = sample.get("choices", []) or []
    task_type = sample.get("task_type", "")

    parts: list[str] = []
    if task_type == "math" or dataset_name in {"math", "gsm8k"}:
        parts.append(f"Problem:\n{question}")
    else:
        parts.append(f"Question:\n{question}")

    if passage:
        parts.append(f"Passage:\n{passage}")

    if choices:
        if len(choices) > 4:
            raise ValueError(
                "build_problem_text supports at most 4 choices because answer "
                "extraction and matching currently support labels A-D"
            )
        option_lines = [
            f"({label}) {choice}"
            for label, choice in zip("ABCD", choices)
        ]
        parts.append("Options:\n" + "\n".join(option_lines))

    return "\n\n".join(parts).strip()


def build_actor_prompt(
    actor_name: str,
    sample: dict[str, Any],
    dataset_name: str,
    *,
    previous_response: str = "",
    critic_feedback: str = "",
) -> str:
    """Build an initial or revision Actor prompt."""
    problem_text = build_problem_text(sample, dataset_name)
    if previous_response.strip():
        return build_actor_revision_prompt(
            actor_name,
            problem_text,
            previous_response,
            critic_feedback,
        )
    return build_actor_initial_prompt(actor_name, problem_text)


def build_critic_prompt(
    critic_name: str,
    sample: dict[str, Any],
    dataset_name: str,
    target_actor_name: str,
    target_actor_response: str,
    peer_summary: str,
) -> str:
    """Build a natural paired Critic prompt."""
    return build_paired_critic_prompt(
        critic_name,
        build_problem_text(sample, dataset_name),
        target_actor_name,
        target_actor_response,
        peer_summary,
    )
