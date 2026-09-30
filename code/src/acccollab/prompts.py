"""Prompts from ACC-Collab Appendix C and the released MMLU implementation."""

from __future__ import annotations

from typing import Any

ACCCOLLAB_PROMPT_VERSION = "acccollab_iclr2025_appendix_c_mmlu_v2"
ROLE_PROMPT_VERSION = "acccollab_role_prefix_v1"

_ROLE_NAME_KEY = "acccollab_role_name"
_ACTOR_ROLE_KEY = "acccollab_actor_instruction"
_CRITIC_ROLE_KEY = "acccollab_critic_instruction"
_ROLE_VERSION_KEY = "acccollab_role_prompt_version"


def specialize_sample(
    sample: dict[str, Any],
    *,
    role_name: str,
    actor_instruction: str,
    critic_instruction: str,
    implementation_version: str = ROLE_PROMPT_VERSION,
) -> dict[str, Any]:
    """Return a copy carrying one role's prompt specialization metadata."""
    role = str(role_name).strip()
    actor = str(actor_instruction).strip()
    critic = str(critic_instruction).strip()
    version = str(implementation_version).strip()
    if not role or role == "original":
        if actor or critic:
            raise ValueError("The original prompt role cannot carry instructions")
        return dict(sample)
    if not actor or not critic or not version:
        raise ValueError("A specialized role requires actor, critic, and version strings")
    return {
        **dict(sample),
        _ROLE_NAME_KEY: role,
        _ACTOR_ROLE_KEY: actor,
        _CRITIC_ROLE_KEY: critic,
        _ROLE_VERSION_KEY: version,
    }


def prompt_version_for_sample(sample: dict[str, Any]) -> str:
    """Return the exact prompt identity for an original or specialized sample."""
    role = str(sample.get(_ROLE_NAME_KEY) or "").strip()
    if not role:
        return ACCCOLLAB_PROMPT_VERSION
    version = str(sample.get(_ROLE_VERSION_KEY) or ROLE_PROMPT_VERSION).strip()
    return f"{ACCCOLLAB_PROMPT_VERSION}+{version}:{role}"


def prompt_version_for_role(
    role_name: str,
    implementation_version: str = ROLE_PROMPT_VERSION,
) -> str:
    """Return the prompt identity encoded in a specialized ACC-Collab config."""
    role = str(role_name).strip()
    if not role or role == "original":
        return ACCCOLLAB_PROMPT_VERSION
    return f"{ACCCOLLAB_PROMPT_VERSION}+{implementation_version.strip()}:{role}"


def _role_prefix(sample: dict[str, Any], *, agent: str) -> str:
    key = _ACTOR_ROLE_KEY if agent == "actor" else _CRITIC_ROLE_KEY
    instruction = str(sample.get(key) or "").strip()
    if not instruction:
        return ""
    role = str(sample.get(_ROLE_NAME_KEY) or "specialized").strip()
    return f"Role specialization ({role}): {instruction}\n\n"


def _task_type(sample: dict[str, Any]) -> str:
    return str(sample.get("task_type") or "multiple_choice")


def _question(sample: dict[str, Any]) -> str:
    return str(sample.get("question") or "").strip()


def _passage(sample: dict[str, Any]) -> str:
    return str(sample.get("passage") or "").strip()


def _option_lines(sample: dict[str, Any]) -> str:
    choices = list(sample.get("choices") or [])
    labels = list(sample.get("choice_labels") or [])
    if len(labels) != len(choices):
        labels = [chr(ord("A") + idx) for idx in range(len(choices))]
    return "\n".join(f"{label}: {choice}" for label, choice in zip(labels, choices))


def build_initial_actor_prompt(sample: dict[str, Any], dataset_name: str) -> str:
    """Build the natural t=0 Actor prompt."""
    if _task_type(sample) == "yes_no":
        if dataset_name.lower() == "bbh":
            return _role_prefix(sample, agent="actor") + (
                "You will be given a yes-no question. You should answer the question as "
                "accurately as possible. You should give an extremely brief justification "
                "for your answer, and you must provide a final answer of either Yes or No."
                f"\nQuestion: {_question(sample)}"
            )
        return _role_prefix(sample, agent="actor") + (
            "You will be given a yes-no question which is based on a passage. "
            "You should use the passage to help you answer the question. "
            "You should give a brief justification for your answer, and you must provide "
            "a final answer of either Yes or No."
            f"\nQuestion: {_question(sample)}?"
            f"\nPassage: {_passage(sample)}"
        )
    if sample.get("choices"):
        return _role_prefix(sample, agent="actor") + (
            "Please answer the following multiple choice question as accurately as possible. "
            "You must provide a extremely brief justification for your answer, and you must "
            "give your final answer as a letter by saying \"Final Answer:\"."
            f"\nQuestion: {_question(sample)},"
            f"\nOptions:\n{_option_lines(sample)}"
        )
    return _role_prefix(sample, agent="actor") + (
        "Please answer the following question as accurately as possible. Give a brief "
        "justification and state your final answer by saying \"Final Answer:\"."
        f"\nQuestion: {_question(sample)}"
    )


def build_guided_initial_actor_prompt(
    sample: dict[str, Any],
    dataset_name: str,
    target_answer: str,
) -> str:
    """Build Appendix C's guided single-shot prompt (provided for provenance/tests)."""
    target = str(target_answer).strip()
    if _task_type(sample) == "yes_no":
        if dataset_name.lower() == "bbh":
            return _role_prefix(sample, agent="actor") + (
                f"You will be given a yes-no question. You should answer the question with "
                f"{target}. You should give an extremely brief justification for your answer "
                f"of {target}, and you must state that your final answer is {target}."
                f"\nQuestion: {_question(sample)}"
            )
        return _role_prefix(sample, agent="actor") + (
            "You will be given a yes-no question which is based on a passage. "
            "You should use the passage to help you answer the question "
            f"with a {target}. You should give a brief justification for your answer of "
            f"{target}, and you must state that your final answer is {target}."
            f"\nQuestion: {_question(sample)}?"
            f"\nPassage: {_passage(sample)}"
        )
    if sample.get("choices"):
        return _role_prefix(sample, agent="actor") + (
            f"Please answer the following multiple choice question with option {target}. "
            f"You must provide a extremely brief justification for your answer of {target}, "
            f"and you must give a final answer of {target}, by saying "
            f"\"Final Answer: {target}\"."
            f"\nQuestion: {_question(sample)},"
            f"\nOptions:\n{_option_lines(sample)}"
        )
    return _role_prefix(sample, agent="actor") + (
        f"Answer the following question with {target}. Give a brief justification and state "
        f"\"Final Answer: {target}\".\nQuestion: {_question(sample)}"
    )


def build_actor_deliberation_prompt(
    sample: dict[str, Any],
    dataset_name: str,
    previous_actor_response: str,
    critic_feedback: str,
    *,
    target_answer: str | None = None,
) -> str:
    """Build the natural or guided t>0 Actor prompt.

    The two prior messages are retained in the release's ``Person 0/1`` order:
    previous Actor response first, previous Critic feedback second.
    """
    is_bbh = dataset_name.lower() == "bbh"
    prefix = (
        "Several people have provided answers to a "
        + ("yes-no" if _task_type(sample) == "yes_no" else "multiple choice")
        + " question. Below are their responses:"
        f"\nPerson 0 said: {previous_actor_response}"
        f"\nPerson 1 said: {critic_feedback}"
        "\n\n"
    )
    if _task_type(sample) == "yes_no":
        if target_answer is None:
            if is_bbh:
                instruction = (
                    "You should take these answers into consideration when answering the "
                    "following yes-no question. You should give an extremely brief justification "
                    "for your answer, and you must provide a final answer of either Yes or No."
                )
            else:
                instruction = (
                    "You should take these answers into consideration when answering the "
                    "following yes-no question, which is based on a passage. You should give a "
                    "brief justification for your answer, and you must provide a final answer "
                    "of either Yes or No."
                )
        else:
            target = str(target_answer).strip()
            if is_bbh:
                instruction = (
                    "You should take these answers into consideration when answering the "
                    f"following yes-no question with {target}. You should give an extremely "
                    f"brief justification for your answer of {target}, and you must state that "
                    f"your final answer is {target}."
                )
            else:
                instruction = (
                    "You should take these answers and the passage into consideration when "
                    f"answering the following question with {target}. You should give a brief "
                    f"justification for your answer of {target}, and you must state that your "
                    f"final answer is {target}."
                )
        prompt = (
            _role_prefix(sample, agent="actor")
            + prefix
            + instruction
            + f"\nQuestion: {_question(sample)}"
        )
        if not is_bbh:
            prompt += f"\nPassage: {_passage(sample)}"
        return prompt

    if sample.get("choices"):
        if target_answer is None:
            instruction = (
                "You should take these answers into consideration when answering the following "
                "multiple choice question. You must give an extremely brief justification for "
                "your answer, and you must provide your final answer as a letter by saying "
                "\"Final Answer:\"."
            )
        else:
            target = str(target_answer).strip()
            instruction = (
                "You should take these answers into consideration and answer the following "
                f"multiple choice question with option {target}. You must give an extremely "
                f"brief justification for your answer, and you must provide a final answer of "
                f"{target} by saying \"Final Answer: {target}\"."
            )
        return (
            _role_prefix(sample, agent="actor")
            + prefix
            + instruction
            + f"\nQuestion: {_question(sample)}"
            + f"\nOptions:\n{_option_lines(sample)}"
        )

    target_clause = "" if target_answer is None else f" Your final answer must be {target_answer}."
    return (
        _role_prefix(sample, agent="actor")
        + prefix
        + "Use the previous responses to answer the following question. Give a brief "
        + f"justification and state \"Final Answer:\".{target_clause}"
        + f"\nQuestion: {_question(sample)}"
    )


def build_critic_prompt(
    sample: dict[str, Any],
    dataset_name: str,
    actor_response: str,
    *,
    target_answer: str | None = None,
) -> str:
    """Build the natural or guided Critic prompt."""
    is_bbh = dataset_name.lower() == "bbh"
    if _task_type(sample) == "yes_no":
        if target_answer is None:
            if is_bbh:
                intro = (
                    "I am answering a yes-no question. I would like you to help me improve the "
                    "correctness of my response by very briefly providing some additional "
                    "details my original response may have missed. You should be very terse."
                )
            else:
                intro = (
                    "I am answering a question based on a passage. I would like you to help me "
                    "improve the correctness of my response by briefly providing some additional "
                    "details my original response may have missed."
                )
        else:
            target = str(target_answer).strip()
            if is_bbh:
                intro = (
                    "I would like you to be a debate assistant. You will be given a yes-no "
                    f"question and my response. You should develop two additional details for "
                    f"why the correct answer is {target}. Your details must be extremely brief "
                    f"and must support the fact that the correct answer is {target}."
                )
            else:
                intro = (
                    "I would like you to be a deliberation assistant. You will be given a "
                    "question based on a passage and my response to the question. You should use "
                    f"the passage and my response to develop additional details for why the "
                    f"correct answer is {target}. Your details must be brief and must support "
                    f"the fact that the correct answer is {target}."
                )
        prompt = (
            _role_prefix(sample, agent="critic")
            + intro
            + f"\nQuestion: {_question(sample)}?"
        )
        if not is_bbh:
            prompt += f"\nPassage: {_passage(sample)}"
        return prompt + f"\nMy Response: {actor_response}"

    if sample.get("choices"):
        if target_answer is None:
            intro = (
                "I am answering a multiple choice question. I would like you to help me improve "
                "the correctness of my response by very briefly providing some additional details "
                "my original response may have missed. You should be very terse."
            )
        else:
            target = str(target_answer).strip()
            intro = (
                "I would like you to be a debate assistant. You will be given a multiple choice "
                f"question, you should provide two additional details for why the correct answer "
                f"is {target}. Your details must be extremely brief and must support the fact "
                f"that the correct answer is {target}."
            )
        return (
            _role_prefix(sample, agent="critic")
            + intro
            + f"\nQuestion: {_question(sample)}?"
            + f"\nOptions:\n{_option_lines(sample)}"
            + f"\nMy Response: {actor_response}"
        )

    if target_answer is None:
        instruction = (
            "Help me improve the correctness of my response with brief additional details."
        )
    else:
        instruction = (
            f"Give brief additional details supporting the fact that the correct answer is "
            f"{target_answer}."
        )
    return (
        _role_prefix(sample, agent="critic")
        + instruction
        + f"\nQuestion: {_question(sample)}"
        + f"\nMy Response: {actor_response}"
    )
