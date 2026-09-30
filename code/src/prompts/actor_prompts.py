"""Actor prompts for the paired Actor/Critic pipeline."""

from __future__ import annotations


ACTOR_PROMPT_VERSION = "paired_actor_v2_direct_evidence_elimination"
FINAL_ANSWER_LINE = "The final result is <answer>."


ACTOR_ROLES: dict[str, str] = {
    "direct": (
        "You are the Direct Actor. Solve the problem directly and carefully. "
        "State the decisive reasoning that leads to the final answer."
    ),
    "evidence": (
        "You are the Evidence Actor. Solve the problem by grounding the answer "
        "in the key facts, definitions, or evidence from the question."
    ),
    "elimination": (
        "You are the Elimination Actor. Solve the problem by comparing plausible "
        "options and eliminating alternatives before deciding."
    ),
}


def _actor_role(actor_name: str) -> str:
    try:
        return ACTOR_ROLES[actor_name]
    except KeyError as e:
        raise ValueError(
            f"Unknown actor {actor_name!r}; expected one of {list(ACTOR_ROLES)}"
        ) from e


def build_actor_initial_prompt(actor_name: str, problem_text: str) -> str:
    """Prompt one Actor to produce an initial answer with a reason."""
    role = _actor_role(actor_name)
    return f"""{role}

{problem_text.strip()}

Give a concise natural-language reason, then end with exactly one final answer sentence:
{FINAL_ANSWER_LINE}""".strip()


def build_actor_revision_prompt(
    actor_name: str,
    problem_text: str,
    previous_response: str,
    critic_feedback: str,
) -> str:
    """Prompt one Actor to revise after its paired Critic feedback."""
    role = _actor_role(actor_name)
    feedback = critic_feedback.strip() or "No critic feedback was provided."
    return f"""{role}

{problem_text.strip()}

Your previous response:
{previous_response.strip()}

Your paired Critic's feedback:
{feedback}

Revise only if the feedback gives a better-supported answer. If you keep your answer,
explain why the criticism is not decisive. End with exactly one final answer sentence:
{FINAL_ANSWER_LINE}""".strip()


def build_actor_guided_revision_prompt(
    actor_name: str,
    problem_text: str,
    previous_response: str,
    critic_feedback: str,
    target_answer: str,
) -> str:
    """Guided actor revision prompt (ACC-Collab Appendix C).

    ``target_answer`` is the answer the actor should support.  Pass the correct
    answer for guided+ (towards y) or a wrong answer for guided- (away from y).

    The critic feedback is fixed to the natural Critic's comment so that the
    only difference between chosen/rejected DPO completions is the target_answer
    guidance — this keeps the DPO prompt identical across the pair.
    """
    role = _actor_role(actor_name)
    feedback = critic_feedback.strip() or "No critic feedback was provided."
    return f"""{role}

You will now revise your answer with the additional instruction that the final answer is {target_answer}.
You should give a brief justification for your answer of {target_answer}.

{problem_text.strip()}

Your previous response:
{previous_response.strip()}

Your paired Critic's feedback:
{feedback}

Revise the response so that it answers the question with {target_answer}.
End with exactly one final answer sentence:
{FINAL_ANSWER_LINE}""".strip()
