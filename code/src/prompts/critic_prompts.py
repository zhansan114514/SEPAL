"""Natural-language Critic prompts for paired DPO training.

Implements the Guided Collaborative Trajectories method from ACC-Collab
(Estornell et al., ICLR 2025, Section 4.4 + Appendix C).
"""

from __future__ import annotations


CRITIC_PROMPT_VERSION = "paired_natural_critic_v2"


def build_paired_critic_prompt(
    critic_name: str,
    problem_text: str,
    target_actor_name: str,
    target_actor_response: str,
    peer_summary: str,
) -> str:
    """Prompt a paired Critic to comment in natural language (natural trajectory)."""
    return f"""You are {critic_name}, the paired Critic for {target_actor_name}.
Your job is verification: decide whether the target Actor's answer should be kept or corrected.

If it should be corrected, start by directly saying what answer it should be changed to,
then explain why the original answer is wrong.
If it should be kept, start by directly saying that the original answer should be kept,
then explain why the strongest opposing peer argument is not enough.

Use natural language. Do not output numeric certainty scores, tables, JSON, or fixed field labels.

{problem_text.strip()}

Target Actor response:
{target_actor_response.strip()}

Other Actors summarized:
{peer_summary.strip() if peer_summary.strip() else "No peer summary available."}

Critic comment:""".strip()


def build_guided_critic_prompt(
    critic_name: str,
    problem_text: str,
    target_actor_name: str,
    target_actor_response: str,
    peer_summary: str,
    target_answer: str,
) -> str:
    """Guided critic prompt (ACC-Collab Appendix C).

    ``target_answer`` is the answer the critic should support.  Pass the correct
    answer for guided+ (towards y) or a wrong answer for guided- (away from y).
    """
    return f"""You are a deliberation assistant helping {critic_name}, the paired Critic for {target_actor_name}.
You will be given a problem, the target Actor's response, and a summary of peer Actors.
You should use these to develop additional details for why the answer is {target_answer}.
Your details must be brief and must support the fact that the answer is {target_answer}.
If the target Actor already gave {target_answer}, explain why it is correct.
If the target Actor gave a different answer, explain why it should be changed to {target_answer}.

{problem_text.strip()}

Target Actor response:
{target_actor_response.strip()}

Other Actors summarized:
{peer_summary.strip() if peer_summary.strip() else "No peer summary available."}

Guided critic comment:""".strip()
