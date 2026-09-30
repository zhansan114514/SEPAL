"""Summary prompts for peer Actor responses."""

from __future__ import annotations


SUMMARY_PROMPT_VERSION = "paired_peer_summary_v1"


def build_peer_summary_prompt(
    problem_text: str,
    target_actor_name: str,
    peer_responses: list[tuple[str, str]],
) -> str:
    """Ask a base model to summarize peer answers and reasons without judging."""
    peer_blocks = []
    for peer_name, response in peer_responses:
        peer_blocks.append(f"{peer_name} response:\n{response.strip()}")
    peers = "\n\n".join(peer_blocks) if peer_blocks else "No peer responses."
    return f"""You are a summarizer, not the final judge.
Summarize the other Actors for {target_actor_name}. Do not solve the problem again
and do not decide who is correct.

For each peer, include its answer, its main reason, and the strongest verification
point or possible weakness.

{problem_text.strip()}

{peers}

Peer summary:""".strip()
