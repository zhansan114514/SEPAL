"""Judge prompts for final actor-answer adjudication."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Global token-budget guard for the judge prompt. The caller passes an input
# budget that already reserves generation tokens, so this trims only to fit the
# prompt itself.
JUDGE_PROMPT_HEADROOM_TOKENS = 0


def _tokenize(tokenizer, text: str) -> list[int]:
    """Encode without special tokens; fall back for non-HF tokenizers."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def _trim_middle(text: str, cap: int, tokenizer) -> str:
    """Keep head + tail of ``text`` and elide the middle.

    ``cap`` is in tokens. The final-answer sentence for MMLU actors sits at the
    end of the response, so tail preservation is required for judge correctness.
    """
    if cap <= 0:
        return ""
    toks = _tokenize(tokenizer, text)
    if len(toks) <= cap:
        return text
    marker = f"\n...[truncated {len(toks) - cap} tokens]...\n"
    marker_tokens = _tokenize(tokenizer, marker)
    if len(marker_tokens) >= cap:
        return tokenizer.decode(toks[-cap:]).lstrip()
    content_cap = cap - len(marker_tokens)
    head_n = content_cap // 2
    tail_n = content_cap - head_n
    head = tokenizer.decode(toks[:head_n]).rstrip()
    tail = tokenizer.decode(toks[-tail_n:]).lstrip()
    return f"{head}{marker}{tail}"


def _assemble_prompt(problem_text: str, candidate_blocks: list[str]) -> str:
    return f"""You are the final judge for multiple Actors.
You will see the problem and each Actor's final-round public response.
Select the answer that is best supported by the problem. Judge correctness first;
do not prefer a longer answer just because it is longer.

{problem_text.strip()}

{chr(10).join(candidate_blocks)}

Give a brief reason and end with exactly one final answer sentence:
The final result is <answer>.""".strip()


def build_judge_prompt(
    problem_text: str,
    candidates: list[tuple[str, str]],
    *,
    tokenizer=None,
    max_input_tokens: int | None = None,
) -> str:
    """Ask a base model to adjudicate final-round Actor responses.

    ``candidates`` is a list of (actor_name, final_round_response) pairs. The
    judge does not receive summaries, Critic feedback, or hidden think blocks.

    When ``tokenizer`` and ``max_input_tokens`` are both supplied, each
    candidate response is trimmed (head + tail preserved) so the assembled
    prompt fits inside ``max_input_tokens``. ``problem_text`` and the wrapper
    template are always kept whole.
    Without a tokenizer the function behaves exactly as before.
    """
    def build_block(name: str, response: str) -> str:
        return "\n".join([f"{name} final-round response:", response.strip()])

    if tokenizer is None or max_input_tokens is None:
        candidate_blocks = [build_block(name, response) for name, response in candidates]
        return _assemble_prompt(problem_text, candidate_blocks)

    # Budget path. Measure the fixed wrapper overhead (template + problem_text
    # + per-candidate labels) by assembling once with empty bodies.
    empty_blocks = [build_block(name, "") for name, _ in candidates]
    fixed_tokens = len(_tokenize(tokenizer, _assemble_prompt(problem_text, empty_blocks)))
    budget = max_input_tokens - fixed_tokens - JUDGE_PROMPT_HEADROOM_TOKENS

    n = max(1, len(candidates))
    if budget <= 0:
        logger.warning(
            "judge prompt fixed overhead (%d tokens) + headroom already exceeds "
            "max_input_tokens=%d; cannot build judge prompt",
            fixed_tokens, max_input_tokens,
        )
        raise ValueError(
            "judge prompt fixed overhead exceeds max_input_tokens; problem text "
            "and template are too long for the configured context"
        )

    response_cap = max(
        0,
        min(max(0, budget // n), max(0, max_input_tokens - fixed_tokens)),
    )

    candidate_blocks: list[str] = []
    for name, response in candidates:
        trimmed_response = _trim_middle(response, response_cap, tokenizer)
        candidate_blocks.append(build_block(name, trimmed_response))

    return _assemble_prompt(problem_text, candidate_blocks)
