"""DPO pair construction following ACC-Collab Guided Collaborative Trajectories.

Preference pairs are always anchored to the *natural* trajectory:

  - If guided+ led to a correct revision where natural did not (∆y ≥ ε):
      chosen = guided+ revision,  rejected = natural revision
  - If natural led to a correct revision where guided- did not (∆!y ≥ ε):
      chosen = natural revision,  rejected = guided- revision

For Critic pairs the natural Critic is the anchor:

  - If natural revision was wrong: chosen = guided+ Critic, rejected = natural Critic
  - If natural revision was right: chosen = natural Critic, rejected = guided- Critic

Both members of a pair share an identical DPO prompt — the standard
(non-guided) revision / critic prompt built with the *natural* Critic feedback
or the *natural* actor response.  This keeps DPO training well-defined: only
the completion differs between chosen and rejected.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def quality_metrics(
    actor_pairs: dict[str, list[dict[str, Any]]],
    critic_pairs: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Per-agent pair counts and dy/dny split for quality gating."""
    by_agent: dict[str, dict[str, Any]] = {}
    for name, pairs in {**actor_pairs, **critic_pairs}.items():
        counts = Counter(
            str(pair.get("metadata", {}).get("case_type") or "unknown")
            for pair in pairs
        )
        total = len(pairs)
        by_agent[name] = {
            "total": total,
            "case_counts": dict(counts),
        }
    return by_agent


def validate_pair_balance(
    actor_pairs: dict[str, list[dict[str, Any]]],
    critic_pairs: dict[str, list[dict[str, Any]]],
    *,
    min_pairs_per_agent: int,
) -> None:
    """Ensure every agent has enough DPO pairs."""
    quality = quality_metrics(actor_pairs, critic_pairs)
    errors: list[str] = []
    for name, item in quality.items():
        total = int(item["total"])
        if total < min_pairs_per_agent:
            errors.append(f"{name}: {total} pairs < {min_pairs_per_agent}")
    if errors:
        raise RuntimeError("DPO pair quality gate failed: " + "; ".join(errors))
