"""Paired Actor/Critic pipeline."""

from __future__ import annotations

ACTOR_NAMES = ("direct", "evidence", "elimination")


def paired_critic_name(actor_name: str) -> str:
    if actor_name not in ACTOR_NAMES:
        raise ValueError(
            f"Unknown paired actor {actor_name!r}; expected one of {list(ACTOR_NAMES)}"
        )
    return f"critic_{actor_name}"
