"""Inference-only baselines for ACC-Collab comparisons."""

from src.baselines.protocols import (
    BASELINE_PROTOCOL_VERSION,
    BaselineSettings,
    build_peer_debate_prompt,
    direct_records_from_debate,
    generate_actor_critic_debate_batch,
    generate_direct_batch,
    generate_som_batch,
    iter_direct_records_from_debate,
    score_actor_critic_debate,
    score_direct,
    score_som,
)

__all__ = [
    "BASELINE_PROTOCOL_VERSION",
    "BaselineSettings",
    "build_peer_debate_prompt",
    "direct_records_from_debate",
    "generate_actor_critic_debate_batch",
    "generate_direct_batch",
    "generate_som_batch",
    "iter_direct_records_from_debate",
    "score_actor_critic_debate",
    "score_direct",
    "score_som",
]
