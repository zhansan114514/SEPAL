"""ACC-Collab Eq. 5 preference selection and pair quality checks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sized
from dataclasses import asdict, dataclass
from typing import Any, Mapping

CANDIDATE_KEYS = ("natural", "guided_positive", "guided_negative")


@dataclass(frozen=True)
class PreferenceDecision:
    """One strict ``if/elif`` decision from paper Eq. 5 / Algorithm 1."""

    case_type: str
    chosen_key: str
    rejected_key: str
    natural_reward: float
    guided_positive_reward: float
    guided_negative_reward: float
    delta_positive: float
    delta_negative: float
    epsilon: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_eq5_pair(
    *,
    natural_reward: float,
    guided_positive_reward: float,
    guided_negative_reward: float,
    epsilon: float,
) -> PreferenceDecision | None:
    """Apply Eq. 5 exactly as the paper's ordered ``if/else if`` rule.

    The guided-positive branch has priority when both deltas meet the threshold.
    This is intentionally *not* two independent ``if`` statements.
    """
    rewards = {
        "natural_reward": float(natural_reward),
        "guided_positive_reward": float(guided_positive_reward),
        "guided_negative_reward": float(guided_negative_reward),
    }
    for name, reward in rewards.items():
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {reward}")
    epsilon = float(epsilon)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

    delta_positive = rewards["guided_positive_reward"] - rewards["natural_reward"]
    delta_negative = rewards["natural_reward"] - rewards["guided_negative_reward"]
    common = {
        **rewards,
        "delta_positive": delta_positive,
        "delta_negative": delta_negative,
        "epsilon": epsilon,
    }
    if delta_positive >= epsilon:
        return PreferenceDecision(
            case_type="guided_positive_over_natural",
            chosen_key="guided_positive",
            rejected_key="natural",
            **common,
        )
    elif delta_negative >= epsilon:
        return PreferenceDecision(
            case_type="natural_over_guided_negative",
            chosen_key="natural",
            rejected_key="guided_negative",
            **common,
        )
    return None


def selection_record(
    *,
    natural_reward: float,
    guided_positive_reward: float,
    guided_negative_reward: float,
    epsilon: float,
) -> dict[str, Any]:
    """Return an auditable selected/dropped representation of Eq. 5."""
    decision = select_eq5_pair(
        natural_reward=natural_reward,
        guided_positive_reward=guided_positive_reward,
        guided_negative_reward=guided_negative_reward,
        epsilon=epsilon,
    )
    if decision is not None:
        return {
            "selected": True,
            "eq5_threshold_passed": True,
            **decision.to_dict(),
        }
    return {
        "selected": False,
        "eq5_threshold_passed": False,
        "drop_reason": "neither_eq5_delta_reached_epsilon",
        "natural_reward": float(natural_reward),
        "guided_positive_reward": float(guided_positive_reward),
        "guided_negative_reward": float(guided_negative_reward),
        "delta_positive": float(guided_positive_reward) - float(natural_reward),
        "delta_negative": float(natural_reward) - float(guided_negative_reward),
        "epsilon": float(epsilon),
    }


def pair_completions_are_distinct(
    candidate_records: Mapping[str, Mapping[str, Any]],
    decision: PreferenceDecision,
) -> bool:
    """Return whether Eq. 5's chosen/rejected texts carry distinct signal."""
    missing = [
        key
        for key in (decision.chosen_key, decision.rejected_key)
        if key not in candidate_records
    ]
    if missing:
        raise ValueError(f"Missing candidate records: {missing}")
    chosen = str(candidate_records[decision.chosen_key].get("response") or "").strip()
    rejected = str(candidate_records[decision.rejected_key].get("response") or "").strip()
    return bool(chosen and rejected and chosen != rejected)


def build_dpo_pair(
    *,
    natural_prompt: str,
    candidate_records: Mapping[str, Mapping[str, Any]],
    decision: PreferenceDecision,
    agent: str,
    iteration: int,
    sample_id: str,
    round_index: int,
    rollouts: int,
    prompt_version: str,
    sample_index: int | None = None,
) -> dict[str, Any]:
    """Materialize a pair whose two completions share the natural prompt."""
    if agent not in {"actor", "critic"}:
        raise ValueError(f"agent must be actor or critic, got {agent!r}")
    missing = [key for key in CANDIDATE_KEYS if key not in candidate_records]
    if missing:
        raise ValueError(f"Missing candidate records: {missing}")
    chosen = str(candidate_records[decision.chosen_key].get("response") or "")
    rejected = str(candidate_records[decision.rejected_key].get("response") or "")
    if not chosen.strip() or not rejected.strip():
        raise ValueError("DPO chosen/rejected completions must be non-empty")
    if not pair_completions_are_distinct(candidate_records, decision):
        raise ValueError("DPO chosen/rejected completions must be different")
    return {
        "prompt": str(natural_prompt),
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "schema_version": 1,
            "pipeline": "acccollab_original",
            "agent": agent,
            "iteration": int(iteration),
            "sample_id": str(sample_id),
            "sample_index": int(sample_index) if sample_index is not None else None,
            "round": int(round_index),
            "case_type": decision.case_type,
            "chosen_candidate": decision.chosen_key,
            "rejected_candidate": decision.rejected_key,
            "rewards": {
                "natural": decision.natural_reward,
                "guided_positive": decision.guided_positive_reward,
                "guided_negative": decision.guided_negative_reward,
            },
            "deltas": {
                "guided_positive_minus_natural": decision.delta_positive,
                "natural_minus_guided_negative": decision.delta_negative,
            },
            "epsilon": decision.epsilon,
            "reward_rollouts": int(rollouts),
            "pair_rule": "paper_eq5_if_elif",
            "dpo_prompt_source": "natural_prompt",
            "prompt_version": prompt_version,
        },
    }


def summarize_pairs(pairs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize pair yield by Eq. 5 branch and deliberation round."""
    cases = Counter()
    rounds = Counter()
    samples: set[str] = set()
    total = 0
    for pair in pairs:
        total += 1
        metadata = pair.get("metadata", {})
        cases[str(metadata.get("case_type") or "unknown")] += 1
        rounds[str(metadata.get("round") or "unknown")] += 1
        sample_id = str(metadata.get("sample_id") or "")
        if sample_id:
            samples.add(sample_id)
    return {
        "total_pairs": total,
        "samples_with_pairs": len(samples),
        "case_counts": dict(sorted(cases.items())),
        "round_counts": dict(sorted(rounds.items(), key=lambda item: item[0])),
    }


def validate_pair_count(
    pairs: Sized | int,
    *,
    minimum: int,
    agent: str,
) -> None:
    """Fail before expensive DPO when Eq. 5 yielded too little signal."""
    if minimum < 1:
        raise ValueError(f"minimum must be positive, got {minimum}")
    count = int(pairs) if isinstance(pairs, int) else len(pairs)
    if count < minimum:
        raise RuntimeError(
            f"ACC-Collab {agent} pair quality gate failed: "
            f"{count} pairs < configured minimum {minimum}"
        )
