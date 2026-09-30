"""One-step Monte Carlo partial-trajectory reward from ACC-Collab Eq. 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.acccollab.generation import (
    GeneratingPolicy,
    generate_actor_records,
    generate_text_records,
)
from src.acccollab.pairs import CANDIDATE_KEYS
from src.acccollab.prompts import build_actor_deliberation_prompt, build_critic_prompt


@dataclass(frozen=True)
class RolloutSettings:
    """Generation settings shared by all Eq. 4 simulations in one stage."""

    actor_max_tokens: int
    critic_max_tokens: int
    temperature: float
    top_p: float
    actor_thinking: bool | None = False
    critic_thinking: bool | None = False


def derive_request_seeds(
    base_seed: int | None,
    request_count: int,
    *,
    stream: int,
) -> list[int] | None:
    """Derive stable, unique per-request seeds for one Eq. 4 stream.

    The 63-bit layout is ``base[32] | request_index[23] | stream[8]``.
    Consequently, request seeds are collision-free for a fixed 32-bit base
    seed across both request indices and up to 256 named streams.  Separate
    streams are used for Critic feedback and Actor revision generation.
    """
    if base_seed is None:
        return None
    if request_count < 0 or request_count >= 1 << 23:
        raise ValueError(f"request_count must be in [0, {1 << 23}), got {request_count}")
    if stream < 0 or stream >= 1 << 8:
        raise ValueError(f"stream must be in [0, 256), got {stream}")

    prefix = int(base_seed) & 0xFFFFFFFF
    return [
        (prefix << 31) | (request_index << 8) | int(stream)
        for request_index in range(request_count)
    ]


def estimate_critic_candidate_rewards(
    *,
    actor_policy: GeneratingPolicy,
    samples: Sequence[dict[str, Any]],
    actor_responses: Sequence[str],
    candidate_feedbacks: Sequence[Mapping[str, Mapping[str, Any] | str]],
    dataset_name: str,
    rollouts: int,
    settings: RolloutSettings,
    seed: int | None = None,
) -> list[dict[str, dict[str, Any]]]:
    """Estimate rewards for Critic candidates with K sampled Actor revisions.

    A candidate Critic response is already the latest response in the partial
    trajectory. Therefore the paper's one additional deliberation step is the
    Actor's next revision. Its gold-answer accuracy is averaged over K draws.
    """
    _validate_inputs(samples, actor_responses, candidate_feedbacks, rollouts)
    prompts: list[str] = []
    repeated_samples: list[dict[str, Any]] = []
    coordinates: list[tuple[int, str]] = []
    for sample_index, (sample, actor_response, candidates) in enumerate(
        zip(samples, actor_responses, candidate_feedbacks)
    ):
        _validate_candidate_keys(candidates)
        for key in CANDIDATE_KEYS:
            feedback = _response_text(candidates[key])
            prompt = build_actor_deliberation_prompt(
                sample,
                dataset_name,
                actor_response,
                feedback,
            )
            for _ in range(rollouts):
                prompts.append(prompt)
                repeated_samples.append(sample)
                coordinates.append((sample_index, key))

    revision_seeds = derive_request_seeds(seed, len(prompts), stream=0)
    revisions = generate_actor_records(
        actor_policy,
        prompts,
        repeated_samples,
        max_tokens=settings.actor_max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        enable_thinking=settings.actor_thinking,
        seed=revision_seeds,
    )
    result = _empty_results(len(samples))
    recorded_seeds = revision_seeds or [None] * len(revisions)
    for coordinate, revision, actor_seed in zip(coordinates, revisions, recorded_seeds):
        sample_index, key = coordinate
        result[sample_index][key]["simulations"].append(
            {"actor_seed": actor_seed, "actor_revision": revision}
        )
    _finalize_rewards(result, rollouts)
    return result


def estimate_actor_candidate_rewards(
    *,
    actor_policy: GeneratingPolicy,
    critic_policy: GeneratingPolicy,
    samples: Sequence[dict[str, Any]],
    candidate_actor_responses: Sequence[Mapping[str, Mapping[str, Any] | str]],
    dataset_name: str,
    rollouts: int,
    settings: RolloutSettings,
    seed: int | None = None,
) -> list[dict[str, dict[str, Any]]]:
    """Estimate Actor-candidate rewards via Critic feedback then Actor revision.

    For each candidate Actor completion, K natural Critic feedbacks are sampled;
    each is followed by one Actor revision. Eq. 4 reward is the mean correctness
    of those revisions, not the correctness of the candidate itself.
    """
    if len(samples) != len(candidate_actor_responses):
        raise ValueError(
            "samples/candidate_actor_responses length mismatch: "
            f"{len(samples)} != {len(candidate_actor_responses)}"
        )
    if rollouts < 1:
        raise ValueError(f"rollouts must be positive, got {rollouts}")

    critic_prompts: list[str] = []
    coordinates: list[tuple[int, str]] = []
    candidate_texts: list[str] = []
    repeated_samples: list[dict[str, Any]] = []
    for sample_index, (sample, candidates) in enumerate(
        zip(samples, candidate_actor_responses)
    ):
        _validate_candidate_keys(candidates)
        for key in CANDIDATE_KEYS:
            actor_response = _response_text(candidates[key])
            prompt = build_critic_prompt(sample, dataset_name, actor_response)
            for _ in range(rollouts):
                critic_prompts.append(prompt)
                coordinates.append((sample_index, key))
                candidate_texts.append(actor_response)
                repeated_samples.append(sample)

    critic_seeds = derive_request_seeds(seed, len(critic_prompts), stream=0)
    critic_records = generate_text_records(
        critic_policy,
        critic_prompts,
        max_tokens=settings.critic_max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        enable_thinking=settings.critic_thinking,
        seed=critic_seeds,
    )
    revision_prompts = [
        build_actor_deliberation_prompt(
            sample,
            dataset_name,
            actor_response,
            str(critic_record["response"]),
        )
        for sample, actor_response, critic_record in zip(
            repeated_samples,
            candidate_texts,
            critic_records,
        )
    ]
    actor_seeds = derive_request_seeds(seed, len(revision_prompts), stream=1)
    revisions = generate_actor_records(
        actor_policy,
        revision_prompts,
        repeated_samples,
        max_tokens=settings.actor_max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        enable_thinking=settings.actor_thinking,
        seed=actor_seeds,
    )

    result = _empty_results(len(samples))
    recorded_critic_seeds = critic_seeds or [None] * len(critic_records)
    recorded_actor_seeds = actor_seeds or [None] * len(revisions)
    for coordinate, critic_record, revision, critic_seed, actor_seed in zip(
        coordinates,
        critic_records,
        revisions,
        recorded_critic_seeds,
        recorded_actor_seeds,
    ):
        sample_index, key = coordinate
        result[sample_index][key]["simulations"].append(
            {
                "critic_seed": critic_seed,
                "critic_feedback": critic_record,
                "actor_seed": actor_seed,
                "actor_revision": revision,
            }
        )
    _finalize_rewards(result, rollouts)
    return result


def _response_text(record_or_text: Mapping[str, Any] | str) -> str:
    if isinstance(record_or_text, Mapping):
        text = str(record_or_text.get("response") or "")
    else:
        text = str(record_or_text)
    return text


def _validate_candidate_keys(candidates: Mapping[str, Any]) -> None:
    missing = [key for key in CANDIDATE_KEYS if key not in candidates]
    extra = sorted(set(candidates) - set(CANDIDATE_KEYS))
    if missing or extra:
        raise ValueError(f"Invalid candidate keys: missing={missing}, extra={extra}")


def _validate_inputs(
    samples: Sequence[dict[str, Any]],
    actor_responses: Sequence[str],
    candidates: Sequence[Mapping[str, Any]],
    rollouts: int,
) -> None:
    if not (len(samples) == len(actor_responses) == len(candidates)):
        raise ValueError(
            "Critic rollout input lengths differ: "
            f"samples={len(samples)}, actor={len(actor_responses)}, "
            f"candidates={len(candidates)}"
        )
    if rollouts < 1:
        raise ValueError(f"rollouts must be positive, got {rollouts}")


def _empty_results(sample_count: int) -> list[dict[str, dict[str, Any]]]:
    return [
        {
            key: {
                "reward": 0.0,
                "correct_count": 0,
                "rollout_count": 0,
                "simulations": [],
            }
            for key in CANDIDATE_KEYS
        }
        for _ in range(sample_count)
    ]


def _finalize_rewards(
    results: list[dict[str, dict[str, Any]]],
    expected_rollouts: int,
) -> None:
    for sample_result in results:
        for key in CANDIDATE_KEYS:
            simulations = sample_result[key]["simulations"]
            if len(simulations) != expected_rollouts:
                raise RuntimeError(
                    f"Candidate {key} has {len(simulations)} simulations; "
                    f"expected {expected_rollouts}"
                )
            correct_count = sum(
                int(bool(simulation["actor_revision"].get("correct")))
                for simulation in simulations
            )
            sample_result[key]["correct_count"] = correct_count
            sample_result[key]["rollout_count"] = expected_rollouts
            sample_result[key]["reward"] = correct_count / expected_rollouts
