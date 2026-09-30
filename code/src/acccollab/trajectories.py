"""Batched off-policy trajectory generation for original ACC-Collab Algorithm 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.acccollab.generation import (
    GeneratingPolicy,
    generate_actor_records,
    generate_text_records,
    guidance_answer,
    wrong_guidance_answer,
)
from src.acccollab.pairs import (
    CANDIDATE_KEYS,
    build_dpo_pair,
    pair_completions_are_distinct,
    select_eq5_pair,
    selection_record,
)
from src.acccollab.prompts import (
    build_actor_deliberation_prompt,
    build_critic_prompt,
    build_initial_actor_prompt,
    prompt_version_for_sample,
)
from src.acccollab.rollout import (
    RolloutSettings,
    estimate_actor_candidate_rewards,
    estimate_critic_candidate_rewards,
)


@dataclass(frozen=True)
class TrajectorySettings:
    """Paper-fixed structure plus generation settings for one preference stage."""

    deliberation_rounds: int
    rollouts: int
    epsilon: float
    actor_max_tokens: int
    critic_max_tokens: int
    temperature: float
    top_p: float
    actor_thinking: bool | None = False
    critic_thinking: bool | None = False

    def validate(self) -> None:
        if self.deliberation_rounds != 5:
            raise ValueError("Original ACC-Collab trajectories require exactly five rounds")
        if self.rollouts < 2:
            raise ValueError("Original ACC-Collab reward estimation requires multiple rollouts")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")

    def rollout_settings(self) -> RolloutSettings:
        return RolloutSettings(
            actor_max_tokens=self.actor_max_tokens,
            critic_max_tokens=self.critic_max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            actor_thinking=self.actor_thinking,
            critic_thinking=self.critic_thinking,
        )


def generate_critic_preference_batch(
    *,
    actor_policy: GeneratingPolicy,
    critic_policy: GeneratingPolicy,
    samples: Sequence[dict[str, Any]],
    dataset_name: str,
    iteration: int,
    settings: TrajectorySettings,
    seed: int | None = None,
    policy_provenance: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect Critic DPO pairs while advancing a fully natural trajectory spine.

    At each ``t=1..4`` the current Actor first produces one natural revision. The
    Critic then produces natural, guided-positive, and guided-negative candidates
    for that same Actor response. Eq. 4 evaluates each candidate by sampling the
    *next Actor revision* K times. Eq. 5 selects at most one pair, and the natural
    Critic response advances the spine regardless of which pair was selected.
    """
    settings.validate()
    samples = [dict(sample) for sample in samples]
    if not samples:
        return [], []
    trajectories, previous_actor, previous_critic = _initial_natural_round(
        actor_policy=actor_policy,
        critic_policy=critic_policy,
        samples=samples,
        dataset_name=dataset_name,
        iteration=iteration,
        stage="critic_dpo_data",
        settings=settings,
        seed=seed,
        policy_provenance=policy_provenance,
    )
    pairs: list[dict[str, Any]] = []
    rollout_settings = settings.rollout_settings()

    for round_index in range(1, settings.deliberation_rounds):
        actor_prompts = [
            build_actor_deliberation_prompt(
                sample,
                dataset_name,
                actor_response,
                critic_response,
            )
            for sample, actor_response, critic_response in zip(
                samples,
                previous_actor,
                previous_critic,
            )
        ]
        natural_actors = generate_actor_records(
            actor_policy,
            actor_prompts,
            samples,
            max_tokens=settings.actor_max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            enable_thinking=settings.actor_thinking,
            seed=_offset_seed(seed, round_index, 10),
        )
        actor_texts = [str(record["response"]) for record in natural_actors]

        candidate_prompts: list[dict[str, str]] = []
        for sample, actor_response in zip(samples, actor_texts):
            candidate_prompts.append(
                {
                    "natural": build_critic_prompt(sample, dataset_name, actor_response),
                    "guided_positive": build_critic_prompt(
                        sample,
                        dataset_name,
                        actor_response,
                        target_answer=guidance_answer(sample),
                    ),
                    "guided_negative": build_critic_prompt(
                        sample,
                        dataset_name,
                        actor_response,
                        target_answer=wrong_guidance_answer(
                            sample,
                            seed=_offset_seed(seed, round_index, 5),
                        ),
                    ),
                }
            )
        candidate_feedbacks = _generate_text_candidates(
            critic_policy,
            candidate_prompts,
            max_tokens=settings.critic_max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            enable_thinking=settings.critic_thinking,
            seed=_offset_seed(seed, round_index, 20),
        )
        rewards = estimate_critic_candidate_rewards(
            actor_policy=actor_policy,
            samples=samples,
            actor_responses=actor_texts,
            candidate_feedbacks=candidate_feedbacks,
            dataset_name=dataset_name,
            rollouts=settings.rollouts,
            settings=rollout_settings,
            seed=_offset_seed(seed, round_index, 30),
        )

        for sample_index, sample in enumerate(samples):
            sample_rewards = rewards[sample_index]
            reward_values = _reward_values(sample_rewards)
            decision = select_eq5_pair(epsilon=settings.epsilon, **reward_values)
            selection = selection_record(epsilon=settings.epsilon, **reward_values)
            invalid_candidate_keys = _invalid_candidate_keys(
                candidate_feedbacks[sample_index]
            )
            if invalid_candidate_keys:
                selection = _mark_empty_candidate_drop(
                    selection,
                    invalid_candidate_keys,
                )
                decision = None
            elif decision is not None and not pair_completions_are_distinct(
                candidate_feedbacks[sample_index], decision
            ):
                selection = _mark_non_distinct_drop(selection)
                decision = None
            round_record = {
                "round": round_index,
                "natural_actor": {
                    "prompt": actor_prompts[sample_index],
                    "completion": natural_actors[sample_index],
                },
                "critic_candidate_prompts": candidate_prompts[sample_index],
                "critic_candidates": candidate_feedbacks[sample_index],
                "eq4_rewards": sample_rewards,
                "eq5_selection": selection,
            }
            trajectories[sample_index]["rounds"].append(round_record)
            if decision is not None:
                pairs.append(
                    build_dpo_pair(
                        natural_prompt=candidate_prompts[sample_index]["natural"],
                        candidate_records=candidate_feedbacks[sample_index],
                        decision=decision,
                        agent="critic",
                        iteration=iteration,
                        sample_id=str(sample["sample_id"]),
                        round_index=round_index,
                        rollouts=settings.rollouts,
                        prompt_version=prompt_version_for_sample(sample),
                        sample_index=int(sample["acccollab_sample_index"]),
                    )
                )

        previous_actor = actor_texts
        previous_critic = [
            str(candidates["natural"]["response"]) for candidates in candidate_feedbacks
        ]

    _finalize_trajectory_pair_counts(trajectories, pairs)
    return trajectories, pairs


def generate_actor_preference_batch(
    *,
    actor_policy: GeneratingPolicy,
    critic_policy: GeneratingPolicy,
    samples: Sequence[dict[str, Any]],
    dataset_name: str,
    iteration: int,
    settings: TrajectorySettings,
    seed: int | None = None,
    policy_provenance: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect Actor DPO pairs with natural Critic responses on the spine.

    For each ``t=1..4`` three Actor candidates share the same previous natural
    Actor/Critic state. Each candidate's Eq. 4 reward samples a natural Critic
    response followed by one Actor revision. A separate natural Critic response
    is then generated for the natural Actor candidate to advance the spine; no
    Monte Carlo simulation is reused as on-policy state.
    """
    settings.validate()
    samples = [dict(sample) for sample in samples]
    if not samples:
        return [], []
    trajectories, previous_actor, previous_critic = _initial_natural_round(
        actor_policy=actor_policy,
        critic_policy=critic_policy,
        samples=samples,
        dataset_name=dataset_name,
        iteration=iteration,
        stage="actor_dpo_data",
        settings=settings,
        seed=seed,
        policy_provenance=policy_provenance,
    )
    pairs: list[dict[str, Any]] = []
    rollout_settings = settings.rollout_settings()

    for round_index in range(1, settings.deliberation_rounds):
        candidate_prompts: list[dict[str, str]] = []
        for sample, actor_response, critic_response in zip(
            samples,
            previous_actor,
            previous_critic,
        ):
            candidate_prompts.append(
                {
                    "natural": build_actor_deliberation_prompt(
                        sample,
                        dataset_name,
                        actor_response,
                        critic_response,
                    ),
                    "guided_positive": build_actor_deliberation_prompt(
                        sample,
                        dataset_name,
                        actor_response,
                        critic_response,
                        target_answer=guidance_answer(sample),
                    ),
                    "guided_negative": build_actor_deliberation_prompt(
                        sample,
                        dataset_name,
                        actor_response,
                        critic_response,
                        target_answer=wrong_guidance_answer(
                            sample,
                            seed=_offset_seed(seed, round_index, 5),
                        ),
                    ),
                }
            )
        actor_candidates = _generate_actor_candidates(
            actor_policy,
            candidate_prompts,
            samples,
            max_tokens=settings.actor_max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            enable_thinking=settings.actor_thinking,
            seed=_offset_seed(seed, round_index, 10),
        )
        rewards = estimate_actor_candidate_rewards(
            actor_policy=actor_policy,
            critic_policy=critic_policy,
            samples=samples,
            candidate_actor_responses=actor_candidates,
            dataset_name=dataset_name,
            rollouts=settings.rollouts,
            settings=rollout_settings,
            seed=_offset_seed(seed, round_index, 20),
        )

        natural_actor_texts = [
            str(candidates["natural"]["response"]) for candidates in actor_candidates
        ]
        natural_critic_prompts = [
            build_critic_prompt(sample, dataset_name, actor_response)
            for sample, actor_response in zip(samples, natural_actor_texts)
        ]
        natural_critics = generate_text_records(
            critic_policy,
            natural_critic_prompts,
            max_tokens=settings.critic_max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            enable_thinking=settings.critic_thinking,
            seed=_offset_seed(seed, round_index, 30),
        )

        for sample_index, sample in enumerate(samples):
            sample_rewards = rewards[sample_index]
            reward_values = _reward_values(sample_rewards)
            decision = select_eq5_pair(epsilon=settings.epsilon, **reward_values)
            selection = selection_record(epsilon=settings.epsilon, **reward_values)
            invalid_candidate_keys = _invalid_candidate_keys(
                actor_candidates[sample_index]
            )
            if invalid_candidate_keys:
                selection = _mark_empty_candidate_drop(
                    selection,
                    invalid_candidate_keys,
                )
                decision = None
            elif decision is not None and not pair_completions_are_distinct(
                actor_candidates[sample_index], decision
            ):
                selection = _mark_non_distinct_drop(selection)
                decision = None
            trajectories[sample_index]["rounds"].append(
                {
                    "round": round_index,
                    "actor_candidate_prompts": candidate_prompts[sample_index],
                    "actor_candidates": actor_candidates[sample_index],
                    "eq4_rewards": sample_rewards,
                    "eq5_selection": selection,
                    "natural_critic": {
                        "prompt": natural_critic_prompts[sample_index],
                        "completion": natural_critics[sample_index],
                        "source": "independent_natural_spine_generation",
                    },
                }
            )
            if decision is not None:
                pairs.append(
                    build_dpo_pair(
                        natural_prompt=candidate_prompts[sample_index]["natural"],
                        candidate_records=actor_candidates[sample_index],
                        decision=decision,
                        agent="actor",
                        iteration=iteration,
                        sample_id=str(sample["sample_id"]),
                        round_index=round_index,
                        rollouts=settings.rollouts,
                        prompt_version=prompt_version_for_sample(sample),
                        sample_index=int(sample["acccollab_sample_index"]),
                    )
                )

        previous_actor = natural_actor_texts
        previous_critic = [str(record["response"]) for record in natural_critics]

    _finalize_trajectory_pair_counts(trajectories, pairs)
    return trajectories, pairs


def _initial_natural_round(
    *,
    actor_policy: GeneratingPolicy,
    critic_policy: GeneratingPolicy,
    samples: list[dict[str, Any]],
    dataset_name: str,
    iteration: int,
    stage: str,
    settings: TrajectorySettings,
    seed: int | None,
    policy_provenance: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    actor_prompts = [build_initial_actor_prompt(sample, dataset_name) for sample in samples]
    actors = generate_actor_records(
        actor_policy,
        actor_prompts,
        samples,
        max_tokens=settings.actor_max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        enable_thinking=settings.actor_thinking,
        seed=_offset_seed(seed, 0, 0),
    )
    actor_texts = [str(record["response"]) for record in actors]
    critic_prompts = [
        build_critic_prompt(sample, dataset_name, actor_response)
        for sample, actor_response in zip(samples, actor_texts)
    ]
    critics = generate_text_records(
        critic_policy,
        critic_prompts,
        max_tokens=settings.critic_max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        enable_thinking=settings.critic_thinking,
        seed=_offset_seed(seed, 0, 1),
    )
    trajectories = []
    for index, sample in enumerate(samples):
        trajectories.append(
            {
                "schema_version": 1,
                "pipeline": "acccollab_original",
                "stage": stage,
                "iteration": int(iteration),
                "sample_id": str(sample["sample_id"]),
                "sample": sample,
                "prompt_version": prompt_version_for_sample(sample),
                "settings": asdict(settings),
                "policy_provenance": dict(policy_provenance or {}),
                "rounds": [
                    {
                        "round": 0,
                        "natural_actor": {
                            "prompt": actor_prompts[index],
                            "completion": actors[index],
                        },
                        "natural_critic": {
                            "prompt": critic_prompts[index],
                            "completion": critics[index],
                        },
                    }
                ],
            }
        )
    return trajectories, actor_texts, [str(record["response"]) for record in critics]


def _generate_text_candidates(
    policy: GeneratingPolicy,
    prompt_maps: Sequence[Mapping[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool | None,
    seed: int | None,
) -> list[dict[str, dict[str, Any]]]:
    flat_prompts = [prompt_map[key] for prompt_map in prompt_maps for key in CANDIDATE_KEYS]
    flat_records = generate_text_records(
        policy,
        flat_prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
        seed=seed,
    )
    return _unflatten_candidates(flat_records, len(prompt_maps))


def _generate_actor_candidates(
    policy: GeneratingPolicy,
    prompt_maps: Sequence[Mapping[str, str]],
    samples: Sequence[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool | None,
    seed: int | None,
) -> list[dict[str, dict[str, Any]]]:
    flat_prompts = [prompt_map[key] for prompt_map in prompt_maps for key in CANDIDATE_KEYS]
    repeated_samples = [sample for sample in samples for _key in CANDIDATE_KEYS]
    flat_records = generate_actor_records(
        policy,
        flat_prompts,
        repeated_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
        seed=seed,
    )
    return _unflatten_candidates(flat_records, len(prompt_maps))


def _unflatten_candidates(
    records: Sequence[dict[str, Any]],
    sample_count: int,
) -> list[dict[str, dict[str, Any]]]:
    expected = sample_count * len(CANDIDATE_KEYS)
    if len(records) != expected:
        raise RuntimeError(f"Candidate output count mismatch: {len(records)} != {expected}")
    result: list[dict[str, dict[str, Any]]] = []
    for sample_index in range(sample_count):
        offset = sample_index * len(CANDIDATE_KEYS)
        result.append(
            {
                key: records[offset + key_index]
                for key_index, key in enumerate(CANDIDATE_KEYS)
            }
        )
    return result


def _mark_non_distinct_drop(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Retain an Eq. 5 threshold hit while dropping a zero-signal DPO pair."""
    return {
        **dict(selection),
        "selected": False,
        "eq5_threshold_passed": True,
        "drop_reason": "identical_chosen_rejected",
    }


def _invalid_candidate_keys(
    candidate_records: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Return candidate names whose cleaned completion is empty."""
    return [
        key
        for key in CANDIDATE_KEYS
        if not str(candidate_records[key].get("response") or "").strip()
    ]


def _mark_empty_candidate_drop(
    selection: Mapping[str, Any],
    invalid_candidate_keys: Sequence[str],
) -> dict[str, Any]:
    """Audit an invalid Eq. 5 comparison without emitting a DPO pair."""
    return {
        **dict(selection),
        "selected": False,
        "drop_reason": "empty_candidate_response",
        "invalid_candidate_keys": list(invalid_candidate_keys),
    }


def _reward_values(sample_rewards: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    return {
        "natural_reward": float(sample_rewards["natural"]["reward"]),
        "guided_positive_reward": float(sample_rewards["guided_positive"]["reward"]),
        "guided_negative_reward": float(sample_rewards["guided_negative"]["reward"]),
    }


def _offset_seed(seed: int | None, round_index: int, offset: int) -> int | None:
    return None if seed is None else int(seed) + int(round_index) * 100 + int(offset)


def _finalize_trajectory_pair_counts(
    trajectories: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> None:
    counts: dict[str, int] = {}
    for pair in pairs:
        sample_id = str(pair["metadata"]["sample_id"])
        counts[sample_id] = counts.get(sample_id, 0) + 1
    for trajectory in trajectories:
        trajectory["selected_pair_count"] = counts.get(str(trajectory["sample_id"]), 0)
