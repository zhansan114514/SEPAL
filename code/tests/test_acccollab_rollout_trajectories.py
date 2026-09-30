from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest

from src.acccollab.generation import generate_text_records
from src.acccollab.pairs import CANDIDATE_KEYS
from src.acccollab.rollout import (
    RolloutSettings,
    derive_request_seeds,
    estimate_actor_candidate_rewards,
    estimate_critic_candidate_rewards,
)
from src.acccollab.trajectories import (
    TrajectorySettings,
    generate_actor_preference_batch,
    generate_critic_preference_batch,
)
import src.acccollab.trajectories as trajectories_module


def _sample() -> dict[str, Any]:
    return {
        "sample_id": "sample-0",
        "acccollab_sample_index": 0,
        "task_type": "multiple_choice",
        "question": "Pick the correct letter",
        "choices": ["one", "two", "three", "four"],
        "choice_labels": ["A", "B", "C", "D"],
        "answer": "B",
    }


def _rollout_settings() -> RolloutSettings:
    return RolloutSettings(
        actor_max_tokens=64,
        critic_max_tokens=32,
        temperature=0.7,
        top_p=0.9,
    )


class _RevisionPolicy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.counts: defaultdict[str, int] = defaultdict(int)

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        self.calls.append({"prompts": list(prompts), **kwargs})
        outputs = []
        for prompt in prompts:
            if "guided positive feedback" in prompt:
                key = "positive"
                answer = "B"
            elif "guided negative feedback" in prompt:
                key = "negative"
                answer = "A"
            elif "natural feedback" in prompt:
                key = "natural"
                answer = "B" if self.counts[key] == 0 else "A"
            elif "guided positive actor" in prompt:
                key = "positive-actor"
                answer = "B"
            elif "guided negative actor" in prompt:
                key = "negative-actor"
                answer = "A"
            elif "natural actor" in prompt:
                key = "natural-actor"
                answer = "B" if self.counts[key] == 0 else "A"
            else:  # pragma: no cover - makes unexpected prompt construction explicit
                raise AssertionError(f"Unexpected rollout prompt: {prompt}")
            self.counts[key] += 1
            outputs.append(f"brief reason. Final Answer: {answer}")
        return outputs


class _NaturalCriticPolicy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        self.calls.append({"prompts": list(prompts), **kwargs})
        return [f"natural rollout critic feedback for prompt {index}" for index, _ in enumerate(prompts)]


class _EmptyThenValidPolicy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        self.calls.append({"prompts": list(prompts), **kwargs})
        if len(self.calls) == 1:
            return ["", "valid second response"]
        return ["valid retry response" for _prompt in prompts]


class _AlwaysEmptyPolicy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        self.calls.append({"prompts": list(prompts), **kwargs})
        return ["" for _prompt in prompts]


def test_generation_retries_only_empty_responses_with_new_seeds() -> None:
    policy = _EmptyThenValidPolicy()
    records = generate_text_records(
        policy,
        ["first prompt", "second prompt"],
        max_tokens=32,
        temperature=0.7,
        top_p=0.9,
        enable_thinking=False,
        seed=[101, 202],
    )

    assert [call["prompts"] for call in policy.calls] == [
        ["first prompt", "second prompt"],
        ["first prompt"],
    ]
    assert policy.calls[1]["seed"] != [101]
    assert [record["response"] for record in records] == [
        "valid retry response",
        "valid second response",
    ]
    assert [record["empty_response_retries"] for record in records] == [1, 0]
    assert all(record["empty_response_exhausted"] is False for record in records)


def test_generation_records_exhausted_empty_response_without_raising() -> None:
    policy = _AlwaysEmptyPolicy()
    records = generate_text_records(
        policy,
        ["only prompt"],
        max_tokens=32,
        temperature=0.7,
        top_p=0.9,
        enable_thinking=False,
        seed=303,
    )

    assert len(policy.calls) == 3
    assert len({call["seed"][0] for call in policy.calls[1:]}) == 2
    assert records[0]["response"] == ""
    assert records[0]["empty_response_retries"] == 2
    assert records[0]["empty_response_exhausted"] is True


def test_critic_eq4_reward_scores_only_next_actor_revision() -> None:
    actor = _RevisionPolicy()
    results = estimate_critic_candidate_rewards(
        actor_policy=actor,
        samples=[_sample()],
        actor_responses=["initial actor response"],
        candidate_feedbacks=[
            {
                "natural": "natural feedback",
                "guided_positive": "guided positive feedback",
                "guided_negative": "guided negative feedback",
            }
        ],
        dataset_name="mmlu",
        rollouts=2,
        settings=_rollout_settings(),
        seed=101,
    )

    assert len(actor.calls) == 1
    assert len(actor.calls[0]["prompts"]) == len(CANDIDATE_KEYS) * 2
    expected_seeds = derive_request_seeds(101, len(CANDIDATE_KEYS) * 2, stream=0)
    assert actor.calls[0]["seed"] == expected_seeds
    assert expected_seeds is not None
    assert len(set(expected_seeds)) == len(expected_seeds)
    assert results[0]["natural"]["reward"] == 0.5
    assert results[0]["guided_positive"]["reward"] == 1.0
    assert results[0]["guided_negative"]["reward"] == 0.0
    for key in CANDIDATE_KEYS:
        candidate = results[0][key]
        assert candidate["rollout_count"] == 2
        assert len(candidate["simulations"]) == 2
        assert all(
            set(simulation) == {"actor_seed", "actor_revision"}
            for simulation in candidate["simulations"]
        )


def test_eq4_request_seeds_are_stable_unique_and_stream_separated() -> None:
    first = derive_request_seeds(123, 12, stream=0)
    repeated = derive_request_seeds(123, 12, stream=0)
    changed_base = derive_request_seeds(124, 12, stream=0)
    second_stream = derive_request_seeds(123, 12, stream=1)

    assert first == repeated
    assert first != changed_base
    assert first is not None
    assert second_stream is not None
    assert len(first) == len(set(first)) == 12
    assert set(first).isdisjoint(second_stream)
    assert derive_request_seeds(None, 12, stream=0) is None


def test_actor_eq4_reward_runs_natural_critic_then_actor_revision() -> None:
    actor = _RevisionPolicy()
    critic = _NaturalCriticPolicy()
    results = estimate_actor_candidate_rewards(
        actor_policy=actor,
        critic_policy=critic,
        samples=[_sample()],
        candidate_actor_responses=[
            {
                "natural": "natural actor",
                "guided_positive": "guided positive actor",
                "guided_negative": "guided negative actor",
            }
        ],
        dataset_name="mmlu",
        rollouts=2,
        settings=_rollout_settings(),
        seed=201,
    )

    assert len(critic.calls) == 1
    assert len(critic.calls[0]["prompts"]) == len(CANDIDATE_KEYS) * 2
    critic_seeds = derive_request_seeds(201, len(CANDIDATE_KEYS) * 2, stream=0)
    actor_seeds = derive_request_seeds(201, len(CANDIDATE_KEYS) * 2, stream=1)
    assert critic.calls[0]["seed"] == critic_seeds
    assert len(actor.calls) == 1
    assert len(actor.calls[0]["prompts"]) == len(CANDIDATE_KEYS) * 2
    assert actor.calls[0]["seed"] == actor_seeds
    assert critic_seeds is not None
    assert actor_seeds is not None
    assert set(critic_seeds).isdisjoint(actor_seeds)
    assert results[0]["natural"]["reward"] == 0.5
    assert results[0]["guided_positive"]["reward"] == 1.0
    assert results[0]["guided_negative"]["reward"] == 0.0
    for key in CANDIDATE_KEYS:
        simulations = results[0][key]["simulations"]
        assert len(simulations) == 2
        assert all(
            set(simulation)
            == {
                "critic_seed",
                "critic_feedback",
                "actor_seed",
                "actor_revision",
            }
            for simulation in simulations
        )


@pytest.mark.parametrize("estimator", ["actor", "critic"])
def test_eq4_rejects_missing_or_extra_candidate_keys(estimator: str) -> None:
    candidates = {
        "natural": "natural actor" if estimator == "actor" else "natural feedback",
        "guided_positive": (
            "guided positive actor" if estimator == "actor" else "guided positive feedback"
        ),
        "unexpected": "not part of Eq. 4",
    }
    with pytest.raises(ValueError, match="Invalid candidate keys"):
        if estimator == "critic":
            estimate_critic_candidate_rewards(
                actor_policy=_RevisionPolicy(),
                samples=[_sample()],
                actor_responses=["actor"],
                candidate_feedbacks=[candidates],
                dataset_name="mmlu",
                rollouts=2,
                settings=_rollout_settings(),
            )
        else:
            estimate_actor_candidate_rewards(
                actor_policy=_RevisionPolicy(),
                critic_policy=_NaturalCriticPolicy(),
                samples=[_sample()],
                candidate_actor_responses=[candidates],
                dataset_name="mmlu",
                rollouts=2,
                settings=_rollout_settings(),
            )


def _trajectory_settings() -> TrajectorySettings:
    return TrajectorySettings(
        deliberation_rounds=5,
        rollouts=2,
        epsilon=0.5,
        actor_max_tokens=64,
        critic_max_tokens=32,
        temperature=0.7,
        top_p=0.9,
    )


def _completion(response: str, *, correct: bool = False) -> dict[str, Any]:
    return {
        "response": response,
        "raw_response": response,
        "correct": correct,
        "parsed": True,
        "truncated": False,
    }


def _install_fake_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    actor_calls = {"count": 0}
    critic_calls = {"count": 0}

    def fake_actor_records(
        _policy: object,
        prompts: list[str],
        samples: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        assert len(prompts) == len(samples)
        call = actor_calls["count"]
        actor_calls["count"] += 1
        return [
            _completion(f"actor-call-{call}-item-{index}")
            for index, _prompt in enumerate(prompts)
        ]

    def fake_text_records(
        _policy: object,
        prompts: list[str],
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        call = critic_calls["count"]
        critic_calls["count"] += 1
        return [
            _completion(f"critic-call-{call}-item-{index}")
            for index, _prompt in enumerate(prompts)
        ]

    monkeypatch.setattr(trajectories_module, "generate_actor_records", fake_actor_records)
    monkeypatch.setattr(trajectories_module, "generate_text_records", fake_text_records)


def _fake_rewards(samples: list[dict[str, Any]], *, include_critic: bool) -> list[dict[str, Any]]:
    result = []
    for _sample_record in samples:
        candidates: dict[str, Any] = {}
        for key, reward in {
            "natural": 0.0,
            "guided_positive": 1.0,
            "guided_negative": 0.0,
        }.items():
            simulation: dict[str, Any] = {
                "actor_revision": _completion("rollout-only-actor", correct=bool(reward)),
            }
            if include_critic:
                simulation["critic_feedback"] = _completion("rollout-only-critic")
            candidates[key] = {
                "reward": reward,
                "correct_count": int(reward * 2),
                "rollout_count": 2,
                "simulations": [dict(simulation), dict(simulation)],
            }
        result.append(candidates)
    return result


def test_actor_trajectory_uses_independent_natural_critic_spine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_generation(monkeypatch)

    def fake_actor_rewards(**kwargs: Any) -> list[dict[str, Any]]:
        return _fake_rewards(list(kwargs["samples"]), include_critic=True)

    monkeypatch.setattr(
        trajectories_module,
        "estimate_actor_candidate_rewards",
        fake_actor_rewards,
    )
    generated, pairs = generate_actor_preference_batch(
        actor_policy=object(),
        critic_policy=object(),
        samples=[_sample()],
        dataset_name="mmlu",
        iteration=1,
        settings=_trajectory_settings(),
        seed=7,
    )

    assert len(generated) == 1
    trajectory = generated[0]
    assert [record["round"] for record in trajectory["rounds"]] == list(range(5))
    assert trajectory["selected_pair_count"] == 4
    assert len(pairs) == 4
    assert {pair["metadata"]["round"] for pair in pairs} == {1, 2, 3, 4}
    for round_record in trajectory["rounds"][1:]:
        natural_critic = round_record["natural_critic"]
        assert natural_critic["source"] == "independent_natural_spine_generation"
        assert natural_critic["completion"]["response"] != "rollout-only-critic"

    round_one_critic = trajectory["rounds"][1]["natural_critic"]["completion"]["response"]
    round_two_prompt = trajectory["rounds"][2]["actor_candidate_prompts"]["natural"]
    assert round_one_critic in round_two_prompt


def test_critic_trajectory_advances_with_natural_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_generation(monkeypatch)

    def fake_critic_rewards(**kwargs: Any) -> list[dict[str, Any]]:
        return _fake_rewards(list(kwargs["samples"]), include_critic=False)

    monkeypatch.setattr(
        trajectories_module,
        "estimate_critic_candidate_rewards",
        fake_critic_rewards,
    )
    generated, pairs = generate_critic_preference_batch(
        actor_policy=object(),
        critic_policy=object(),
        samples=[_sample()],
        dataset_name="mmlu",
        iteration=1,
        settings=_trajectory_settings(),
        seed=9,
    )

    trajectory = generated[0]
    assert [record["round"] for record in trajectory["rounds"]] == list(range(5))
    assert len(pairs) == 4
    assert len({pair["metadata"]["round"] for pair in pairs}) == len(pairs)
    natural_feedback = trajectory["rounds"][1]["critic_candidates"]["natural"]["response"]
    next_actor_prompt = trajectory["rounds"][2]["natural_actor"]["prompt"]
    assert natural_feedback in next_actor_prompt


@pytest.mark.parametrize("agent", ["actor", "critic"])
def test_trajectory_drops_eq5_pair_when_completions_are_identical(
    agent: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def identical_actor_records(
        _policy: object,
        prompts: list[str],
        samples: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        assert len(prompts) == len(samples)
        return [_completion("identical actor completion") for _prompt in prompts]

    def identical_text_records(
        _policy: object,
        prompts: list[str],
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        return [_completion("identical critic completion") for _prompt in prompts]

    monkeypatch.setattr(
        trajectories_module,
        "generate_actor_records",
        identical_actor_records,
    )
    monkeypatch.setattr(
        trajectories_module,
        "generate_text_records",
        identical_text_records,
    )

    if agent == "actor":
        monkeypatch.setattr(
            trajectories_module,
            "estimate_actor_candidate_rewards",
            lambda **kwargs: _fake_rewards(list(kwargs["samples"]), include_critic=True),
        )
        generate = generate_actor_preference_batch
    else:
        monkeypatch.setattr(
            trajectories_module,
            "estimate_critic_candidate_rewards",
            lambda **kwargs: _fake_rewards(list(kwargs["samples"]), include_critic=False),
        )
        generate = generate_critic_preference_batch

    generated, pairs = generate(
        actor_policy=object(),
        critic_policy=object(),
        samples=[_sample()],
        dataset_name="mmlu",
        iteration=1,
        settings=_trajectory_settings(),
        seed=11,
    )

    assert pairs == []
    assert generated[0]["selected_pair_count"] == 0
    for round_record in generated[0]["rounds"][1:]:
        selection = round_record["eq5_selection"]
        assert selection["selected"] is False
        assert selection["eq5_threshold_passed"] is True
        assert selection["drop_reason"] == "identical_chosen_rejected"


@pytest.mark.parametrize("agent", ["actor", "critic"])
def test_trajectory_drops_pair_when_any_candidate_remains_empty(
    agent: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def actor_records(
        _policy: object,
        prompts: list[str],
        samples: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        assert len(prompts) == len(samples)
        responses = (
            ["natural actor", "", "guided negative actor"]
            if agent == "actor" and len(prompts) == len(CANDIDATE_KEYS)
            else ["valid actor" for _prompt in prompts]
        )
        return [_completion(response) for response in responses]

    def text_records(
        _policy: object,
        prompts: list[str],
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        responses = (
            ["natural critic", "", "guided negative critic"]
            if agent == "critic" and len(prompts) == len(CANDIDATE_KEYS)
            else ["valid critic" for _prompt in prompts]
        )
        return [_completion(response) for response in responses]

    monkeypatch.setattr(trajectories_module, "generate_actor_records", actor_records)
    monkeypatch.setattr(trajectories_module, "generate_text_records", text_records)
    if agent == "actor":
        monkeypatch.setattr(
            trajectories_module,
            "estimate_actor_candidate_rewards",
            lambda **kwargs: _fake_rewards(list(kwargs["samples"]), include_critic=True),
        )
        generate = generate_actor_preference_batch
    else:
        monkeypatch.setattr(
            trajectories_module,
            "estimate_critic_candidate_rewards",
            lambda **kwargs: _fake_rewards(list(kwargs["samples"]), include_critic=False),
        )
        generate = generate_critic_preference_batch

    generated, pairs = generate(
        actor_policy=object(),
        critic_policy=object(),
        samples=[_sample()],
        dataset_name="mmlu",
        iteration=1,
        settings=_trajectory_settings(),
        seed=13,
    )

    assert pairs == []
    assert generated[0]["selected_pair_count"] == 0
    for round_record in generated[0]["rounds"][1:]:
        selection = round_record["eq5_selection"]
        assert selection["selected"] is False
        assert selection["eq5_threshold_passed"] is True
        assert selection["drop_reason"] == "empty_candidate_response"
        assert selection["invalid_candidate_keys"] == ["guided_positive"]
