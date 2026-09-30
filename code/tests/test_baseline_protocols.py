from __future__ import annotations

from typing import Any

from src.baselines.protocols import (
    BaselineSettings,
    build_peer_debate_prompt,
    direct_records_from_debate,
    generate_actor_critic_debate_batch,
    generate_som_batch,
    score_actor_critic_debate,
    score_direct,
    score_som,
)


class FakePolicy:
    def __init__(self, answer: str = "A") -> None:
        self.answer = answer
        self.calls: list[tuple[list[str], Any]] = []

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        self.calls.append((list(prompts), kwargs.get("seed")))
        return [f"Brief reason.\nFinal Answer: {self.answer}" for _ in prompts]


def _sample(index: int = 0) -> dict[str, Any]:
    return {
        "sample_id": f"sample-{index}",
        "acccollab_sample_index": index,
        "question": "Which option is correct?",
        "choices": ["first", "second", "third", "fourth"],
        "choice_labels": ["A", "B", "C", "D"],
        "answer": "A",
        "task_type": "multiple_choice",
    }


def test_peer_prompt_contains_every_previous_response_and_options() -> None:
    prompt = build_peer_debate_prompt(_sample(), "mmlu", ["answer one", "answer two"])
    assert "Person 0 said: answer one" in prompt
    assert "Person 1 said: answer two" in prompt
    assert "A: first" in prompt
    assert 'Final Answer:' in prompt


def test_som_generates_five_rounds_and_scores_mean_agent_accuracy() -> None:
    policy = FakePolicy()
    records = generate_som_batch(
        policy=policy,
        samples=[_sample(0), _sample(1)],
        dataset_name="mmlu",
        agents=2,
        settings=BaselineSettings(max_tokens=32),
        seed=42,
    )
    assert len(policy.calls) == 5
    assert len(records) == 2
    assert [round_record["round"] for round_record in records[0]["rounds"]] == list(
        range(5)
    )
    assert all(len(round_record["agents"]) == 2 for round_record in records[0]["rounds"])
    metrics = score_som(records, agents=2)
    assert metrics["final"]["accuracy"] == 1.0
    assert metrics["final"]["agent_decisions"] == 4
    assert metrics["final"]["plurality_vote_accuracy"] == 1.0


def test_untrained_debate_round_zero_is_reused_as_direct() -> None:
    policy = FakePolicy()
    debate = generate_actor_critic_debate_batch(
        policy=policy,
        samples=[_sample()],
        dataset_name="mmlu",
        settings=BaselineSettings(max_tokens=32),
        seed=7,
    )
    assert len(policy.calls) == 10
    debate_metrics = score_actor_critic_debate(debate)
    assert debate_metrics["final"]["accuracy"] == 1.0
    direct = direct_records_from_debate(debate)
    assert direct[0]["reused_from"] == "actor_critic_debate_round_0"
    assert direct[0]["completion"] == debate[0]["rounds"][0]["actor"]["completion"]
    assert score_direct(direct)["accuracy"] == 1.0


def test_scoring_reparses_raw_response_instead_of_trusting_stale_flags() -> None:
    record = {
        "schema_version": 1,
        "pipeline": "inference_baseline",
        "method": "direct",
        "protocol_version": "acccollab_inference_baselines_v1",
        "sample_id": "boolq-0",
        "sample": {
            "sample_id": "boolq-0",
            "question": "A question?",
            "answer": "NO",
            "task_type": "yes_no",
        },
        "completion": {
            "raw_response": 'The answer to the question "A question?" is No.',
            "response": 'The answer to the question "A question?" is No.',
            "parsed": False,
            "correct": False,
            "truncated": False,
        },
    }
    metrics = score_direct([record])
    assert metrics["parse_rate"] == 1.0
    assert metrics["accuracy"] == 1.0
