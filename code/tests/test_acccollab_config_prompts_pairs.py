from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.acccollab.config import ACCCollabConfigError, load_acccollab_config
from src.acccollab.data import expand_preference_trials
from src.acccollab.generation import guidance_answer, wrong_guidance_answer
from src.acccollab.pairs import build_dpo_pair, select_eq5_pair, selection_record
from src.acccollab.prompts import (
    ACCCOLLAB_PROMPT_VERSION,
    build_actor_deliberation_prompt,
    build_critic_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "acccollab"


def _sample() -> dict[str, object]:
    return {
        "sample_id": "mmlu-0",
        "task_type": "multiple_choice",
        "question": "Which option is correct?",
        "choices": ["alpha", "beta", "gamma", "delta"],
        "choice_labels": ["A", "B", "C", "D"],
        "answer": "B",
    }


@pytest.mark.parametrize(
    ("name", "iterations", "preference_samples", "eval_samples", "rollouts", "trials"),
    [
        ("llama3_8b_instruct_mmlu_original.yaml", 1, 6000, None, 10, 5),
        ("llama3_8b_instruct_mmlu_plus.yaml", 2, 6000, None, 10, 5),
        ("llama3_8b_instruct_mmlu_original_smoke.yaml", 1, 4, 8, 2, 1),
    ],
)
def test_paper_configs_load_with_fixed_protocol(
    name: str,
    iterations: int,
    preference_samples: int,
    eval_samples: int | None,
    rollouts: int,
    trials: int,
) -> None:
    config = load_acccollab_config(str(CONFIG_DIR / name))

    assert config.method.alternating_iterations == iterations
    assert config.method.deliberation_rounds == 5
    assert config.method.training_order == "critic_then_actor"
    assert config.method.reward_estimator == "one_step_mc"
    assert config.method.pair_rule == "paper_eq5_if_elif"
    assert config.training.lora.r == 256
    assert config.training.dpo.nll_weight == 1.0
    assert config.data.mmlu_load_mode == "all"
    assert config.data.preference.samples == preference_samples
    assert config.data.eval.samples == eval_samples
    assert config.reward.rollouts == rollouts
    assert config.evaluation.trials == trials


def test_validation1531_config_uses_full_mmlu_validation_split() -> None:
    config = load_acccollab_config(
        str(CONFIG_DIR / "llama3_8b_instruct_mmlu_validation1531.yaml")
    )

    assert config.data.dataset == "mmlu"
    assert config.data.mmlu_load_mode == "all"
    assert config.data.preference.split == "validation"
    assert config.data.preference.samples is None
    assert config.data.preference.strategy == "full"
    assert config.data.preference.expected_samples == 1531
    assert config.data.preference_trials == 1


def test_preference_trials_expand_trajectories_without_changing_source_set() -> None:
    samples = [
        {"sample_id": "mmlu-0", "acccollab_sample_index": 0, "answer": "A"},
        {"sample_id": "mmlu-1", "acccollab_sample_index": 1, "answer": "B"},
    ]

    expanded = expand_preference_trials(samples, trials=5)

    assert len(expanded) == 10
    assert {row["source_sample_id"] for row in expanded} == {"mmlu-0", "mmlu-1"}
    assert len({row["sample_id"] for row in expanded}) == 10
    assert [row["acccollab_sample_index"] for row in expanded] == list(range(10))
    assert {row["acccollab_preference_trial"] for row in expanded} == set(range(5))


def test_config_rejects_non_positive_preference_trials() -> None:
    source = CONFIG_DIR / "llama3_8b_instruct_mmlu_original_smoke.yaml"
    with pytest.raises(ACCCollabConfigError, match="preference_trials"):
        load_acccollab_config(str(source), ["data.preference_trials=0"])


def test_config_rejects_unknown_and_missing_required_keys(tmp_path: Path) -> None:
    source = CONFIG_DIR / "llama3_8b_instruct_mmlu_original_smoke.yaml"
    with pytest.raises(ACCCollabConfigError, match="Unknown config key"):
        load_acccollab_config(str(source), ["method.unpublished_shortcut=true"])

    payload = OmegaConf.load(source)
    del payload.run.name
    missing = tmp_path / "missing.yaml"
    OmegaConf.save(config=payload, f=str(missing))
    with pytest.raises(ACCCollabConfigError, match=r"config\.run\.name"):
        load_acccollab_config(str(missing))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("method.deliberation_rounds=4", "deliberation_rounds must be 5"),
        ("method.alternating_iterations=3", "must be 1 .* or 2"),
        ("method.training_order=actor_then_critic", "training_order"),
        ("method.reward_estimator=candidate_correctness", "reward_estimator"),
        ("method.pair_rule=two_independent_if", "pair_rule"),
        ("training.lora.r=128", "must be 256"),
        ("training.dpo.nll_weight=0", "must be 1.0"),
    ],
)
def test_config_rejects_protocol_drift(override: str, message: str) -> None:
    source = CONFIG_DIR / "llama3_8b_instruct_mmlu_original_smoke.yaml"
    with pytest.raises(ACCCollabConfigError, match=message):
        load_acccollab_config(str(source), [override])


def test_plus_config_derives_continuation_paths() -> None:
    config = load_acccollab_config(
        str(CONFIG_DIR / "llama3_8b_instruct_mmlu_plus.yaml")
    )

    assert config.actor_adapter_before(1) is None
    assert config.critic_adapter_before(1) is None
    assert config.actor_adapter_before(2) == config.paths.adapter_dir(1, "actor")
    assert config.critic_adapter_before(2) == config.paths.adapter_dir(1, "critic")
    assert config.paths.adapter_dir(2, "actor") != config.paths.adapter_dir(1, "actor")


def test_actor_prompt_preserves_person_order_and_guidance_targets() -> None:
    sample = _sample()
    natural = build_actor_deliberation_prompt(
        sample,
        "mmlu",
        "previous actor answer",
        "critic feedback",
    )
    positive = build_actor_deliberation_prompt(
        sample,
        "mmlu",
        "previous actor answer",
        "critic feedback",
        target_answer="B",
    )
    negative = build_actor_deliberation_prompt(
        sample,
        "mmlu",
        "previous actor answer",
        "critic feedback",
        target_answer="A",
    )

    person_zero = natural.index("Person 0 said: previous actor answer")
    person_one = natural.index("Person 1 said: critic feedback")
    assert person_zero < person_one
    assert "Final Answer: B" in positive
    assert "Final Answer: A" in negative


def test_boolq_guided_actor_prompt_requires_the_target_as_final_answer() -> None:
    sample = {
        "sample_id": "boolq-0",
        "task_type": "yes_no",
        "question": "Is the claim supported?",
        "passage": "A short supporting passage.",
        "answer": "No",
    }
    prompt = build_actor_deliberation_prompt(
        sample,
        "boolq",
        "previous actor answer",
        "critic feedback",
        target_answer="No",
    )

    assert "answering the following question with No" in prompt
    assert "must state that your final answer is No" in prompt


def test_critic_guidance_and_wrong_answer_are_auditable() -> None:
    sample = _sample()
    gold = guidance_answer(sample)
    wrong = wrong_guidance_answer(sample)

    assert gold == "B"
    assert wrong != gold
    assert f"correct answer is {gold}" in build_critic_prompt(
        sample,
        "mmlu",
        "actor response",
        target_answer=gold,
    )
    assert f"correct answer is {wrong}" in build_critic_prompt(
        sample,
        "mmlu",
        "actor response",
        target_answer=wrong,
    )


def test_wrong_guidance_is_seeded_reproducible_and_varied() -> None:
    sample = _sample()
    first = wrong_guidance_answer(sample, seed=17)
    repeated = wrong_guidance_answer(sample, seed=17)
    targets = {wrong_guidance_answer(sample, seed=seed) for seed in range(32)}

    assert first == repeated
    assert first != guidance_answer(sample)
    assert guidance_answer(sample) not in targets
    assert len(targets) > 1

    yes_no = {
        "sample_id": "boolq-1",
        "task_type": "yes_no",
        "question": "A question",
        "passage": "A passage",
        "answer": "Yes",
    }
    assert wrong_guidance_answer(yes_no, seed=1) == "No"


def test_acccollab_implementation_is_isolated_from_paired_pipeline() -> None:
    source_paths = list((ROOT / "src" / "acccollab").glob("*.py"))
    script_paths = list((ROOT / "scripts" / "acccollab").glob("*.py"))

    offenders = [
        str(path.relative_to(ROOT))
        for path in source_paths + script_paths
        if "src.paired" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_eq5_guided_positive_branch_and_priority() -> None:
    ordinary = select_eq5_pair(
        natural_reward=0.2,
        guided_positive_reward=0.9,
        guided_negative_reward=0.1,
        epsilon=0.6,
    )
    both = select_eq5_pair(
        natural_reward=0.6,
        guided_positive_reward=1.0,
        guided_negative_reward=0.0,
        epsilon=0.4,
    )

    assert ordinary is not None
    assert ordinary.case_type == "guided_positive_over_natural"
    assert ordinary.chosen_key == "guided_positive"
    assert ordinary.rejected_key == "natural"
    assert both is not None
    assert both.case_type == "guided_positive_over_natural"


def test_eq5_natural_over_negative_and_drop_branches() -> None:
    negative = select_eq5_pair(
        natural_reward=0.8,
        guided_positive_reward=0.9,
        guided_negative_reward=0.1,
        epsilon=0.6,
    )
    dropped = selection_record(
        natural_reward=0.5,
        guided_positive_reward=0.6,
        guided_negative_reward=0.4,
        epsilon=0.2,
    )

    assert negative is not None
    assert negative.case_type == "natural_over_guided_negative"
    assert negative.chosen_key == "natural"
    assert negative.rejected_key == "guided_negative"
    assert dropped["selected"] is False
    assert dropped["drop_reason"] == "neither_eq5_delta_reached_epsilon"


def test_dpo_pair_uses_one_natural_prompt_and_records_protocol_metadata() -> None:
    decision = select_eq5_pair(
        natural_reward=0.1,
        guided_positive_reward=0.9,
        guided_negative_reward=0.0,
        epsilon=0.6,
    )
    assert decision is not None
    pair = build_dpo_pair(
        natural_prompt="the natural prompt",
        candidate_records={
            "natural": {"response": "natural completion"},
            "guided_positive": {"response": "positive completion"},
            "guided_negative": {"response": "negative completion"},
        },
        decision=decision,
        agent="critic",
        iteration=2,
        sample_id="sample-7",
        sample_index=7,
        round_index=3,
        rollouts=10,
        prompt_version=ACCCOLLAB_PROMPT_VERSION,
    )

    assert pair["prompt"] == "the natural prompt"
    assert pair["chosen"] == "positive completion"
    assert pair["rejected"] == "natural completion"
    metadata = pair["metadata"]
    assert metadata["pipeline"] == "acccollab_original"
    assert metadata["agent"] == "critic"
    assert metadata["iteration"] == 2
    assert metadata["round"] == 3
    assert metadata["reward_rollouts"] == 10
    assert metadata["pair_rule"] == "paper_eq5_if_elif"
    assert metadata["dpo_prompt_source"] == "natural_prompt"
    assert metadata["prompt_version"] == ACCCOLLAB_PROMPT_VERSION


def test_dpo_pair_rejects_identical_completions_after_stripping() -> None:
    decision = select_eq5_pair(
        natural_reward=0.1,
        guided_positive_reward=0.9,
        guided_negative_reward=0.0,
        epsilon=0.6,
    )
    assert decision is not None

    with pytest.raises(ValueError, match="must be different"):
        build_dpo_pair(
            natural_prompt="the natural prompt",
            candidate_records={
                "natural": {"response": "same completion"},
                "guided_positive": {"response": "  same completion\n"},
                "guided_negative": {"response": "different completion"},
            },
            decision=decision,
            agent="actor",
            iteration=1,
            sample_id="sample-0",
            sample_index=0,
            round_index=1,
            rollouts=10,
            prompt_version=ACCCOLLAB_PROMPT_VERSION,
        )
