from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from src.acccollab.evaluation import (
    aggregate_trial_metrics,
    compare_with_paper,
    score_evaluation_records,
)
from src.acccollab.io import merge_sorted_jsonl, read_jsonl, write_jsonl
from src.acccollab.prompts import ACCCOLLAB_PROMPT_VERSION


def _evaluation_record(
    sample_id: str,
    correct_by_round: list[bool],
) -> dict[str, Any]:
    assert len(correct_by_round) == 5
    return {
        "pipeline": "acccollab_original",
        "prompt_version": ACCCOLLAB_PROMPT_VERSION,
        "decision_rule": "single_final_actor_answer",
        "uses_majority_vote": False,
        "uses_judge_fallback": False,
        "sample_id": sample_id,
        "rounds": [
            {
                "round": round_index,
                "actor": {
                    "completion": {
                        "correct": correct,
                        "parsed": True,
                        "truncated": False,
                    }
                },
            }
            for round_index, correct in enumerate(correct_by_round)
        ],
    }


def test_headline_uses_round_four_single_actor_not_earlier_rounds() -> None:
    metrics = score_evaluation_records(
        [_evaluation_record("s0", [True, True, True, True, False])],
        expected_sample_ids=["s0"],
    )

    assert metrics["per_round"][0]["accuracy"] == 1.0
    assert metrics["per_round"][4]["accuracy"] == 0.0
    assert metrics["final"]["accuracy"] == 0.0
    assert metrics["final"]["round"] == 4
    assert metrics["final"]["source"] == "round_4_single_actor"
    assert metrics["uses_majority_vote"] is False
    assert metrics["uses_judge_fallback"] is False


def test_round_four_correct_answer_defines_correct_headline() -> None:
    metrics = score_evaluation_records(
        [_evaluation_record("s0", [False, False, False, False, True])]
    )
    assert metrics["final"]["accuracy"] == 1.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pipeline", "paired_actor_critic", "does not belong"),
        ("prompt_version", "stale", "prompt_version"),
        ("decision_rule", "majority_vote", "single final Actor"),
        ("uses_majority_vote", True, "majority vote"),
        ("uses_judge_fallback", True, "Judge fallback"),
    ],
)
def test_evaluation_rejects_protocol_drift(field: str, value: Any, message: str) -> None:
    record = _evaluation_record("s0", [True] * 5)
    record[field] = value
    with pytest.raises(ValueError, match=message):
        score_evaluation_records([record])


def test_evaluation_rejects_round_sample_and_coverage_errors() -> None:
    record = _evaluation_record("s0", [True] * 5)
    missing_round = copy.deepcopy(record)
    missing_round["rounds"].pop(2)
    with pytest.raises(ValueError, match="rounds 0..4 exactly"):
        score_evaluation_records([missing_round])

    with pytest.raises(ValueError, match="Duplicate evaluation sample_id"):
        score_evaluation_records([record, copy.deepcopy(record)])

    with pytest.raises(ValueError, match="coverage mismatch"):
        score_evaluation_records([record], expected_sample_ids=["s0", "s1"])


def test_trial_aggregation_requires_round_four_protocol_and_equal_coverage() -> None:
    first = score_evaluation_records([_evaluation_record("s0", [True] * 5)])
    second = score_evaluation_records([_evaluation_record("s1", [False] * 5)])
    aggregate = aggregate_trial_metrics([first, second])

    assert aggregate["trials"] == 2
    assert aggregate["final_accuracy"]["mean"] == 0.5
    assert aggregate["decision_rule"] == "single_final_actor_answer"

    stale = copy.deepcopy(second)
    stale["final"]["source"] = "majority_vote"
    with pytest.raises(ValueError, match="round 4"):
        aggregate_trial_metrics([first, stale])

    unequal = score_evaluation_records(
        [
            _evaluation_record("s1", [False] * 5),
            _evaluation_record("s2", [False] * 5),
        ]
    )
    with pytest.raises(ValueError, match="same non-negative sample count"):
        aggregate_trial_metrics([first, unequal])


def test_paper_comparison_uses_reported_ci95_half_widths() -> None:
    original = compare_with_paper(
        {"final_accuracy": {"mean": 0.660}},
        dataset_name="mmlu",
        model_type="llama3",
        alternating_iterations=1,
    )
    plus = compare_with_paper(
        {"final_accuracy": {"mean": 0.690}},
        dataset_name="mmlu",
        model_type="llama3",
        alternating_iterations=2,
    )

    assert original is not None
    assert plus is not None
    assert original["method"] == "ACC-Collab"
    assert original["paper"]["accuracy"] == 0.644
    assert original["paper"]["ci95_low"] == pytest.approx(0.634)
    assert original["paper"]["ci95_high"] == pytest.approx(0.654)
    assert original["absolute_delta"] == pytest.approx(0.016)
    assert original["above_paper_mean"] is True
    assert original["within_paper_reported_ci95"] is False
    assert "not a standard deviation" in original["paper"]["note"]

    assert plus["method"] == "ACC-Collab+"
    assert plus["paper"]["accuracy"] == 0.683
    assert plus["paper"]["ci95_low"] == pytest.approx(0.671)
    assert plus["paper"]["ci95_high"] == pytest.approx(0.695)
    assert plus["above_paper_mean"] is True
    assert plus["within_paper_reported_ci95"] is True


@pytest.mark.parametrize(
    ("dataset_name", "iterations", "accuracy", "half_width"),
    [
        ("mmlu", 1, 0.644, 0.010),
        ("sciq", 1, 0.952, 0.000),
        ("boolq", 2, 0.894, 0.003),
    ],
)
def test_paper_comparison_dispatches_to_matching_table_one_row(
    dataset_name: str,
    iterations: int,
    accuracy: float,
    half_width: float,
) -> None:
    result = compare_with_paper(
        {"final_accuracy": {"mean": accuracy}},
        dataset_name=dataset_name,
        model_type="llama3_8b_instruct",
        alternating_iterations=iterations,
    )

    assert result is not None
    assert result["dataset"] == dataset_name
    assert result["paper"]["accuracy"] == accuracy
    assert result["paper"]["reported_ci95_half_width"] == half_width


def test_paper_comparison_returns_none_without_matching_reference() -> None:
    metrics = {"final_accuracy": {"mean": 0.5}}

    assert compare_with_paper(
        metrics,
        dataset_name="unknown",
        model_type="llama3",
        alternating_iterations=1,
    ) is None
    assert compare_with_paper(
        metrics,
        dataset_name="mmlu",
        model_type="qwen",
        alternating_iterations=1,
    ) is None


def test_paper_comparison_supports_all_paper_models_and_new_datasets() -> None:
    aggregate = {"final_accuracy": {"mean": 0.5}}
    mistral = compare_with_paper(
        aggregate,
        dataset_name="bbh",
        model_type="mistral",
        alternating_iterations=1,
    )
    gemma = compare_with_paper(
        aggregate,
        dataset_name="arc",
        model_type="gemma2",
        alternating_iterations=1,
    )
    assert mistral["paper"]["accuracy"] == 0.519
    assert gemma["paper"]["accuracy"] == 0.852


def test_merge_sorted_jsonl_streams_and_stably_orders_equal_keys(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    write_jsonl(
        first,
        [
            {"key": 1, "value": "a0"},
            {"key": 1, "value": "a1"},
            {"key": 3, "value": "a3"},
        ],
    )
    write_jsonl(
        second,
        [
            {"key": 1, "value": "b0"},
            {"key": 2, "value": "b2"},
        ],
    )

    count = merge_sorted_jsonl([first, second], output, key=lambda row: int(row["key"]))

    assert count == 5
    assert [row["value"] for row in read_jsonl(output)] == ["a0", "a1", "b0", "b2", "a3"]


def test_merge_failure_does_not_replace_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "unsorted.jsonl"
    output = tmp_path / "output.jsonl"
    write_jsonl(source, [{"key": 2}, {"key": 1}])
    output.write_text('{"sentinel": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="not sorted"):
        merge_sorted_jsonl([source], output, key=lambda row: int(row["key"]))

    assert output.read_text(encoding="utf-8") == '{"sentinel": true}\n'


def test_write_jsonl_failure_is_atomic(tmp_path: Path) -> None:
    output = tmp_path / "rows.jsonl"
    output.write_text('{"sentinel": true}\n', encoding="utf-8")

    def broken_rows():
        yield {"row": 1}
        raise RuntimeError("generator failed")

    with pytest.raises(RuntimeError, match="generator failed"):
        write_jsonl(output, broken_rows())

    assert output.read_text(encoding="utf-8") == '{"sentinel": true}\n'
