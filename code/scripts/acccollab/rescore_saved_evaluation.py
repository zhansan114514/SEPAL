"""Re-score saved ACC-Collab evaluation records with the current answer parser."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.acccollab.evaluation import (
    aggregate_trial_metrics,
    compare_with_paper,
    score_evaluation_records,
)
from src.acccollab.io import iter_jsonl, read_json, write_json
from src.evaluation.answer_resolution import answers_match, normalize_task_answer
from src.parsing import answer_extractor
from src.parsing.answer_extractor import extract_answer


IMPLEMENTATION_VERSION = "saved_evaluation_answer_parser_rescore_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--model-type", default="gemma2")
    parser.add_argument("--alternating-iterations", type=int, default=1)
    parser.add_argument("--tag", default="current_answer_parser")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    eval_dir = Path(args.eval_dir)
    records_path = eval_dir / "records.jsonl"
    metrics_path = eval_dir / "metrics.json"
    if not records_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"Expected records.jsonl and metrics.json under {eval_dir}"
        )

    original_metrics = read_json(metrics_path)
    trials = int(original_metrics.get("trials") or 0)
    if trials < 1:
        raise ValueError(f"Invalid trial count in {metrics_path}: {trials}")

    first_record = next(iter_jsonl(records_path), None)
    if first_record is None:
        raise ValueError(f"Cannot re-score empty records file: {records_path}")
    dataset_name = str(dict(first_record.get("sample") or {}).get("dataset") or "")
    if not dataset_name:
        raise ValueError(f"Cannot determine dataset from {records_path}")

    diagnostics = _empty_diagnostics()
    trial_metrics = [
        score_evaluation_records(
            _rescored_trial_records(records_path, trial_index, diagnostics),
            deliberation_rounds=5,
        )
        for trial_index in range(trials)
    ]
    aggregate = aggregate_trial_metrics(trial_metrics)
    comparison = compare_with_paper(
        aggregate,
        dataset_name=dataset_name,
        model_type=str(args.model_type),
        alternating_iterations=int(args.alternating_iterations),
    )

    output_dir = eval_dir / "rescored" / str(args.tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    parser_path = Path(answer_extractor.__file__).resolve()
    payload = {
        "schema_version": 1,
        "pipeline": "acccollab_original",
        "stage": "rescore_saved_evaluation",
        "implementation_version": IMPLEMENTATION_VERSION,
        "dataset": dataset_name,
        "model_type": str(args.model_type),
        "decision_rule": "single_final_actor_answer",
        "uses_majority_vote": False,
        "uses_judge_fallback": False,
        "source": {
            "records_path": str(records_path),
            "records_sha256": _sha256(records_path),
            "original_metrics_path": str(metrics_path),
            "original_metrics_sha256": _sha256(metrics_path),
        },
        "parser": {
            "path": str(parser_path),
            "sha256": _sha256(parser_path),
        },
        "original": {
            "final_accuracy": _aggregate_mean(original_metrics, "final_accuracy"),
            "final_parse_rate": _aggregate_mean(
                original_metrics,
                "final_parse_rate",
            ),
        },
        "rescored": {
            "trial_metrics": trial_metrics,
            "aggregate": aggregate,
            "paper_comparison": comparison,
        },
        "diagnostics": diagnostics,
    }
    output_path = output_dir / "metrics.json"
    write_json(output_path, payload)
    write_json(
        output_dir / "_SUCCESS",
        {
            "schema_version": 1,
            "stage": "rescore_saved_evaluation",
            "implementation_version": IMPLEMENTATION_VERSION,
            "metrics_path": str(output_path),
            "metrics_sha256": _sha256(output_path),
        },
    )
    print(
        f"{dataset_name}: "
        f"{payload['original']['final_accuracy']:.6f} -> "
        f"{aggregate['final_accuracy']['mean']:.6f}; "
        f"parse {payload['original']['final_parse_rate']:.6f} -> "
        f"{aggregate['final_parse_rate']['mean']:.6f}"
    )


def _rescored_trial_records(
    records_path: Path,
    trial_index: int,
    diagnostics: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    for record in iter_jsonl(records_path):
        if int(record.get("trial", -1)) != trial_index:
            continue
        yield _rescore_record(record, diagnostics)


def _rescore_record(
    record: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    sample = dict(record.get("sample") or {})
    task_type = str(sample.get("task_type") or "multiple_choice")
    label = sample.get("answer")
    diagnostics["records"] += 1
    for round_index, round_record in enumerate(list(record.get("rounds") or [])):
        actor = dict(round_record.get("actor") or {})
        completion = dict(actor.get("completion") or {})
        response = str(completion.get("response") or completion.get("raw_response") or "")
        extracted = extract_answer(response, task_type)
        normalized = normalize_task_answer(extracted.answer, task_type)
        answer = normalized if normalized is not None else extracted.answer
        parsed = extracted.answer is not None
        correct = answers_match(extracted.answer, label, task_type)

        old_answer = completion.get("answer")
        old_parsed = bool(completion.get("parsed"))
        old_correct = bool(completion.get("correct"))
        per_round = diagnostics["per_round"][round_index]
        per_round["completions"] += 1
        per_round["changed_answer"] += int(old_answer != answer)
        per_round["recovered_parse"] += int(not old_parsed and parsed)
        per_round["lost_parse"] += int(old_parsed and not parsed)
        per_round["changed_correct"] += int(old_correct != correct)

        completion.update(
            answer=answer,
            answer_source=extracted.source,
            parse_confidence=extracted.confidence,
            parsed=parsed,
            correct=correct,
        )
        actor["completion"] = completion
        round_record["actor"] = actor
    return record


def _empty_diagnostics() -> dict[str, Any]:
    return {
        "records": 0,
        "per_round": [
            {
                "round": round_index,
                "completions": 0,
                "changed_answer": 0,
                "recovered_parse": 0,
                "lost_parse": 0,
                "changed_correct": 0,
            }
            for round_index in range(5)
        ],
    }


def _aggregate_mean(metrics: Mapping[str, Any], key: str) -> float:
    aggregate = dict(metrics.get("aggregate") or {})
    value = dict(aggregate.get(key) or {})
    return float(value.get("mean") or 0.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
