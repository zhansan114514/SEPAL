"""Re-score saved multi-role evaluation records with the current answer parser.

The generated model responses are never modified.  This command writes a
separate, parser-versioned majority result beneath ``eval_majority/rescored``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, Mapping

from src.acccollab.io import iter_jsonl, read_json, write_json, write_jsonl
from src.evaluation.answer_resolution import answers_match, normalize_task_answer
from src.multi_acccollab.majority import (
    _aggregate_trials,
    _empty_trial_counts,
    _final_actor_completion,
    _finalize_trial,
    _record_key,
    _sample_signature,
    _update_counts,
    _validate_source_record,
    resolve_majority,
)
from src.parsing import answer_extractor
from src.parsing.answer_extractor import extract_answer
from src.utils.artifacts import file_sha256, stable_fingerprint


IMPLEMENTATION_VERSION = "saved_multi_evaluation_answer_parser_rescore_v1"
ROLE_NAMES = ("direct", "evidence", "verification")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Multi-ACC-Collab dataset output containing roles/ and eval_majority/.",
    )
    parser.add_argument("--tag", default="current_answer_parser")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = Path(args.experiment_dir)
    role_paths = {
        role: experiment_dir / "roles" / role / "eval" / "records.jsonl"
        for role in ROLE_NAMES
    }
    missing = [str(path) for path in role_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Role evaluation records are missing: {missing}")

    original_metrics_path = experiment_dir / "eval_majority" / "metrics.json"
    if not original_metrics_path.is_file():
        raise FileNotFoundError(
            f"Original majority metrics are missing: {original_metrics_path}"
        )

    parser_path = Path(answer_extractor.__file__).resolve()
    provenance = {
        "parser": {
            "path": str(parser_path),
            "sha256": file_sha256(parser_path),
        },
        "role_records": {
            role: {"path": str(path), "sha256": file_sha256(path)}
            for role, path in role_paths.items()
        },
        "original_metrics": {
            "path": str(original_metrics_path),
            "sha256": file_sha256(original_metrics_path),
        },
    }
    fingerprint = stable_fingerprint(
        {
            "pipeline": "multi_acccollab",
            "stage": "rescore_saved_evaluation",
            "implementation_version": IMPLEMENTATION_VERSION,
            "provenance": provenance,
        }
    )
    output_dir = experiment_dir / "eval_majority" / "rescored" / str(args.tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = {
        role: {
            "completions": 0,
            "changed_answer": 0,
            "recovered_parse": 0,
            "lost_parse": 0,
            "changed_correct": 0,
            "answer_sources": Counter(),
        }
        for role in ROLE_NAMES
    }
    trial_counts: dict[int, dict[str, Any]] = {}
    iterators = [iter_jsonl(role_paths[role]) for role in ROLE_NAMES]

    def decisions():
        for row_index, aligned in enumerate(zip_longest(*iterators), start=1):
            if any(record is None for record in aligned):
                raise RuntimeError("Role evaluation record files have different lengths")
            records = {
                role: dict(record) for role, record in zip(ROLE_NAMES, aligned)
            }
            for record in records.values():
                _validate_source_record(record)
            keys = {_record_key(record) for record in records.values()}
            if len(keys) != 1:
                raise RuntimeError(
                    f"Role evaluation alignment mismatch at row {row_index}: "
                    f"{sorted(keys)}"
                )
            signatures = {_sample_signature(record) for record in records.values()}
            if len(signatures) != 1:
                raise RuntimeError(
                    f"Role sample/label mismatch at row {row_index}"
                )

            trial, sample_index, sample_id = next(iter(keys))
            sample = dict(records["direct"].get("sample") or {})
            task_type = str(sample.get("task_type") or "multiple_choice")
            gold = sample.get("answer")
            completions = {
                role: _rescore_completion(
                    _final_actor_completion(records[role]),
                    task_type=task_type,
                    gold=gold,
                    diagnostics=diagnostics[role],
                )
                for role in ROLE_NAMES
            }
            votes = {
                role: completion.get("answer")
                for role, completion in completions.items()
            }
            decision = resolve_majority(
                votes,
                task_type=task_type,
                fallback_role="direct",
            )
            correct = answers_match(decision["answer"], gold, task_type)
            counts = trial_counts.setdefault(
                trial,
                _empty_trial_counts(ROLE_NAMES),
            )
            _update_counts(counts, completions, decision, correct)
            yield {
                "schema_version": 1,
                "pipeline": "multi_acccollab",
                "stage": "rescore_saved_evaluation",
                "implementation_version": IMPLEMENTATION_VERSION,
                "trial": trial,
                "sample_index": sample_index,
                "sample_id": sample_id,
                "task_type": task_type,
                "gold_answer": gold,
                "role_votes": {
                    role: {
                        "answer": decision["normalized_votes"][role],
                        "parsed": bool(completions[role].get("parsed")),
                        "correct": bool(completions[role].get("correct")),
                        "answer_source": completions[role].get("answer_source"),
                        "truncated": bool(completions[role].get("truncated")),
                    }
                    for role in ROLE_NAMES
                },
                "decision_rule": "final_round_majority_then_fixed_direct",
                "uses_judge_fallback": False,
                "decision": {
                    key: value
                    for key, value in decision.items()
                    if key != "normalized_votes"
                },
                "correct": correct,
            }

    records_path = output_dir / "records.jsonl"
    write_jsonl(records_path, decisions())
    trials = sorted(trial_counts)
    if trials != list(range(len(trials))):
        raise RuntimeError(f"Expected contiguous trials starting at zero, found {trials}")
    trial_metrics = [
        _finalize_trial(trial, trial_counts[trial], ROLE_NAMES)
        for trial in trials
    ]
    aggregate = _aggregate_trials(trial_metrics, ROLE_NAMES)
    original_metrics = read_json(original_metrics_path)
    original_aggregate = dict(original_metrics.get("aggregate") or {})

    serializable_diagnostics = {
        role: {
            **{
                key: value
                for key, value in values.items()
                if key != "answer_sources"
            },
            "answer_sources": dict(values["answer_sources"]),
        }
        for role, values in diagnostics.items()
    }
    payload = {
        "schema_version": 1,
        "pipeline": "multi_acccollab",
        "stage": "rescore_saved_evaluation",
        "implementation_version": IMPLEMENTATION_VERSION,
        "fingerprint": fingerprint,
        "decision_rule": "final_round_majority_then_fixed_direct",
        "uses_majority_vote": True,
        "uses_judge_fallback": False,
        "fallback_role": "direct",
        "roles": list(ROLE_NAMES),
        "trials": len(trials),
        "headline_metric": {
            "name": "final_round_majority_then_fixed_direct_accuracy",
            "value": aggregate["final_accuracy"],
        },
        "aggregate": aggregate,
        "trial_metrics": trial_metrics,
        "original": {
            "final_accuracy": _mean(original_aggregate.get("final_accuracy")),
            "parse_rate": _mean(original_aggregate.get("parse_rate")),
            "majority_coverage": _mean(
                original_aggregate.get("majority_coverage")
            ),
        },
        "diagnostics": serializable_diagnostics,
        "provenance": provenance,
    }
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, payload)
    success_path = output_dir / "_SUCCESS"
    write_json(
        success_path,
        {
            "schema_version": 1,
            "pipeline": "multi_acccollab",
            "status": "complete",
            "stage": "rescore_saved_evaluation",
            "implementation_version": IMPLEMENTATION_VERSION,
            "fingerprint": fingerprint,
            "artifacts": {
                "records": {
                    "path": str(records_path),
                    "sha256": file_sha256(records_path),
                },
                "metrics": {
                    "path": str(metrics_path),
                    "sha256": file_sha256(metrics_path),
                },
            },
        },
    )
    print(
        f"{experiment_dir.name}: "
        f"accuracy {_mean(original_aggregate.get('final_accuracy')):.6f} -> "
        f"{_mean(aggregate['final_accuracy']):.6f}; "
        f"parse {_mean(original_aggregate.get('parse_rate')):.6f} -> "
        f"{_mean(aggregate['parse_rate']):.6f}"
    )


def _rescore_completion(
    completion: Mapping[str, Any],
    *,
    task_type: str,
    gold: Any,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    rescored = dict(completion)
    response = str(
        rescored.get("response") or rescored.get("raw_response") or ""
    )
    extracted = extract_answer(response, task_type)
    answer = normalize_task_answer(extracted.answer, task_type)
    if answer is None:
        answer = extracted.answer
    parsed = extracted.answer is not None
    correct = answers_match(extracted.answer, gold, task_type)

    diagnostics["completions"] += 1
    diagnostics["changed_answer"] += int(rescored.get("answer") != answer)
    diagnostics["recovered_parse"] += int(not rescored.get("parsed") and parsed)
    diagnostics["lost_parse"] += int(bool(rescored.get("parsed")) and not parsed)
    diagnostics["changed_correct"] += int(
        bool(rescored.get("correct")) != correct
    )
    diagnostics["answer_sources"][extracted.source] += 1
    rescored.update(
        answer=answer,
        answer_source=extracted.source,
        parse_confidence=extracted.confidence,
        parsed=parsed,
        correct=correct,
    )
    return rescored


def _mean(value: Any) -> float:
    if isinstance(value, Mapping):
        return float(value.get("mean") or 0.0)
    return float(value or 0.0)


if __name__ == "__main__":
    main()
