"""Judge-free final-round majority aggregation for independent ACC-Collab roles."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

from src.acccollab.config import load_acccollab_config
from src.acccollab.io import iter_jsonl, read_json, write_json, write_jsonl
from src.acccollab.registry import evaluation_state
from src.acccollab.stages import evaluation_stage_fingerprint, validate_stage_success
from src.evaluation.answer_resolution import answers_match, normalize_task_answer
from src.multi_acccollab.config import MultiACCCollabConfig, config_snapshot
from src.utils.artifacts import file_sha256, stable_fingerprint


def resolve_majority(
    votes: Mapping[str, str | None],
    *,
    task_type: str,
    fallback_role: str = "direct",
) -> dict[str, Any]:
    """Resolve a three-role vote with a deterministic Direct fallback."""
    if fallback_role not in votes:
        raise ValueError(f"Fallback role {fallback_role!r} is missing from votes")
    normalized = {
        role: normalize_task_answer(answer, task_type)
        for role, answer in votes.items()
    }
    counts = Counter(answer for answer in normalized.values() if answer is not None)
    winners = [answer for answer, count in counts.items() if count >= 2]
    if len(winners) > 1:
        raise RuntimeError(f"Three-role vote produced multiple majority answers: {counts}")
    if winners:
        answer = winners[0]
        count = counts[answer]
        source = "unanimous" if count == len(votes) else "majority"
        selected_role = None
    else:
        answer = normalized[fallback_role]
        count = counts.get(answer, 0) if answer is not None else 0
        source = "fixed_direct_fallback"
        selected_role = fallback_role
    return {
        "answer": answer,
        "source": source,
        "selected_role": selected_role,
        "majority_reached": bool(winners),
        "vote_count": count,
        "normalized_votes": normalized,
    }


def aggregate_role_evaluations(
    config: MultiACCCollabConfig,
    *,
    validate_inputs: bool = True,
) -> Path:
    """Stream aligned role records, write decisions, and score all five trials."""
    role_paths = {
        role.name: config.paths.role_output(role.name) / "eval" / "records.jsonl"
        for role in config.roles
    }
    missing = [str(path) for path in role_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Role evaluation records are missing: {missing}")
    if validate_inputs:
        _validate_role_evaluation_artifacts(config, role_paths)
    fingerprint = stable_fingerprint(
        {
            "pipeline": "multi_acccollab",
            "config": config_snapshot(config),
            "decision_rule": config.evaluation.decision_rule,
            "uses_judge": False,
            "role_records": {
                role: {"path": str(path), "sha256": file_sha256(path)}
                for role, path in role_paths.items()
            },
        }
    )
    output_dir = config.paths.majority_eval_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    success_path = output_dir / "_SUCCESS"
    if _success_matches(success_path, fingerprint):
        return output_dir

    role_names = [role.name for role in config.roles]
    iterators = [iter_jsonl(role_paths[name]) for name in role_names]
    trial_counts: dict[int, dict[str, Any]] = {}

    def decisions():
        for row_index, aligned in enumerate(zip_longest(*iterators), start=1):
            if any(record is None for record in aligned):
                raise RuntimeError("Role evaluation record files have different lengths")
            records = {name: dict(record) for name, record in zip(role_names, aligned)}
            for record in records.values():
                _validate_source_record(record)
            keys = {_record_key(record) for record in records.values()}
            if len(keys) != 1:
                raise RuntimeError(
                    f"Role evaluation alignment mismatch at row {row_index}: {sorted(keys)}"
                )
            trial, sample_index, sample_id = next(iter(keys))
            signatures = {_sample_signature(record) for record in records.values()}
            if len(signatures) != 1:
                raise RuntimeError(
                    f"Role sample/label mismatch at trial={trial}, sample={sample_id}"
                )
            direct_sample = dict(records[config.evaluation.fallback_role].get("sample") or {})
            task_type = str(direct_sample.get("task_type") or "multiple_choice")
            gold = direct_sample.get("answer")
            completions = {
                role: _final_actor_completion(record) for role, record in records.items()
            }
            votes = {role: completion.get("answer") for role, completion in completions.items()}
            decision = resolve_majority(
                votes,
                task_type=task_type,
                fallback_role=config.evaluation.fallback_role,
            )
            correct = answers_match(decision["answer"], gold, task_type)
            counts = trial_counts.setdefault(trial, _empty_trial_counts(role_names))
            _update_counts(counts, completions, decision, correct)
            yield {
                "schema_version": 1,
                "pipeline": "multi_acccollab",
                "implementation_version": config.implementation_version,
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
                        "truncated": bool(completions[role].get("truncated")),
                    }
                    for role in role_names
                },
                "decision_rule": config.evaluation.decision_rule,
                "uses_judge_fallback": False,
                "decision": {
                    key: value for key, value in decision.items() if key != "normalized_votes"
                },
                "correct": correct,
            }

    records_path = output_dir / "records.jsonl"
    write_jsonl(records_path, decisions())
    expected_trials = config.base_config().evaluation.trials
    if sorted(trial_counts) != list(range(expected_trials)):
        raise RuntimeError(
            f"Expected trials 0..{expected_trials - 1}, found {sorted(trial_counts)}"
        )
    trial_metrics = [
        _finalize_trial(trial, trial_counts[trial], role_names)
        for trial in range(expected_trials)
    ]
    expected_samples = config.base_config().data.eval.expected_samples
    if validate_inputs and expected_samples is not None:
        mismatched = [
            (int(metric["trial"]), int(metric["samples"]))
            for metric in trial_metrics
            if int(metric["samples"]) != int(expected_samples)
        ]
        if mismatched:
            raise RuntimeError(
                f"Majority evaluation coverage must be {expected_samples} per trial; "
                f"found {mismatched}"
            )
    for trial, metrics in enumerate(trial_metrics):
        write_json(output_dir / f"trial_{trial:02d}_metrics.json", metrics)
    aggregate = _aggregate_trials(trial_metrics, role_names)
    metrics = {
        "schema_version": 1,
        "pipeline": "multi_acccollab",
        "implementation_version": config.implementation_version,
        "fingerprint": fingerprint,
        "decision_rule": config.evaluation.decision_rule,
        "uses_majority_vote": True,
        "uses_judge_fallback": False,
        "fallback_role": config.evaluation.fallback_role,
        "roles": role_names,
        "trials": expected_trials,
        "headline_metric": {
            "name": "final_round_majority_then_fixed_direct_accuracy",
            "value": aggregate["final_accuracy"],
        },
        "aggregate": aggregate,
        "trial_metrics": trial_metrics,
        "provenance": {
            "role_records": {
                role: {"path": str(path), "sha256": file_sha256(path)}
                for role, path in role_paths.items()
            },
            "role_manifest": str(config.paths.role_manifest),
        },
    }
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    write_json(
        success_path,
        {
            "schema_version": 1,
            "pipeline": "multi_acccollab",
            "status": "complete",
            "stage": "aggregate_majority",
            "fingerprint": fingerprint,
            "artifacts": {
                "records": {"path": str(records_path), "sha256": file_sha256(records_path)},
                "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
            },
        },
    )
    return output_dir


def _record_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    sample = dict(record.get("sample") or {})
    key = (
        int(record.get("trial", -1)),
        int(sample.get("acccollab_sample_index", -1)),
        str(record.get("sample_id") or ""),
    )
    if key[0] < 0 or key[1] < 0 or not key[2]:
        raise ValueError(f"Malformed role evaluation record key: {key}")
    return key


def _final_actor_completion(record: Mapping[str, Any]) -> dict[str, Any]:
    rounds = list(record.get("rounds") or [])
    if [int(item.get("round", -1)) for item in rounds] != list(range(5)):
        raise ValueError("Every role evaluation record must contain rounds 0..4")
    return dict(dict(rounds[-1].get("actor") or {}).get("completion") or {})


def _validate_source_record(record: Mapping[str, Any]) -> None:
    if record.get("pipeline") != "acccollab_original":
        raise ValueError("Majority input must come from acccollab_original evaluation")
    if record.get("decision_rule") != "single_final_actor_answer":
        raise ValueError("Majority input must contain single-Actor source decisions")
    if record.get("uses_majority_vote") is not False:
        raise ValueError("Source role evaluation must not already use majority voting")
    if record.get("uses_judge_fallback") is not False:
        raise ValueError("Source role evaluation must not use Judge fallback")


def _sample_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    sample = dict(record.get("sample") or {})
    return (
        str(sample.get("task_type") or ""),
        str(sample.get("question") or ""),
        tuple(str(choice) for choice in list(sample.get("choices") or [])),
        str(sample.get("passage") or ""),
        str(sample.get("answer") or ""),
    )


def _validate_role_evaluation_artifacts(
    config: MultiACCCollabConfig,
    role_paths: Mapping[str, Path],
) -> None:
    try:
        manifest = read_json(config.paths.role_manifest)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Missing role manifest: {config.paths.role_manifest}") from exc
    if (
        manifest.get("pipeline") != "multi_acccollab"
        or manifest.get("implementation_version") != config.implementation_version
        or manifest.get("config_fingerprint")
        != stable_fingerprint(config_snapshot(config))
    ):
        raise RuntimeError("Role manifest is stale for the current multi-role config")
    manifest_roles = manifest.get("roles")
    if not isinstance(manifest_roles, Mapping):
        raise RuntimeError("Role manifest has no role mapping")

    for role in config.roles:
        info_any = manifest_roles.get(role.name)
        if not isinstance(info_any, Mapping):
            raise RuntimeError(f"Role manifest is missing {role.name}")
        config_path = config.paths.role_config(role.name)
        if (
            not config_path.is_file()
            or info_any.get("config_sha256") != file_sha256(config_path)
        ):
            raise RuntimeError(f"Resolved role config is stale or corrupted: {config_path}")
        role_config = load_acccollab_config(str(config_path))
        marker_path = role_config.paths.eval_dir / "_SUCCESS"
        try:
            raw_marker = read_json(marker_path)
            num_shards = int(dict(raw_marker.get("metadata") or {}).get("num_shards"))
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Malformed role evaluation marker: {marker_path}") from exc
        state = evaluation_state(role_config)
        fingerprint = evaluation_stage_fingerprint(
            role_config,
            state=state,
            num_shards=num_shards,
        )
        marker = validate_stage_success(
            marker_path,
            expected_stage="evaluate",
            expected_fingerprint=fingerprint,
        )
        if marker is None:
            raise RuntimeError(f"Role evaluation is stale or corrupted: {role.name}")
        records_any = dict(marker.get("artifacts") or {}).get("records")
        if not isinstance(records_any, Mapping):
            raise RuntimeError(f"Role evaluation marker has no records: {role.name}")
        marked_path = Path(str(records_any.get("path") or "")).resolve(strict=False)
        if marked_path != role_paths[role.name].resolve(strict=False):
            raise RuntimeError(f"Role evaluation marker redirects records: {role.name}")


def _empty_trial_counts(role_names: Sequence[str]) -> dict[str, Any]:
    return {
        "samples": 0,
        "correct": 0,
        "parsed": 0,
        "majority": 0,
        "majority_correct": 0,
        "fallback": 0,
        "fallback_correct": 0,
        "unanimous": 0,
        "oracle_any_correct": 0,
        "roles": {
            role: {"correct": 0, "parsed": 0, "truncated": 0}
            for role in role_names
        },
    }


def _update_counts(
    counts: dict[str, Any],
    completions: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    correct: bool,
) -> None:
    counts["samples"] += 1
    counts["correct"] += int(correct)
    counts["parsed"] += int(decision.get("answer") is not None)
    source = str(decision["source"])
    if decision["majority_reached"]:
        counts["majority"] += 1
        counts["majority_correct"] += int(correct)
    else:
        counts["fallback"] += 1
        counts["fallback_correct"] += int(correct)
    counts["unanimous"] += int(source == "unanimous")
    counts["oracle_any_correct"] += int(
        any(bool(completion.get("correct")) for completion in completions.values())
    )
    for role, completion in completions.items():
        role_counts = counts["roles"][role]
        role_counts["correct"] += int(bool(completion.get("correct")))
        role_counts["parsed"] += int(bool(completion.get("parsed")))
        role_counts["truncated"] += int(bool(completion.get("truncated")))


def _finalize_trial(
    trial: int,
    counts: Mapping[str, Any],
    role_names: Sequence[str],
) -> dict[str, Any]:
    total = int(counts["samples"])
    majority = int(counts["majority"])
    fallback = int(counts["fallback"])
    return {
        "schema_version": 1,
        "pipeline": "multi_acccollab",
        "trial": trial,
        "samples": total,
        "decision_rule": "final_round_majority_then_fixed_direct",
        "uses_judge_fallback": False,
        "accuracy": _ratio(int(counts["correct"]), total),
        "parse_rate": _ratio(int(counts["parsed"]), total),
        "majority_coverage": _ratio(majority, total),
        "majority_conditional_accuracy": _ratio(int(counts["majority_correct"]), majority),
        "fallback_rate": _ratio(fallback, total),
        "fallback_accuracy": _ratio(int(counts["fallback_correct"]), fallback),
        "unanimous_rate": _ratio(int(counts["unanimous"]), total),
        "oracle_any_actor_accuracy": _ratio(int(counts["oracle_any_correct"]), total),
        "per_role": {
            role: {
                "accuracy": _ratio(int(counts["roles"][role]["correct"]), total),
                "parse_rate": _ratio(int(counts["roles"][role]["parsed"]), total),
                "truncation_rate": _ratio(
                    int(counts["roles"][role]["truncated"]), total
                ),
            }
            for role in role_names
        },
    }


def _aggregate_trials(
    trials: Sequence[Mapping[str, Any]],
    role_names: Sequence[str],
) -> dict[str, Any]:
    samples = {int(trial["samples"]) for trial in trials}
    if len(samples) != 1:
        raise RuntimeError("All majority evaluation trials must cover the same samples")
    return {
        "samples_per_trial": next(iter(samples)),
        "final_accuracy": _summary([float(trial["accuracy"]) for trial in trials]),
        "parse_rate": _summary([float(trial["parse_rate"]) for trial in trials]),
        "majority_coverage": _summary(
            [float(trial["majority_coverage"]) for trial in trials]
        ),
        "majority_conditional_accuracy": _summary(
            [float(trial["majority_conditional_accuracy"]) for trial in trials]
        ),
        "fallback_accuracy": _summary(
            [float(trial["fallback_accuracy"]) for trial in trials]
        ),
        "oracle_any_actor_accuracy": _summary(
            [float(trial["oracle_any_actor_accuracy"]) for trial in trials]
        ),
        "per_role_accuracy": {
            role: _summary(
                [float(dict(trial["per_role"])[role]["accuracy"]) for trial in trials]
            )
            for role in role_names
        },
        "trial_values": [
            {
                "trial": int(trial["trial"]),
                "accuracy": float(trial["accuracy"]),
                "majority_coverage": float(trial["majority_coverage"]),
                "fallback_accuracy": float(trial["fallback_accuracy"]),
            }
            for trial in trials
        ],
    }


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = std / math.sqrt(len(values)) if values else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "sample_std": std,
        "standard_error": standard_error,
        "ci95_normal_half_width": 1.96 * standard_error,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _success_matches(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        from src.acccollab.io import read_json

        payload = read_json(path)
    except (OSError, ValueError):
        return False
    if not (
        payload.get("pipeline") == "multi_acccollab"
        and payload.get("status") == "complete"
        and payload.get("fingerprint") == fingerprint
    ):
        return False
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        return False
    for artifact_any in artifacts.values():
        if not isinstance(artifact_any, Mapping):
            return False
        artifact = Path(str(artifact_any.get("path") or ""))
        if not artifact.is_file() or artifact_any.get("sha256") != file_sha256(artifact):
            return False
    return True
