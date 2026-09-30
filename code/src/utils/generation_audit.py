"""Generation-length metadata, aggregation, and truncation policy helpers."""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable, Mapping


class GenerationText(str):
    """A ``str`` carrying vLLM finish metadata without breaking callers."""

    finish_reason: str | None
    generated_tokens: int | None
    max_tokens: int | None
    length_truncated: bool

    def __new__(
        cls,
        value: str,
        *,
        finish_reason: str | None = None,
        generated_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> "GenerationText":
        obj = super().__new__(cls, value)
        obj.finish_reason = finish_reason
        obj.generated_tokens = generated_tokens
        obj.max_tokens = max_tokens
        obj.length_truncated = bool(
            finish_reason == "length"
            or (
                generated_tokens is not None
                and max_tokens is not None
                and generated_tokens >= max_tokens
                and finish_reason not in {"stop", "abort"}
            )
        )
        return obj


def generation_metadata(value: Any) -> dict[str, Any]:
    """Return JSON-safe finish metadata from a generated string."""
    return {
        "finish_reason": getattr(value, "finish_reason", None),
        "generated_tokens": getattr(value, "generated_tokens", None),
        "max_tokens": getattr(value, "max_tokens", None),
        "length_truncated": bool(getattr(value, "length_truncated", False)),
    }


def empty_generation_stats() -> dict[str, Any]:
    return {"schema_version": 1, "by_max_tokens": {}}


def record_generation(
    stats: dict[str, Any],
    *,
    max_tokens: int,
    finish_reason: str | None,
    generated_tokens: int | None,
) -> None:
    """Update an in-memory compact histogram for one generated completion."""
    key = str(int(max_tokens))
    groups = stats.setdefault("by_max_tokens", {})
    group = groups.setdefault(
        key,
        {
            "outputs": 0,
            "length_finished": 0,
            "generated_tokens_total": 0,
            "generated_tokens_observed": 0,
            "token_histogram": {},
        },
    )
    group["outputs"] += 1
    length_finished = finish_reason == "length"
    if generated_tokens is not None and generated_tokens >= int(max_tokens):
        if finish_reason not in {"stop", "abort"}:
            length_finished = True
    group["length_finished"] += int(length_finished)
    if generated_tokens is not None:
        tokens = int(generated_tokens)
        group["generated_tokens_total"] += tokens
        group["generated_tokens_observed"] += 1
        histogram = group.setdefault("token_histogram", {})
        hist_key = str(tokens)
        histogram[hist_key] = int(histogram.get(hist_key, 0)) + 1


def clone_generation_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(stats))


def subtract_generation_stats(
    after: Mapping[str, Any],
    before: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the non-negative per-group delta between two engine snapshots."""
    result = empty_generation_stats()
    after_groups = dict(after.get("by_max_tokens", {}))
    before_groups = dict(before.get("by_max_tokens", {}))
    for key, after_group_any in after_groups.items():
        after_group = dict(after_group_any)
        before_group = dict(before_groups.get(key, {}))
        group = {
            field: max(0, int(after_group.get(field, 0)) - int(before_group.get(field, 0)))
            for field in (
                "outputs",
                "length_finished",
                "generated_tokens_total",
                "generated_tokens_observed",
            )
        }
        histogram: dict[str, int] = {}
        after_hist = dict(after_group.get("token_histogram", {}))
        before_hist = dict(before_group.get("token_histogram", {}))
        for token_count, count in after_hist.items():
            delta = int(count) - int(before_hist.get(token_count, 0))
            if delta > 0:
                histogram[str(token_count)] = delta
        group["token_histogram"] = histogram
        if group["outputs"] > 0:
            result["by_max_tokens"][str(key)] = group
    return result


def merge_generation_stats(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge raw engine snapshots/deltas from checkpoint JSONL rows."""
    merged = empty_generation_stats()
    for row in rows:
        groups = row.get("by_max_tokens", {})
        if not isinstance(groups, Mapping):
            continue
        for key, source_any in groups.items():
            if not isinstance(source_any, Mapping):
                continue
            target = merged["by_max_tokens"].setdefault(
                str(key),
                {
                    "outputs": 0,
                    "length_finished": 0,
                    "generated_tokens_total": 0,
                    "generated_tokens_observed": 0,
                    "token_histogram": {},
                },
            )
            for field in (
                "outputs",
                "length_finished",
                "generated_tokens_total",
                "generated_tokens_observed",
            ):
                target[field] += int(source_any.get(field, 0))
            histogram = source_any.get("token_histogram", {})
            if isinstance(histogram, Mapping):
                for token_count, count in histogram.items():
                    key_count = str(token_count)
                    target["token_histogram"][key_count] = (
                        int(target["token_histogram"].get(key_count, 0)) + int(count)
                    )
    return merged


def _percentile(histogram: Mapping[str, Any], quantile: float) -> int | None:
    pairs = sorted((int(token_count), int(count)) for token_count, count in histogram.items())
    total = sum(count for _token_count, count in pairs)
    if total <= 0:
        return None
    target = max(1, math.ceil(total * quantile))
    seen = 0
    for token_count, count in pairs:
        seen += count
        if seen >= target:
            return token_count
    return pairs[-1][0]


def summarize_generation_stats(
    rows_or_stats: Iterable[Mapping[str, Any]] | Mapping[str, Any],
) -> dict[str, Any]:
    """Build human-readable truncation metrics, including p50/p95/p99 lengths."""
    if isinstance(rows_or_stats, Mapping):
        merged = merge_generation_stats([rows_or_stats])
    else:
        merged = merge_generation_stats(rows_or_stats)

    total_outputs = 0
    total_length_finished = 0
    total_tokens = 0
    total_observed = 0
    overall_histogram: dict[str, int] = {}
    by_limit: dict[str, Any] = {}
    for key in sorted(merged["by_max_tokens"], key=lambda item: int(item)):
        group = merged["by_max_tokens"][key]
        outputs = int(group.get("outputs", 0))
        length_finished = int(group.get("length_finished", 0))
        observed = int(group.get("generated_tokens_observed", 0))
        token_total = int(group.get("generated_tokens_total", 0))
        histogram = dict(group.get("token_histogram", {}))
        total_outputs += outputs
        total_length_finished += length_finished
        total_tokens += token_total
        total_observed += observed
        for token_count, count in histogram.items():
            overall_histogram[str(token_count)] = (
                int(overall_histogram.get(str(token_count), 0)) + int(count)
            )
        by_limit[key] = {
            "outputs": outputs,
            "length_finished": length_finished,
            "length_truncation_rate": length_finished / outputs if outputs else 0.0,
            "generated_tokens_mean": token_total / observed if observed else None,
            "generated_tokens_p50": _percentile(histogram, 0.50),
            "generated_tokens_p95": _percentile(histogram, 0.95),
            "generated_tokens_p99": _percentile(histogram, 0.99),
            "generated_tokens_max": max((int(item) for item in histogram), default=None),
        }

    return {
        "schema_version": 1,
        "outputs": total_outputs,
        "length_finished": total_length_finished,
        "length_truncation_rate": (
            total_length_finished / total_outputs if total_outputs else 0.0
        ),
        "generated_tokens_mean": total_tokens / total_observed if total_observed else None,
        "generated_tokens_p50": _percentile(overall_histogram, 0.50),
        "generated_tokens_p95": _percentile(overall_histogram, 0.95),
        "generated_tokens_p99": _percentile(overall_histogram, 0.99),
        "generated_tokens_max": max((int(item) for item in overall_histogram), default=None),
        "by_max_tokens": by_limit,
    }


def _round_up(value: float, quantum: int = 64) -> int:
    return int(math.ceil(value / quantum) * quantum)


def assess_generation_stats(
    rows_or_stats: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    warn_rate: float,
    fail_rate: float,
    max_model_len: int | None = None,
) -> dict[str, Any]:
    """Summarize generation lengths and attach a pass/warn/fail assessment.

    Recommendations are deliberately conservative: when a limit produces any
    length-finished output, the suggested next smoke-test limit adds 25%
    headroom and rounds to 64 tokens. It is capped below ``max_model_len`` when
    supplied; prompt length still needs to be considered before applying it.
    """
    summary = summarize_generation_stats(rows_or_stats)
    rate = float(summary["length_truncation_rate"])
    if rate >= fail_rate and summary["outputs"]:
        status = "fail"
    elif rate >= warn_rate and summary["outputs"]:
        status = "warn"
    else:
        status = "pass"

    recommendations: dict[str, int] = {}
    for limit_text, group in summary["by_max_tokens"].items():
        limit = int(limit_text)
        recommended = limit
        if int(group["length_finished"]) > 0:
            p99 = int(group.get("generated_tokens_p99") or limit)
            recommended = _round_up(max(limit + 64, limit * 1.25, p99 * 1.25))
            if max_model_len is not None:
                recommended = min(recommended, max(1, int(max_model_len) - 1))
        recommendations[limit_text] = recommended

    summary["assessment"] = {
        "status": status,
        "warn_rate": float(warn_rate),
        "fail_rate": float(fail_rate),
        "max_model_len": int(max_model_len) if max_model_len is not None else None,
        "recommended_max_tokens_by_limit": recommendations,
        "note": (
            "Recommendations require a smoke rerun because available prompt context varies "
            "per request."
        ),
    }
    return summary


def enforce_generation_assessment(summary: Mapping[str, Any], *, fail_on_excess: bool) -> None:
    """Raise after outputs/metrics are written when configured truncation policy fails."""
    assessment = summary.get("assessment", {})
    if fail_on_excess and assessment.get("status") == "fail":
        raise RuntimeError(
            "Generation truncation rate exceeded configured fail threshold: "
            f"rate={summary.get('length_truncation_rate')}, "
            f"fail_rate={assessment.get('fail_rate')}, "
            "see generation_audit metrics for recommended max_tokens."
        )
