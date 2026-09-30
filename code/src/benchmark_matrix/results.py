"""Collect comparable headline metrics from a completed benchmark matrix."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.acccollab.config import load_acccollab_config
from src.multi_acccollab.config import load_multi_acccollab_config


class BenchmarkResultError(RuntimeError):
    """Raised when a matrix result is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    method: str
    dataset: str
    samples: int
    expected_samples: int
    accuracy: float
    parse_rate: float
    trials: int
    metrics_path: str


def collect_matrix_results(
    plan_path: str | Path,
    *,
    project_root: str | Path = ".",
) -> list[BenchmarkResult]:
    """Load and validate every method/dataset result named by a matrix plan."""
    root = Path(project_root).resolve()
    plan_file = _resolve(root, plan_path)
    plan = _read_json(plan_file)
    coverage = {str(key): int(value) for key, value in dict(plan["coverage"]).items()}
    methods = [str(method) for method in plan["methods"]]
    configs_by_model = dict(plan["configs"])
    results: list[BenchmarkResult] = []

    for model in [str(name) for name in plan["models"]]:
        model_configs = dict(configs_by_model[model])
        for method in methods:
            config_key = "original_evaluations" if method == "original" else "multi_evaluations"
            evaluation_configs = dict(model_configs[config_key])
            for dataset, expected_samples in coverage.items():
                config_path = _resolve(root, evaluation_configs[dataset])
                metrics_path = _metrics_path(root, method, config_path)
                metrics = _read_json(metrics_path)
                result = _extract_result(
                    model=model,
                    method=method,
                    dataset=dataset,
                    expected_samples=expected_samples,
                    metrics_path=metrics_path,
                    metrics=metrics,
                )
                if result.samples != expected_samples:
                    raise BenchmarkResultError(
                        f"Coverage mismatch for {model}/{method}/{dataset}: "
                        f"expected {expected_samples}, found {result.samples}"
                    )
                results.append(result)
    return results


def write_matrix_summary(
    results: Sequence[BenchmarkResult],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write long-form CSV plus JSON rows and current-minus-original deltas."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda row: (row.model, row.dataset, row.method))
    csv_path = destination / "benchmark_results.csv"
    fieldnames = list(BenchmarkResult.__dataclass_fields__)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in ordered)

    comparisons = _comparisons(ordered)
    json_path = destination / "benchmark_results.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "results": [asdict(row) for row in ordered],
                "current_minus_original": comparisons,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def _metrics_path(root: Path, method: str, config_path: Path) -> Path:
    if method == "original":
        config = load_acccollab_config(str(config_path))
        return _resolve(root, Path(config.run.output_dir) / "eval" / "metrics.json")
    if method == "current":
        config = load_multi_acccollab_config(str(config_path))
        return _resolve(root, config.paths.majority_eval_dir / "metrics.json")
    raise BenchmarkResultError(f"Unsupported method in matrix plan: {method}")


def _extract_result(
    *,
    model: str,
    method: str,
    dataset: str,
    expected_samples: int,
    metrics_path: Path,
    metrics: Mapping[str, Any],
) -> BenchmarkResult:
    aggregate = dict(metrics.get("aggregate") or {})
    try:
        if method == "original":
            samples = int(metrics["samples"])
            accuracy = float(dict(aggregate["final_accuracy"])["mean"])
            parse_rate = float(dict(aggregate["final_parse_rate"])["mean"])
        else:
            samples = int(aggregate["samples_per_trial"])
            accuracy = float(dict(aggregate["final_accuracy"])["mean"])
            parse_rate = float(dict(aggregate["parse_rate"])["mean"])
        trials = int(metrics["trials"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkResultError(f"Malformed metrics file: {metrics_path}") from exc
    return BenchmarkResult(
        model=model,
        method=method,
        dataset=dataset,
        samples=samples,
        expected_samples=expected_samples,
        accuracy=accuracy,
        parse_rate=parse_rate,
        trials=trials,
        metrics_path=str(metrics_path),
    )


def _comparisons(results: Sequence[BenchmarkResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, BenchmarkResult]] = {}
    for row in results:
        grouped.setdefault((row.model, row.dataset), {})[row.method] = row
    comparisons = []
    for (model, dataset), methods in sorted(grouped.items()):
        if set(methods) != {"original", "current"}:
            raise BenchmarkResultError(
                f"Cannot compare {model}/{dataset}; found methods {sorted(methods)}"
            )
        original = methods["original"].accuracy
        current = methods["current"].accuracy
        comparisons.append(
            {
                "model": model,
                "dataset": dataset,
                "original_accuracy": original,
                "current_accuracy": current,
                "absolute_delta": current - original,
                "delta_percentage_points": 100.0 * (current - original),
            }
        )
    return comparisons


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkResultError(f"Required result is missing: {path}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkResultError(f"Expected a JSON object: {path}")
    return payload
