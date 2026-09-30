#!/usr/bin/env python3
"""Verify every paper result against the packaged aggregate artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
LOCKED = ROOT / "results" / "locked"
DATASETS = ("BoolQ", "MMLU", "BBH", "SciQ", "ARC")
DATASET_SLUG = {name: name.lower() for name in DATASETS}
MODEL_KEY = {
    "Meta-Llama-3-8B-Instruct": "llama3_8b",
    "Qwen2.5-3B-Instruct": "qwen25_3b",
    "Gemma-2-2B-it": "gemma2_2b",
    "Phi-4-mini-instruct": "phi4_mini",
    "Mistral-7B-Instruct-v0.3": "mistral7b_v03",
}
SHORT_MODEL_KEY = {
    "Llama-3-8B": "llama3_8b",
    "Qwen2.5-3B": "qwen25_3b",
    "Gemma-2-2B": "gemma2_2b",
    "Phi-4-mini": "phi4_mini",
    "Mistral-7B": "mistral7b_v03",
}
BASELINE_PATH = {
    "direct": "direct",
    "debate": "debate",
    "som_2x": "som_2x",
    "som_4x": "som_4x",
}
VARIANT_PATH = {
    "SFT-only": "sft_only",
    "SFT+Base-C": "sft_base_critic",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def baseline_accuracy(payload: dict) -> float:
    if "accuracy" in payload:
        return float(payload["accuracy"])
    return float(payload["final"]["accuracy"])


def acc_accuracy(payload: dict) -> float:
    trials = payload.get("aggregate", {}).get("trial_values", [])
    if trials:
        trial_zero = next((item for item in trials if int(item["trial"]) == 0), None)
        if trial_zero is not None:
            return float(trial_zero["accuracy"])
    return float(payload["headline_metric"]["value"]["mean"])


def assert_close(label: str, observed: float, expected: float, tolerance: float) -> None:
    if abs(observed - expected) > tolerance:
        raise AssertionError(
            f"{label}: packaged={observed:.12f}, locked={expected:.12f}, "
            f"difference={observed - expected:+.12f}"
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing_tokens = {"", "-", "--", "n/a", "na", "nan", "none", "null"}
    for row_number, row in enumerate(rows, start=2):
        for column, value in row.items():
            if value is None or value.strip().lower() in missing_tokens:
                raise AssertionError(f"Missing value in {path.name}:{row_number}:{column}")
    return rows


def packaged_ablation_accuracy(model_key: str, dataset: str, variant: str) -> float:
    base = ROOT / "results/ablations" / model_key
    if variant in ('SFT+Trained-C','No-SFT'):
        slug={'SFT+Trained-C':'sft_trained_critic','No-SFT':'no_sft_full'}[variant]
        payload=load_json(ROOT/'results/extended_ablations'/model_key/dataset/(slug+'.json'))
        return float(payload['headline']['accuracy'])*100.0
    if variant in VARIANT_PATH:
        payload = load_json(
            base
            / "policy_lattice"
            / dataset
            / VARIANT_PATH[variant]
            / "aggregate/metrics.json"
        )
        return float(payload["headline"]["accuracy"]) * 100.0
    if variant in {"Full-R0", "Full-R4"}:
        payload = load_json(base / "offline_full" / dataset / "metrics.json")
        if variant == "Full-R0":
            return float(payload["per_round"][0]["accuracy"]) * 100.0
        return float(payload["headline"]["accuracy"]) * 100.0
    raise AssertionError(f"Unknown variant: {variant}")


def verify_main() -> int:
    rows = read_csv(LOCKED / "main_results.csv")
    if len(rows) != 25:
        raise AssertionError(f"Expected 25 main rows, found {len(rows)}")
    historical_path = ROOT / "results/main/historical_llama_gemma_extract.csv"
    historical = {
        (row["model"], row["dataset"]): row for row in read_csv(historical_path)
    }

    checks = 0
    for row in rows:
        model = row["model"]
        dataset = row["dataset"]
        model_key = MODEL_KEY[model]
        dataset_key = DATASET_SLUG[dataset]
        for column, artifact_method in BASELINE_PATH.items():
            payload = load_json(
                ROOT
                / "results/main/baselines"
                / model_key
                / dataset_key
                / artifact_method
                / "aggregate_metrics.json"
            )
            assert_close(
                f"main:{model}:{dataset}:{column}",
                baseline_accuracy(payload),
                float(row[column]),
                5e-12,
            )
            checks += 1

        full = load_json(
            ROOT
            / "results/ablations"
            / model_key
            / "offline_full"
            / dataset_key
            / "metrics.json"
        )
        assert_close(
            f"main:{model}:{dataset}:multi_actor_critic",
            float(full["headline"]["accuracy"]),
            float(row["multi_actor_critic"]),
            5e-12,
        )
        checks += 1

        if model in {
            "Qwen2.5-3B-Instruct",
            "Phi-4-mini-instruct",
            "Mistral-7B-Instruct-v0.3",
        }:
            acc_payload = load_json(
                ROOT
                / "results/main/acccollab"
                / model_key
                / dataset_key
                / "metrics.json"
            )
            observed_acc = acc_accuracy(acc_payload)
        else:
            source_row = historical[(model, dataset)]
            observed_acc = float(source_row["acccollab"])
            for column in (
                "direct",
                "debate",
                "som_2x",
                "som_4x",
                "multi_actor_critic",
            ):
                assert_close(
                    f"historical-crosscheck:{model}:{dataset}:{column}",
                    float(source_row[column]),
                    float(row[column]),
                    5e-12,
                )
                checks += 1
        assert_close(
            f"main:{model}:{dataset}:acccollab",
            observed_acc,
            float(row["acccollab"]),
            5e-12,
        )
        checks += 1
    return checks


def verify_ablations() -> int:
    rows = read_csv(LOCKED / "ablation_results.csv")
    if len(rows) != 150:
        raise AssertionError(f"Expected 150 ablation rows, found {len(rows)}")
    checks = 0
    for row in rows:
        model_key = SHORT_MODEL_KEY[row["model"]]
        dataset = DATASET_SLUG[row["dataset"]]
        variant = row["variant"]
        observed = packaged_ablation_accuracy(model_key, dataset, variant)
        assert_close(
            f"ablation:{row['model']}:{variant}:{row['dataset']}",
            observed,
            float(row["accuracy"]),
            0.0051,
        )
        checks += 1
    return checks


def verify_contrasts() -> int:
    rows = read_csv(LOCKED / "component_contrasts.csv")
    if len(rows) != 4:
        raise AssertionError(f"Expected 4 contrast rows, found {len(rows)}")
    checks = 0
    for row in rows:
        deltas = []
        for model_key in SHORT_MODEL_KEY.values():
            for dataset_name in DATASETS:
                dataset = DATASET_SLUG[dataset_name]
                upper = packaged_ablation_accuracy(
                    model_key, dataset, row["upper_variant"]
                )
                lower = packaged_ablation_accuracy(
                    model_key, dataset, row["lower_variant"]
                )
                deltas.append(upper - lower)
        observed = mean(deltas)
        assert_close(
            f"contrast:{row['contrast']}",
            observed,
            float(row["mean_change_points"]),
            5e-12,
        )
        positive = sum(delta > 0 for delta in deltas)
        if positive != int(row["positive_cells"]):
            raise AssertionError(
                f"contrast:{row['contrast']}: positive cells "
                f"packaged={positive}, locked={row['positive_cells']}"
            )
        checks += 2
    return checks


def verify_rounds() -> int:
    rows = read_csv(LOCKED / "round_results.csv")
    if len(rows) != 30:
        raise AssertionError(f"Expected 30 round rows, found {len(rows)}")
    expected = {(row["model"], row["round"]): float(row["macro_accuracy"]) for row in rows}
    by_round: dict[int, list[float]] = defaultdict(list)
    checks = 0
    for paper_model, model_key in SHORT_MODEL_KEY.items():
        for round_index in range(5):
            values = []
            for dataset in DATASETS:
                payload = load_json(
                    ROOT
                    / "results/ablations"
                    / model_key
                    / "offline_full"
                    / DATASET_SLUG[dataset]
                    / "metrics.json"
                )
                values.append(float(payload["per_round"][round_index]["accuracy"]) * 100.0)
            observed = mean(values)
            by_round[round_index].append(observed)
            assert_close(
                f"round:{paper_model}:R{round_index}",
                observed,
                expected[(paper_model, f"R{round_index}")],
                0.0051,
            )
            checks += 1
    for round_index, values in by_round.items():
        assert_close(
            f"round:Mean:R{round_index}",
            mean(values),
            expected[("Mean", f"R{round_index}")],
            0.0051,
        )
        checks += 1
    return checks


def verify_decision_diagnostics() -> int:
    rows = read_csv(LOCKED / "decision_diagnostics.csv")
    if len(rows) != 25:
        raise AssertionError(f"Expected 25 decision-diagnostic rows, found {len(rows)}")
    pairs = {
        "de": "direct+evidence",
        "dv": "direct+verification",
        "ev": "evidence+verification",
    }
    checks = 0
    for row in rows:
        payload = load_json(
            ROOT
            / "results/ablations"
            / row["model"]
            / "offline_full"
            / row["dataset"]
            / "metrics.json"
        )
        headline = payload["headline"]
        headline_fields = {
            "accuracy": headline["accuracy"],
            "majority_coverage": headline["majority_coverage"],
            "unanimous_rate": headline["unanimous_rate"],
            "oracle_any_role_accuracy": headline["oracle_any_role_accuracy"],
            "fallback_rate": headline["fallback_rate"],
            "parse_rate": headline["parse_rate"],
            **{
                f"{role}_accuracy": headline["per_role"][role]["accuracy"]
                for role in ("direct", "evidence", "verification")
            },
        }
        if int(row["samples"]) != int(headline["samples"]):
            raise AssertionError(f"diagnostic sample mismatch: {row['model']}:{row['dataset']}")
        checks += 1
        for field, observed in headline_fields.items():
            assert_close(
                f"diagnostic:{row['model']}:{row['dataset']}:{field}",
                float(observed),
                float(row[field]),
                5e-12,
            )
            checks += 1
        for short, pair in pairs.items():
            for suffix, field in (("agreement_coverage", "coverage"), ("conditional_accuracy", "conditional_accuracy")):
                assert_close(
                    f"diagnostic:{row['model']}:{row['dataset']}:{short}_{suffix}",
                    float(payload["pair_agreement"][pair][field]),
                    float(row[f"{short}_{suffix}"]),
                    5e-12,
                )
                checks += 1
    return checks


def verify_inventory() -> int:
    manifest = load_json(ROOT / "PROVENANCE_MANIFEST.json")
    checks = 0
    for record in manifest["files"]:
        path = ROOT / record["packaged_path"]
        if not path.is_file():
            raise AssertionError(f"Missing packaged file: {record['packaged_path']}")
        if path.stat().st_size != record["bytes"]:
            raise AssertionError(f"File-size mismatch: {record['packaged_path']}")
        if record.get("sha256"):
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != record["sha256"]:
                raise AssertionError(f"SHA-256 mismatch: {record['packaged_path']}")
        checks += 1
    return checks


def main() -> None:
    counts = {
        "main_artifact_checks": verify_main(),
        "ablation_cell_checks": verify_ablations(),
        "component_contrast_checks": verify_contrasts(),
        "round_macro_checks": verify_rounds(),
        "decision_diagnostic_checks": verify_decision_diagnostics(),
        "inventory_checks": verify_inventory(),
    }
    print(json.dumps({"status": "pass", **counts}, indent=2))


if __name__ == "__main__":
    main()
