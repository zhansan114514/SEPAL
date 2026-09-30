from __future__ import annotations

from pathlib import Path

import pytest

from src.acccollab.io import read_json, write_json, write_jsonl
from src.ablations.config import (
    ABLATION_MANIFEST_VERSION,
    DATASET_EXPECTED_SAMPLES,
    DATASET_LOGICAL_SHARDS,
    DATASET_ORDER,
    MODEL_LAYOUTS,
    POLICY_VARIANT_DATASETS,
    ROLE_NAMES,
    AblationConfigError,
    load_ablation_manifest,
)
from src.ablations.policy_lattice import aggregate_role_records, batch_seed
from src.utils.artifacts import stable_fingerprint


def test_all_policy_variants_cover_all_five_datasets() -> None:
    assert set(POLICY_VARIANT_DATASETS) == {
        "sft_only",
        "sft_base_critic",
        "sft_trained_critic",
    }
    assert all(
        tuple(datasets) == DATASET_ORDER
        for datasets in POLICY_VARIANT_DATASETS.values()
    )


def test_phi_and_mistral_layouts_resolve_completed_main_experiment_names() -> None:
    phi = MODEL_LAYOUTS["phi4"]
    assert phi.primary_multi_output == Path(
        "output/multi_acccollab/phi4_mini_instruct_mmlu_sft10k"
    )
    assert phi.dataset_multi_output("boolq") == Path(
        "output/multi_acccollab/phi4_mini_instruct_eval_boolq"
    )

    mistral = MODEL_LAYOUTS["mistral_v03"]
    assert mistral.primary_multi_output == Path(
        "output/multi_acccollab/"
        "mistral_7b_instruct_v03_mmlu_sft10k_trials5_singlebos"
    )
    assert mistral.dataset_multi_output("arc") == Path(
        "output/multi_acccollab/"
        "mistral_7b_instruct_v03_eval_arc_trials5_singlebos"
    )


def test_ablation_manifest_rejects_fingerprint_drift(tmp_path: Path) -> None:
    datasets = {
        dataset: {
            "expected_samples": DATASET_EXPECTED_SAMPLES[dataset],
            "logical_shards": DATASET_LOGICAL_SHARDS[dataset],
            "roles": {role: {} for role in ROLE_NAMES},
        }
        for dataset in DATASET_ORDER
    }
    payload = {
        "schema_version": 1,
        "implementation_version": ABLATION_MANIFEST_VERSION,
        "model": {"key": "llama3"},
        "datasets": datasets,
    }
    payload["fingerprint"] = stable_fingerprint(payload)
    path = tmp_path / "manifest.json"
    write_json(path, payload)
    assert load_ablation_manifest(path)["model"]["key"] == "llama3"

    payload["datasets"]["mmlu"]["expected_samples"] = 1
    write_json(path, payload)
    with pytest.raises(AblationConfigError, match="fingerprint"):
        load_ablation_manifest(path)


def test_ablation_batch_seed_preserves_logical_shard_identity() -> None:
    first = batch_seed(42, trial_index=0, shard_idx=0, batch_index=0)
    assert first == batch_seed(42, trial_index=0, shard_idx=0, batch_index=0)
    assert len(
        {
            batch_seed(42, trial_index=0, shard_idx=shard, batch_index=batch)
            for shard in range(4)
            for batch in range(3)
        }
    ) == 12


def test_round_aggregate_reparses_raw_completions(tmp_path: Path) -> None:
    samples = [
        {
            "sample_id": "yn-1",
            "acccollab_sample_index": 0,
            "task_type": "yes_no",
            "question": "Is the statement supported?",
            "choices": [],
            "passage": "The statement is supported.",
            "answer": "YES",
        },
        {
            "sample_id": "mc-1",
            "acccollab_sample_index": 1,
            "task_type": "multiple_choice",
            "question": "Choose B.",
            "choices": ["A", "B", "C"],
            "passage": "",
            "answer": "B",
        },
    ]
    raw = {
        "direct": [
            ["Final Answer: Yes", "Final Answer: No"],
            ["Final Answer: A", "Final Answer: C"],
        ],
        "evidence": [
            ["Final Answer: No", "Final Answer: No"],
            ["Final Answer: B", "Final Answer: C"],
        ],
        "verification": [
            ["Final Answer: Yes", "Final Answer: No"],
            ["Final Answer: B", "Final Answer: B"],
        ],
    }
    paths: dict[str, Path] = {}
    for role in ROLE_NAMES:
        rows = []
        for sample_index, sample in enumerate(samples):
            rows.append(
                {
                    "trial": 0,
                    "sample_id": sample["sample_id"],
                    "sample": sample,
                    "rounds": [
                        {
                            "round": round_index,
                            "actor": {
                                "completion": {
                                    "raw_response": response,
                                    "response": response,
                                    # Deliberately stale fields: the aggregate must
                                    # recover answers from raw text.
                                    "answer": None,
                                    "parsed": False,
                                    "correct": False,
                                    "truncated": False,
                                }
                            },
                        }
                        for round_index, response in enumerate(raw[role][sample_index])
                    ],
                }
            )
        path = tmp_path / f"{role}.jsonl"
        write_jsonl(path, rows)
        paths[role] = path

    output = aggregate_role_records(
        paths,
        output_dir=tmp_path / "aggregate",
        dataset_name="synthetic",
        variant="full_trained_rounds",
        expected_samples=2,
        reparse_completions=True,
    )
    metrics = read_json(output / "metrics.json")
    assert metrics["rescored_from_raw_response"] is True
    assert metrics["per_round"][0]["accuracy"] == 1.0
    assert metrics["per_round"][0]["parse_rate"] == 1.0
    assert metrics["per_round"][1]["accuracy"] == 0.0
    assert metrics["per_round"][1]["oracle_any_role_accuracy"] == 0.5
    assert metrics["headline"] == metrics["per_round"][1]
