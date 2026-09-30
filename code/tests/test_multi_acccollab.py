from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from src.acccollab.config import load_acccollab_config
from src.acccollab.io import read_json, write_json, write_jsonl
from src.acccollab.prompts import (
    ACCCOLLAB_PROMPT_VERSION,
    build_actor_deliberation_prompt,
    build_critic_prompt,
    build_initial_actor_prompt,
    prompt_version_for_sample,
    specialize_sample,
)
from src.multi_acccollab.config import (
    build_role_acccollab_config,
    config_snapshot,
    load_multi_acccollab_config,
    write_role_configs,
)
from src.multi_acccollab.majority import aggregate_role_evaluations, resolve_majority
from src.multi_acccollab.sft import select_matched_sft_rows, validate_balanced_sft_rows
from src.utils.artifacts import file_sha256, path_identity, write_adapter_success

MULTI_CONFIG = "configs/multi_acccollab/llama3_8b_instruct_mmlu_sft10k.yaml"
BASE_CONFIG = "configs/acccollab/llama3_8b_instruct_mmlu_validation1531.yaml"
SMOKE_CONFIGS = [
    "configs/multi_acccollab/llama3_8b_instruct_smoke_mmlu.yaml",
    "configs/multi_acccollab/llama3_8b_instruct_smoke_boolq.yaml",
    "configs/multi_acccollab/llama3_8b_instruct_smoke_sciq.yaml",
]
FORMAL_EVAL_CONFIGS = [
    "configs/multi_acccollab/llama3_8b_instruct_eval_boolq.yaml",
    "configs/multi_acccollab/llama3_8b_instruct_eval_sciq.yaml",
]


def _load_formal_script():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts" / "multi_acccollab"
    path = scripts_dir / "07_formal_experiment.py"
    previous_utils = sys.modules.pop("_utils", None)
    original_path = list(sys.path)
    sys.path.insert(0, str(scripts_dir))
    try:
        spec = importlib.util.spec_from_file_location("formal_experiment_test", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path
        sys.modules.pop("_utils", None)
        if previous_utils is not None:
            sys.modules["_utils"] = previous_utils


def test_primary_config_locks_sft_and_original_acccollab_splits() -> None:
    config = load_multi_acccollab_config(MULTI_CONFIG)
    base = config.base_config()

    assert config.sft.data.split == "train"
    assert config.sft.data.samples == 10000
    assert [role.name for role in config.roles] == ["direct", "evidence", "verification"]
    assert config.evaluation.use_judge is False
    assert base.data.preference.split == "validation"
    assert base.data.preference.expected_samples == 1531
    assert base.reward.rollouts == 10
    assert base.training.dpo.epochs == 3
    assert base.training.dpo.nll_weight == 1.0
    assert base.evaluation.trials == 1


def test_formal_role_scheduler_uses_four_gpu_data_and_parallel_training() -> None:
    formal = _load_formal_script()
    config = load_multi_acccollab_config(MULTI_CONFIG)
    role = config.role("evidence")
    role_config = config.paths.role_config(role.name)

    data_command = formal._role_stage_command(role, role_config, "critic-data")
    train_command = formal._role_stage_command(role, role_config, "critic-train")

    assert data_command[data_command.index("--devices") + 1] == "0,1,2,3"
    assert train_command[train_command.index("--devices") + 1] == "1"
    assert data_command[data_command.index("--training-devices") + 1] == "1"
    assert data_command[-2:] == ["--only", "critic-data"]
    assert train_command[-2:] == ["--only", "critic-train"]


def test_formal_plan_uses_sequential_cross_eval_on_one_gpu() -> None:
    formal = _load_formal_script()
    primary = load_multi_acccollab_config(MULTI_CONFIG)
    boolq = load_multi_acccollab_config(FORMAL_EVAL_CONFIGS[0])
    sciq = load_multi_acccollab_config(FORMAL_EVAL_CONFIGS[1])
    args = formal.build_parser().parse_args([])

    plan = formal._formal_plan(args, primary, boolq, sciq, (3,))

    schedule = plan["gpu_schedule"]
    assert "phase_1_concurrent" not in schedule
    assert schedule["phase_1_sequential"] == {
        "boolq": {"devices": [3], "shards_per_role": 1},
        "sciq": {"devices": [3], "shards_per_role": 1},
    }


def test_formal_completion_accepts_a_completed_stage_from_an_earlier_process(
    tmp_path: Path,
) -> None:
    formal = _load_formal_script()
    marker = tmp_path / "sft.json"
    marker.write_text(
        '{"pipeline":"multi_acccollab_formal","stage":"sft","status":"complete"}',
        encoding="utf-8",
    )

    assert formal._formal_stage_complete(marker, "sft") is True
    assert formal._formal_stage_complete(marker, "role-training") is False
    assert formal._formal_stage_complete(tmp_path / "missing.json", "sft") is False


def test_formal_recovery_reuses_only_manifest_authenticated_role_configs(
    tmp_path: Path,
) -> None:
    formal = _load_formal_script()
    loaded = load_multi_acccollab_config(MULTI_CONFIG)
    config = replace(
        loaded,
        run=replace(loaded.run, output_dir=str(tmp_path / "multi")),
    )
    roles = {}
    config.paths.role_config_dir.mkdir(parents=True)
    for role in config.roles:
        role_config = config.paths.role_config(role.name)
        role_config.write_text(f"role: {role.name}\n", encoding="utf-8")
        roles[role.name] = {"config_sha256": file_sha256(role_config)}
    write_json(
        config.paths.role_manifest,
        {
            "pipeline": "multi_acccollab",
            "roles": roles,
        },
    )

    formal._validate_existing_role_configs(config)
    args = formal.build_parser().parse_args(["--reuse-existing-role-configs"])
    assert args.reuse_existing_role_configs is True

    config.paths.role_config("direct").write_text("role: changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no longer matches recovery manifest"):
        formal._validate_existing_role_configs(config)


def test_all_smoke_configs_load_with_expected_dataset_profiles() -> None:
    configs = [load_multi_acccollab_config(path) for path in SMOKE_CONFIGS]

    assert [config.profile for config in configs] == [
        "smoke_train",
        "smoke_eval",
        "smoke_eval",
    ]
    assert [config.base_config().data.dataset for config in configs] == [
        "mmlu",
        "boolq",
        "sciq",
    ]
    assert configs[1].evaluation.policy_source_output_dir
    assert configs[2].evaluation.policy_source_output_dir


def test_benchmark_matrix_materializes_single_trial_paper_splits(tmp_path: Path) -> None:
    from src.acccollab.config import load_acccollab_config
    from src.benchmark_matrix.configs import materialize_model_configs
    from src.multi_acccollab.config import load_multi_acccollab_config

    paper_data = tmp_path / "paper_data"
    paper_data.mkdir()
    resolved = materialize_model_configs(
        "mistral",
        output_root=tmp_path / "matrix",
        benchmark_data_dir=paper_data,
    )

    primary = load_acccollab_config(str(resolved.original_primary))
    bbh = load_acccollab_config(str(resolved.original_evaluations["bbh"]))
    arc = load_acccollab_config(str(resolved.original_evaluations["arc"]))
    multi_bbh = load_multi_acccollab_config(str(resolved.multi_evaluations["bbh"]))

    assert primary.model.type == "mistral"
    assert primary.data.preference.expected_samples == 1531
    assert primary.data.preference_trials == 5
    assert primary.evaluation.trials == 1
    assert resolved.original_output_dir.name.endswith("_trials5_singlebos")
    assert resolved.multi_output_dir.name.endswith("_trials5_singlebos")
    assert bbh.data.eval.expected_samples == 1260
    assert arc.data.eval.expected_samples == 3548
    assert bbh.data.benchmark_data_dir == str(paper_data.resolve())
    assert multi_bbh.profile == "cross_eval"
    assert multi_bbh.evaluation.use_judge is False


def test_qwen3_matrix_uses_two_gpu_pool_and_h100_batches(tmp_path: Path) -> None:
    from src.acccollab.config import load_acccollab_config
    from src.benchmark_matrix.configs import materialize_model_configs
    from src.multi_acccollab.config import load_multi_acccollab_config

    paper_data = tmp_path / "paper_data"
    paper_data.mkdir()
    resolved = materialize_model_configs(
        "qwen3",
        output_root=tmp_path / "matrix",
        benchmark_data_dir=paper_data,
        devices=(5, 6),
    )

    original = load_acccollab_config(str(resolved.original_primary))
    current = load_multi_acccollab_config(str(resolved.multi_primary))
    original_boolq = load_acccollab_config(str(resolved.original_evaluations["boolq"]))
    current_boolq = load_multi_acccollab_config(str(resolved.multi_evaluations["boolq"]))

    assert original.model.type == "qwen3"
    assert original.runtime.generation_devices == [5, 6]
    assert original.runtime.training_devices == [5]
    assert original.runtime.batch.preference_generation == 16
    assert original.runtime.batch.evaluation == 256
    assert original.runtime.enforce_eager is True
    assert original.runtime.max_num_batched_tokens == 65536
    assert original.runtime.max_num_seqs == 256
    assert original_boolq.runtime.batch.evaluation == 64
    assert original_boolq.runtime.max_num_batched_tokens == 32768
    assert current_boolq.base_config().runtime.batch.evaluation == 256
    assert current_boolq.base_config().runtime.max_num_batched_tokens == 65536
    assert [role.device for role in current.roles] == [5, 6, 5]
    assert current.runtime.sft_generation_devices == [5, 6]
    assert current.runtime.sft_generation_batch_size == 256
    assert current.sft.training.batch_size == 16
    assert current.sft.training.gradient_accumulation_steps == 1


def test_qwen25_matrix_uses_a800_profile_and_single_trial(tmp_path: Path) -> None:
    from src.acccollab.config import load_acccollab_config
    from src.benchmark_matrix.configs import materialize_model_configs
    from src.multi_acccollab.config import load_multi_acccollab_config

    paper_data = tmp_path / "paper_data"
    paper_data.mkdir()
    resolved = materialize_model_configs(
        "qwen25",
        output_root=tmp_path / "matrix",
        benchmark_data_dir=paper_data,
        devices=(0, 1, 2, 3),
    )

    original = load_acccollab_config(str(resolved.original_primary))
    current = load_multi_acccollab_config(str(resolved.multi_primary))
    boolq = load_acccollab_config(str(resolved.original_evaluations["boolq"]))

    assert resolved.spec.slug == "qwen2_5_3b_instruct"
    assert original.model.type == "qwen2.5"
    assert original.model.name.endswith("/models/Qwen2.5-3B-Instruct")
    assert original.data.preference.expected_samples == 1531
    assert original.data.preference_trials == 1
    assert original.evaluation.trials == 1
    assert original.runtime.generation_devices == [0, 1, 2, 3]
    assert original.runtime.batch.preference_generation == 16
    assert original.runtime.batch.evaluation == 128
    assert original.runtime.max_num_batched_tokens == 32768
    assert original.runtime.max_num_seqs == 256
    assert original.training.dpo.batch_size == 4
    assert original.training.dpo.gradient_accumulation_steps == 1
    assert boolq.data.eval.expected_samples == 3270
    assert boolq.evaluation.trials == 1
    assert [role.device for role in current.roles] == [0, 1, 2]
    assert current.runtime.sft_generation_devices == [0, 1, 2, 3]
    assert current.runtime.sft_generation_batch_size == 128
    assert current.runtime.parallel_role_pipelines is True
    assert current.sft.data.expected_samples == 10000
    assert current.sft.training.batch_size == 16
    assert current.sft.training.gradient_accumulation_steps == 1
    assert current.evaluation.use_judge is False


@pytest.mark.parametrize(
    ("model_key", "slug", "model_type", "device_ids", "preference_trials"),
    [
        ("phi4mini", "phi4_mini_instruct", "phi4", (3,), 1),
        ("mistral_v03", "mistral_7b_instruct_v03", "mistral_v03", (1, 2), 5),
    ],
)
def test_new_model_matrix_profiles_are_isolated_and_resumable(
    tmp_path: Path,
    model_key: str,
    slug: str,
    model_type: str,
    device_ids: tuple[int, ...],
    preference_trials: int,
) -> None:
    from src.acccollab.config import load_acccollab_config
    from src.benchmark_matrix.configs import materialize_model_configs
    from src.multi_acccollab.config import load_multi_acccollab_config

    paper_data = tmp_path / "paper_data"
    paper_data.mkdir()
    resolved = materialize_model_configs(
        model_key,
        output_root=tmp_path / "matrix",
        benchmark_data_dir=paper_data,
        devices=device_ids,
    )

    original = load_acccollab_config(str(resolved.original_primary))
    current = load_multi_acccollab_config(str(resolved.multi_primary))

    assert resolved.spec.slug == slug
    assert original.model.type == model_type
    assert original.runtime.generation_devices == list(device_ids)
    assert original.runtime.training_devices == [device_ids[0]]
    assert original.data.preference_trials == preference_trials
    assert original.evaluation.trials == 1
    assert set(resolved.original_evaluations) == {"boolq", "sciq", "bbh", "arc", "mmlu"}
    assert set(resolved.multi_evaluations) == {"boolq", "sciq", "bbh", "arc", "mmlu"}
    assert [role.device for role in current.roles] == [
        device_ids[index % len(device_ids)] for index in range(3)
    ]
    assert current.runtime.sft_generation_devices == list(device_ids)
    assert current.evaluation.use_judge is False
    if model_key == "phi4mini":
        assert original.model.max_model_len == 4096
        assert original.runtime.batch.evaluation == 128
        assert current.runtime.parallel_role_pipelines is True
    else:
        assert resolved.original_output_dir.name.endswith("_trials5_singlebos")
        assert resolved.multi_output_dir.name.endswith("_trials5_singlebos")
        assert original.runtime.gpu_memory_utilization == 0.90
        assert original.runtime.batch.preference_generation == 8
        assert original.runtime.max_num_batched_tokens == 32768
        assert original.runtime.max_num_seqs == 256


def test_formal_cross_evaluation_configs_lock_coverage_and_source_policies() -> None:
    primary = load_multi_acccollab_config(MULTI_CONFIG)
    boolq, sciq = [load_multi_acccollab_config(path) for path in FORMAL_EVAL_CONFIGS]

    assert [boolq.profile, sciq.profile] == ["cross_eval", "cross_eval"]
    assert [boolq.base_config().data.dataset, sciq.base_config().data.dataset] == [
        "boolq",
        "sciq",
    ]
    assert [
        boolq.base_config().data.eval.expected_samples,
        sciq.base_config().data.eval.expected_samples,
    ] == [3270, 1000]
    assert [boolq.base_config().evaluation.trials, sciq.base_config().evaluation.trials] == [
        1,
        1,
    ]
    assert boolq.evaluation.policy_source_output_dir == primary.run.output_dir
    assert sciq.evaluation.policy_source_output_dir == primary.run.output_dir
    assert [role.actor_instruction for role in boolq.roles] == [
        role.actor_instruction for role in primary.roles
    ]
    assert [role.critic_instruction for role in sciq.roles] == [
        role.critic_instruction for role in primary.roles
    ]
    assert boolq.evaluation.use_judge is False
    assert sciq.evaluation.use_judge is False


def test_config_snapshot_authenticates_base_yaml_and_executable_sources() -> None:
    snapshot = config_snapshot(load_multi_acccollab_config(MULTI_CONFIG))

    assert len(snapshot["resolved_base_acccollab"]["sha256"]) == 64
    assert "src/multi_acccollab/config.py" in snapshot["source_tree"]
    assert len(snapshot["source_tree"]["src/multi_acccollab/config.py"]) == 64


def test_derived_role_uses_sft_actor_but_base_critic() -> None:
    config = load_multi_acccollab_config(MULTI_CONFIG)
    role = config.role("verification")
    derived = build_role_acccollab_config(config, role, require_sft_adapter=False)

    assert derived.initialization.actor_adapter
    assert derived.initialization.actor_adapter.endswith("verification_adapter")
    assert derived.initialization.critic_adapter is None
    assert derived.prompt_role.name == "verification"
    assert derived.runtime.generation_devices == [2]
    assert derived.runtime.training_devices == [2]
    assert derived.run.seed == config.run.seed + role.seed_offset
    assert derived.data.preference.sampling_seed == config.run.seed
    assert derived.data.eval.sampling_seed == config.run.seed
    assert derived.data.preference.expected_samples == 1531
    assert derived.training.dpo.learning_rate == 1.41e-5


def test_original_config_and_prompts_remain_unchanged_without_role() -> None:
    config = load_acccollab_config(BASE_CONFIG)
    sample = {
        "task_type": "multiple_choice",
        "question": "Which option is correct?",
        "choices": ["one", "two", "three", "four"],
        "choice_labels": ["A", "B", "C", "D"],
    }
    original = build_initial_actor_prompt(sample, "mmlu")

    assert config.initialization.actor_adapter is None
    assert config.initialization.critic_adapter is None
    assert config.prompt_role.name == "original"
    assert config.data.preference.sampling_seed is None
    assert config.data.eval.sampling_seed is None
    assert prompt_version_for_sample(sample) == ACCCOLLAB_PROMPT_VERSION
    assert original.startswith("Please answer the following multiple choice question")


def test_role_prefix_is_additive_and_versioned() -> None:
    sample = {
        "task_type": "multiple_choice",
        "question": "Which option is correct?",
        "choices": ["one", "two", "three", "four"],
        "choice_labels": ["A", "B", "C", "D"],
    }
    baseline = build_initial_actor_prompt(sample, "mmlu")
    specialized = specialize_sample(
        sample,
        role_name="evidence",
        actor_instruction="Use the key evidence.",
        critic_instruction="Check the key evidence.",
    )
    role_prompt = build_initial_actor_prompt(specialized, "mmlu")

    assert role_prompt.endswith(baseline)
    assert role_prompt.startswith("Role specialization (evidence):")
    assert prompt_version_for_sample(specialized).endswith(":evidence")


def test_bbh_yes_no_prompts_do_not_invent_a_passage() -> None:
    sample = {
        "task_type": "yes_no",
        "question": "Is this statement true?",
        "passage": "",
    }

    initial = build_initial_actor_prompt(sample, "bbh")
    deliberation = build_actor_deliberation_prompt(
        sample,
        "bbh",
        "Final Answer: Yes",
        "Check the expression.",
    )
    critic = build_critic_prompt(sample, "bbh", "Final Answer: Yes")

    assert "based on a passage" not in initial
    assert "Passage:" not in initial
    assert "Passage:" not in deliberation
    assert "Passage:" not in critic
    assert "yes-no question" in initial


def test_majority_and_fixed_direct_fallback() -> None:
    majority = resolve_majority(
        {"direct": "A", "evidence": "B", "verification": "B"},
        task_type="multiple_choice",
    )
    fallback = resolve_majority(
        {"direct": "A", "evidence": "B", "verification": "C"},
        task_type="multiple_choice",
    )
    parsed_pair = resolve_majority(
        {"direct": "A", "evidence": "A", "verification": None},
        task_type="multiple_choice",
    )

    assert majority["answer"] == "B"
    assert majority["source"] == "majority"
    assert fallback["answer"] == "A"
    assert fallback["source"] == "fixed_direct_fallback"
    assert fallback["selected_role"] == "direct"
    assert parsed_pair["answer"] == "A"
    assert parsed_pair["majority_reached"] is True


def test_sft_selection_is_matched_balanced_and_deduplicated() -> None:
    config = load_multi_acccollab_config(MULTI_CONFIG)
    low_temperature, high_temperature = config.sft.generation.temperatures[:2]
    candidates = []
    for sample_index in (0, 1):
        for role in config.roles:
            for temperature in (low_temperature, high_temperature):
                candidates.append(
                    {
                        "sample_index": sample_index,
                        "sample_id": f"sample-{sample_index}",
                        "role": role.name,
                        "temperature": temperature,
                        "prompt_version": f"prompt-{role.name}",
                        "prompt": f"prompt-{sample_index}-{role.name}",
                        "response": f"response-{temperature}",
                        "answer": "A",
                        "correct": not (
                            sample_index == 1 and role.name == "verification"
                        ),
                        "truncated": False,
                    }
                )

    selected = select_matched_sft_rows(config, candidates)

    assert len(selected) == 3
    assert validate_balanced_sft_rows(config, selected) == 1
    assert {row["sample_id"] for row in selected} == {"sample-0"}
    assert {row["temperature"] for row in selected} == {low_temperature}


def test_weight_identity_hashes_adapter_payload(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"first")
    first = path_identity(adapter, hash_weights=True)
    weights.write_bytes(b"second")
    second = path_identity(adapter, hash_weights=True)

    first_hash = first["selected"]["adapter_model.safetensors"]["sha256"]
    second_hash = second["selected"]["adapter_model.safetensors"]["sha256"]
    assert first_hash != second_hash


def test_streaming_aggregate_uses_final_round_majority_without_judge(tmp_path: Path) -> None:
    loaded = load_multi_acccollab_config(MULTI_CONFIG)
    config = replace(
        loaded,
        run=replace(loaded.run, output_dir=str(tmp_path / "multi")),
    )
    role_answers = {"direct": "A", "evidence": "B", "verification": "B"}
    for role in config.roles:
        records = [
            _evaluation_record(
                trial=trial,
                answer=role_answers[role.name],
                correct=role.name != "direct",
            )
            for trial in range(config.base_config().evaluation.trials)
        ]
        write_jsonl(config.paths.role_output(role.name) / "eval" / "records.jsonl", records)

    output = aggregate_role_evaluations(config, validate_inputs=False)
    metrics = read_json(output / "metrics.json")

    assert metrics["uses_judge_fallback"] is False
    assert metrics["decision_rule"] == "final_round_majority_then_fixed_direct"
    assert metrics["headline_metric"]["value"]["mean"] == 1.0
    assert metrics["aggregate"]["majority_coverage"]["mean"] == 1.0
    assert metrics["aggregate"]["per_role_accuracy"]["direct"]["mean"] == 0.0


def test_resolved_role_configs_authenticate_sft_and_preserve_original_dpo(tmp_path: Path) -> None:
    loaded = load_multi_acccollab_config(MULTI_CONFIG)
    config = replace(
        loaded,
        run=replace(loaded.run, output_dir=str(tmp_path / "multi")),
    )
    for role in config.roles:
        adapter = config.paths.sft_adapter(role.name)
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"test")
        write_adapter_success(
            adapter,
            training_fingerprint=f"test-{role.name}",
            base_model=config.base_config().model.name,
            output_dir=str(config.paths.sft_training_output(role.name)),
            training_kind="sft",
        )

    manifest = write_role_configs(config)
    direct = load_acccollab_config(str(config.paths.role_config("direct")))

    assert set(manifest["roles"]) == {"direct", "evidence", "verification"}
    assert direct.initialization.actor_adapter.endswith("direct_adapter")
    assert direct.initialization.critic_adapter is None
    assert direct.reward.rollouts == 10
    assert direct.training.dpo.epochs == 3
    assert direct.data.preference.expected_samples == 1531


def _evaluation_record(*, trial: int, answer: str, correct: bool) -> dict:
    completion = {
        "answer": answer,
        "parsed": True,
        "correct": correct,
        "truncated": False,
    }
    return {
        "pipeline": "acccollab_original",
        "decision_rule": "single_final_actor_answer",
        "uses_majority_vote": False,
        "uses_judge_fallback": False,
        "trial": trial,
        "sample_id": "mmlu_test_00000000",
        "sample": {
            "sample_id": "mmlu_test_00000000",
            "acccollab_sample_index": 0,
            "task_type": "multiple_choice",
            "answer": "B",
        },
        "rounds": [
            {"round": round_index, "actor": {"completion": dict(completion)}}
            for round_index in range(5)
        ],
    }
