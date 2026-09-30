from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.acccollab.config import ACCCollabConfig, load_acccollab_config
from src.acccollab.registry import (
    ACCCollabRegistryError,
    PolicyState,
    actor_data_state,
    build_final_registry,
    critic_data_state,
    evaluation_state,
    final_registry_matches,
    iteration_registry_matches,
    write_final_registry,
    write_iteration_registry,
)
from src.acccollab.stages import (
    build_pipeline_stages,
    evaluation_stage_fingerprint,
    preference_stage_fingerprint,
    validate_stage_success,
    write_stage_success,
)
import src.acccollab.stages as stages_module

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts" / "acccollab"
CONFIG_DIR = ROOT / "configs" / "acccollab"
def _load_script_module(name: str, filename: str):
    path = SCRIPTS_DIR / filename
    helper_names = ("_utils", "_preference_data", "_training")
    saved_helpers = {helper: sys.modules.get(helper) for helper in helper_names}
    original_path = list(sys.path)
    for helper in helper_names:
        sys.modules.pop(helper, None)
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path
        for helper in helper_names:
            sys.modules.pop(helper, None)
            previous = saved_helpers[helper]
            if previous is not None:
                sys.modules[helper] = previous


def _config(tmp_path: Path, *, plus: bool = True) -> ACCCollabConfig:
    name = (
        "llama3_8b_instruct_mmlu_plus.yaml"
        if plus
        else "llama3_8b_instruct_mmlu_original_smoke.yaml"
    )
    config = load_acccollab_config(str(CONFIG_DIR / name))
    return replace(
        config,
        run=replace(config.run, output_dir=str(tmp_path / "acccollab-run")),
    )


def _create_completed_adapter(
    config: ACCCollabConfig,
    iteration: int,
    role: str,
    *,
    fingerprint: str | None = None,
) -> Path:
    adapter = config.paths.adapter_dir(iteration, role)
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"role": role, "iteration": iteration}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(
        f"weights-{role}-{iteration}".encode()
    )
    (adapter / "_SUCCESS").write_text(
        json.dumps(
            {
                "status": "complete",
                "training_fingerprint": fingerprint or f"fp-{role}-{iteration}",
            }
        ),
        encoding="utf-8",
    )
    return adapter


def _external_evaluation_config(
    source: ACCCollabConfig,
    tmp_path: Path,
) -> ACCCollabConfig:
    return replace(
        source,
        run=replace(source.run, output_dir=str(tmp_path / "evaluation-run")),
        evaluation=replace(
            source.evaluation,
            policy_output_dir=source.run.output_dir,
        ),
    )


def test_plus_policy_states_follow_critic_then_actor_alternation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    actor_one = _create_completed_adapter(config, 1, "actor")
    critic_one = _create_completed_adapter(config, 1, "critic")

    critic_two_data = critic_data_state(config, 2)
    assert critic_two_data.actor_adapter == str(actor_one)
    assert critic_two_data.critic_adapter == str(critic_one)
    assert critic_two_data.actor_iteration == 1
    assert critic_two_data.critic_iteration == 1

    critic_two = _create_completed_adapter(config, 2, "critic")
    actor_two_data = actor_data_state(config, 2)
    assert actor_two_data.actor_adapter == str(actor_one)
    assert actor_two_data.critic_adapter == str(critic_two)
    assert actor_two_data.actor_iteration == 1
    assert actor_two_data.critic_iteration == 2

    actor_two = _create_completed_adapter(config, 2, "actor")
    final = evaluation_state(config)
    assert final.actor_adapter == str(actor_two)
    assert final.critic_adapter == str(critic_two)
    assert final.actor_iteration == 2
    assert final.critic_iteration == 2


def test_registries_match_exact_adapter_identities(tmp_path: Path) -> None:
    config = _config(tmp_path, plus=False)
    actor = _create_completed_adapter(config, 1, "actor")
    _create_completed_adapter(config, 1, "critic")

    iteration_payload = write_iteration_registry(config, 1)
    final_payload = write_final_registry(config)
    assert iteration_registry_matches(config, 1)
    assert final_registry_matches(config)
    assert iteration_payload["training_order"] == "critic_then_actor"
    assert final_payload == build_final_registry(config)
    assert final_payload["state"]["actor_iteration"] == 1

    (actor / "adapter_config.json").write_text('{"mutated": true}', encoding="utf-8")
    assert not iteration_registry_matches(config, 1)
    assert not final_registry_matches(config)


def test_registry_invalidates_when_success_marker_changes(tmp_path: Path) -> None:
    config = _config(tmp_path, plus=False)
    actor = _create_completed_adapter(config, 1, "actor")
    _create_completed_adapter(config, 1, "critic")
    write_iteration_registry(config, 1)
    assert iteration_registry_matches(config, 1)

    (actor / "_SUCCESS").write_text(
        json.dumps({"status": "complete", "training_fingerprint": "different"}),
        encoding="utf-8",
    )
    assert not iteration_registry_matches(config, 1)


def test_external_evaluation_loads_authenticated_final_registry(tmp_path: Path) -> None:
    source = _config(tmp_path, plus=False)
    actor = _create_completed_adapter(source, 1, "actor")
    critic = _create_completed_adapter(source, 1, "critic")
    write_final_registry(source)
    evaluation_config = _external_evaluation_config(source, tmp_path)

    state = evaluation_state(evaluation_config)

    assert state == PolicyState(str(actor), str(critic), 1, 1)


def test_external_evaluation_rejects_identity_and_adapter_redirect_tampering(
    tmp_path: Path,
) -> None:
    source = _config(tmp_path, plus=False)
    _create_completed_adapter(source, 1, "actor")
    critic = _create_completed_adapter(source, 1, "critic")
    write_final_registry(source)
    evaluation_config = _external_evaluation_config(source, tmp_path)

    payload = json.loads(source.paths.final_registry.read_text(encoding="utf-8"))
    payload["actor_identity"] = {"tampered": True}
    source.paths.final_registry.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ACCCollabRegistryError, match="actor_identity"):
        evaluation_state(evaluation_config)

    write_final_registry(source)
    payload = json.loads(source.paths.final_registry.read_text(encoding="utf-8"))
    payload["state"]["actor_adapter"] = str(critic)
    payload["state"]["actor_policy"] = str(critic)
    source.paths.final_registry.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ACCCollabRegistryError, match="redirects actor"):
        evaluation_state(evaluation_config)


def test_external_adapter_update_changes_evaluation_fingerprint(tmp_path: Path) -> None:
    source = _config(tmp_path, plus=False)
    actor = _create_completed_adapter(source, 1, "actor")
    _create_completed_adapter(source, 1, "critic")
    write_final_registry(source)
    evaluation_config = _external_evaluation_config(source, tmp_path)

    original_state = evaluation_state(evaluation_config)
    original = evaluation_stage_fingerprint(
        evaluation_config,
        state=original_state,
        num_shards=2,
    )

    (actor / "adapter_model.safetensors").write_bytes(b"legitimate-updated-weights")
    write_final_registry(source)
    updated_state = evaluation_state(evaluation_config)
    updated = evaluation_stage_fingerprint(
        evaluation_config,
        state=updated_state,
        num_shards=2,
    )

    assert updated != original


def test_stage_success_marker_detects_artifact_hash_and_size_changes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.jsonl"
    artifact.write_text("abc", encoding="utf-8")
    marker = tmp_path / "_SUCCESS"
    write_stage_success(
        marker,
        stage="critic_dpo_data",
        fingerprint="semantic-fingerprint",
        artifacts={"data": artifact},
    )

    assert validate_stage_success(
        marker,
        expected_stage="critic_dpo_data",
        expected_fingerprint="semantic-fingerprint",
    )
    artifact.write_text("xyz", encoding="utf-8")
    assert (
        validate_stage_success(
            marker,
            expected_stage="critic_dpo_data",
            expected_fingerprint="semantic-fingerprint",
        )
        is None
    )
    assert validate_stage_success(
        marker,
        expected_stage="critic_dpo_data",
        expected_fingerprint="semantic-fingerprint",
        verify_hashes=False,
    )

    artifact.write_text("longer", encoding="utf-8")
    assert (
        validate_stage_success(
            marker,
            expected_stage="critic_dpo_data",
            expected_fingerprint="semantic-fingerprint",
            verify_hashes=False,
        )
        is None
    )


def test_prompt_version_is_part_of_preference_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, plus=False)
    state = PolicyState(None, None, 0, 0)
    original = preference_stage_fingerprint(
        config,
        agent="critic",
        iteration=1,
        state=state,
        num_shards=1,
    )

    monkeypatch.setattr(stages_module, "ACCCOLLAB_PROMPT_VERSION", "changed-prompt")
    changed = preference_stage_fingerprint(
        config,
        agent="critic",
        iteration=1,
        state=state,
        num_shards=1,
    )
    assert changed != original


def test_plus_pipeline_has_two_strict_alternations_then_evaluation(tmp_path: Path) -> None:
    stages = build_pipeline_stages(_config(tmp_path))

    assert [stage.key for stage in stages] == [
        "iteration-01-critic-data",
        "iteration-01-critic-train",
        "iteration-01-actor-data",
        "iteration-01-actor-train",
        "iteration-02-critic-data",
        "iteration-02-critic-train",
        "iteration-02-actor-data",
        "iteration-02-actor-train",
        "evaluate",
    ]
    assert [stage.selector for stage in stages[:-1]] == [
        "critic-data",
        "critic-train",
        "actor-data",
        "actor-train",
    ] * 2
    assert stages[-1].iteration is None


def test_pipeline_selectors_and_commands_are_unambiguous(tmp_path: Path) -> None:
    pipeline = _load_script_module("test_acccollab_pipeline", "06_pipeline.py")
    stages = build_pipeline_stages(_config(tmp_path))

    selected = pipeline._resolve_stage_tokens(stages, ["critic-data"], option="--only")
    assert selected == {"iteration-01-critic-data", "iteration-02-critic-data"}
    with pytest.raises(ValueError, match="Unknown --only stage"):
        pipeline._resolve_stage_tokens(stages, ["critic-dtaa"], option="--only")

    critic_stage = stages[0]
    critic_command = pipeline._script_command(
        critic_stage,
        config_path="config.yaml",
        overrides=["run.seed=8"],
        extra=("--iteration", "1", "--device", "0"),
    )
    assert critic_command.count("--iteration") == 1
    assert critic_command[critic_command.index("--iteration") + 1] == "1"

    evaluation_command = pipeline._script_command(
        stages[-1],
        config_path="config.yaml",
        overrides=[],
        extra=("--device", "0", "--shard-idx", "0", "--num-shards", "1"),
    )
    assert "--iteration" not in evaluation_command


def test_single_and_multi_gpu_shard_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _load_script_module("test_acccollab_pipeline_commands", "06_pipeline.py")
    stage = build_pipeline_stages(_config(tmp_path))[0]
    completed: list[list[str]] = []
    monkeypatch.setattr(pipeline, "_run_process", lambda command: completed.append(command))

    result = pipeline._run_shardable_stage(
        stage,
        config_path="config.yaml",
        overrides=[],
        devices=[3],
    )
    assert result["mode"] == "single_process"
    assert completed[0].count("--iteration") == 1
    assert completed[0][completed[0].index("--device") + 1] == "3"

    worker_commands: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command: list[str], **_kwargs: Any) -> None:
            self.command = command
            worker_commands.append(command)

        def wait(self) -> int:
            return 0

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:  # pragma: no cover - failure cleanup only
            raise AssertionError("successful workers must not be terminated")

        def kill(self) -> None:  # pragma: no cover - failure cleanup only
            raise AssertionError("successful workers must not be killed")

    completed.clear()
    monkeypatch.setattr(pipeline.subprocess, "Popen", FakeProcess)
    result = pipeline._run_shardable_stage(
        stage,
        config_path="config.yaml",
        overrides=[],
        devices=[2, 5],
    )

    assert result == {"mode": "data_parallel", "devices": [2, 5], "num_shards": 2}
    assert len(worker_commands) == 2
    assert [command[command.index("--device") + 1] for command in worker_commands] == [
        "2",
        "5",
    ]
    assert all(command.count("--iteration") == 1 for command in worker_commands)
    assert len(completed) == 1
    assert "--merge-shards" in completed[0]
    assert completed[0].count("--iteration") == 1


def test_empty_preference_shard_never_builds_vllm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preference = _load_script_module(
        "test_acccollab_preference_data",
        "_preference_data.py",
    )
    config = _config(tmp_path, plus=False)
    only_sample = {
        "sample_id": "only-sample",
        "acccollab_sample_index": 0,
        "task_type": "multiple_choice",
        "question": "q",
        "choices": ["a", "b"],
        "answer": "A",
    }
    monkeypatch.setattr(preference, "load_preference_samples", lambda _config: [only_sample])

    def forbidden_bundle(*_args: Any, **_kwargs: Any):
        raise AssertionError("empty shards must not initialize vLLM")

    monkeypatch.setattr(preference, "build_policy_bundle", forbidden_bundle)
    shard_dir = preference.generate_preference_shard(
        config,
        agent="critic",
        iteration=1,
        device=0,
        shard_idx=1,
        num_shards=2,
    )

    assert (shard_dir / "_SUCCESS").is_file()
    assert (shard_dir / "pairs.jsonl").read_text(encoding="utf-8") == ""
    assert (shard_dir / "trajectories.jsonl").read_text(encoding="utf-8") == ""
    marker = json.loads((shard_dir / "_SUCCESS").read_text(encoding="utf-8"))
    assert marker["metadata"]["samples"] == 0
    assert marker["metadata"]["pairs"] == 0
