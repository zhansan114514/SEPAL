from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from src.acccollab.policy import RolePolicy
from src.inference.vllm_server import VLLMInference
from src.training import dpo_trainer
from src.training._dpo_runner import _prepare_policy_model
import src.training.lora_config as lora_config_module


def test_first_dpo_iteration_creates_lora_directly_from_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    peft_config = object()
    monkeypatch.setattr(
        lora_config_module,
        "get_lora_config",
        lambda **_kwargs: peft_config,
    )

    prepared, returned_peft_config, resolved = _prepare_policy_model(
        model,
        initial_lora_path=None,
        model_type="llama3",
        lora_r=256,
        lora_alpha=512,
    )

    assert prepared is model
    assert returned_peft_config is peft_config
    assert resolved is None


def test_later_dpo_iteration_continues_trainable_default_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = tmp_path / "prior_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    base_model = object()
    loaded_model = types.SimpleNamespace(set_adapter_calls=[])

    def set_adapter(name: str) -> None:
        loaded_model.set_adapter_calls.append(name)

    loaded_model.set_adapter = set_adapter
    calls: list[dict[str, Any]] = []

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model: object, path: str, **kwargs: Any):
            calls.append({"model": model, "path": path, **kwargs})
            return loaded_model

    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    prepared, peft_config, resolved = _prepare_policy_model(
        base_model,
        initial_lora_path=str(adapter),
        model_type="llama3",
        lora_r=256,
        lora_alpha=512,
    )

    assert prepared is loaded_model
    assert peft_config is None
    assert resolved == str(adapter)
    assert calls == [
        {
            "model": base_model,
            "path": str(adapter),
            "adapter_name": "default",
            "is_trainable": True,
        }
    ]
    assert loaded_model.set_adapter_calls == ["default"]


def test_train_dpo_passes_continuation_and_caller_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_subprocess(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "trained-adapter"

    monkeypatch.setattr(dpo_trainer, "run_training_subprocess", fake_subprocess)
    dataset = object()
    result = dpo_trainer.train_dpo(
        model_name_or_path="base-model",
        preference_dataset=dataset,
        output_dir="training-output",
        initial_lora_path="prior-adapter",
        lora_r=256,
        lora_alpha=512,
        training_fingerprint="exact-fingerprint",
    )

    assert result == "trained-adapter"
    assert captured["dataset"] is dataset
    assert captured["config"]["initial_lora_path"] == "prior-adapter"
    assert captured["config"]["training_fingerprint"] == "exact-fingerprint"
    assert captured["config"]["lora_r"] == 256
    assert captured["config"]["lora_alpha"] == 512


class _FakeRoleEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        self.calls.append({"method": "base", "prompts": prompts, **kwargs})
        return ["base-output"] * len(prompts)

    def generate_with_lora(
        self,
        prompts: list[str],
        request: object,
        **kwargs: Any,
    ) -> list[str]:
        self.calls.append(
            {"method": "lora", "prompts": prompts, "request": request, **kwargs}
        )
        return ["lora-output"] * len(prompts)


def test_role_policy_forwards_seed_for_base_and_lora_generation() -> None:
    engine = _FakeRoleEngine()
    base = RolePolicy(engine, role="actor")
    request = object()
    lora = RolePolicy(engine, role="critic", lora_request=request)

    base.generate(
        ["p0"],
        max_tokens=32,
        temperature=0.7,
        top_p=0.9,
        enable_thinking=False,
        seed=17,
    )
    lora.generate(
        ["p1"],
        max_tokens=64,
        temperature=0.2,
        top_p=0.8,
        enable_thinking=True,
        seed=23,
    )
    base.generate(
        ["p2", "p3"],
        max_tokens=32,
        temperature=0.7,
        top_p=0.9,
        enable_thinking=False,
        seed=(37, 41),
    )

    assert engine.calls[0]["method"] == "base"
    assert engine.calls[0]["seed"] == 17
    assert engine.calls[1]["method"] == "lora"
    assert engine.calls[1]["request"] is request
    assert engine.calls[1]["seed"] == 23
    assert engine.calls[2]["method"] == "base"
    assert engine.calls[2]["seed"] == [37, 41]


def test_vllm_wrapper_supports_scalar_and_per_request_seeds_for_all_lora_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    llm_calls: list[dict[str, Any]] = []

    class FakeLLM:
        def generate(
            self,
            prompts: list[str],
            params: object,
            **kwargs: Any,
        ) -> list[str]:
            llm_calls.append({"prompts": prompts, "params": params, **kwargs})
            return ["raw-output"] * len(prompts)

    inference = object.__new__(VLLMInference)
    inference.model_name = "fake-model"
    inference._max_model_len = 4096
    inference._llm = FakeLLM()
    inference._tokenizer = None
    inference._enable_lora = True
    monkeypatch.setattr(inference, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        inference,
        "_format_prompts_for_thinking",
        lambda prompts, _enable_thinking: prompts,
    )
    monkeypatch.setattr(
        inference,
        "_record_outputs",
        lambda outputs, *, max_tokens: list(outputs),
    )

    assert inference.generate(["p0", "p1"], seed=29) == ["raw-output"] * 2
    assert isinstance(llm_calls[-1]["params"], FakeSamplingParams)
    assert llm_calls[-1]["params"].kwargs["seed"] == 29
    assert llm_calls[-1]["tokenization_kwargs"] == {
        "max_length": 3840,
        "truncation": True,
    }

    assert inference.generate(["p0", "p1"], seed=[37, 41]) == ["raw-output"] * 2
    assert [params.kwargs["seed"] for params in llm_calls[-1]["params"]] == [37, 41]

    request = object()
    assert inference.generate_with_lora(
        ["p0", "p1"], request, seed=31
    ) == ["raw-output"] * 2
    assert isinstance(llm_calls[-1]["params"], FakeSamplingParams)
    assert llm_calls[-1]["params"].kwargs["seed"] == 31
    assert llm_calls[-1]["lora_request"] is request
    assert llm_calls[-1]["tokenization_kwargs"] == {
        "max_length": 3840,
        "truncation": True,
    }

    assert inference.generate_with_lora(
        ["p0", "p1"], request, seed=[43, 47]
    ) == ["raw-output"] * 2
    assert [params.kwargs["seed"] for params in llm_calls[-1]["params"]] == [43, 47]
    assert llm_calls[-1]["lora_request"] is request

    requests = [object(), object()]
    assert inference.generate_with_loras(
        ["p0", "p1"], requests, seed=53
    ) == ["raw-output"] * 2
    assert isinstance(llm_calls[-1]["params"], FakeSamplingParams)
    assert llm_calls[-1]["params"].kwargs["seed"] == 53
    assert llm_calls[-1]["lora_request"] == requests
    assert llm_calls[-1]["tokenization_kwargs"] == {
        "max_length": 3840,
        "truncation": True,
    }

    assert inference.generate_with_loras(
        ["p0", "p1"], requests, seed=[59, 61]
    ) == ["raw-output"] * 2
    assert [params.kwargs["seed"] for params in llm_calls[-1]["params"]] == [59, 61]
    assert llm_calls[-1]["lora_request"] == requests

    with pytest.raises(ValueError, match="seed count must match prompt count"):
        inference.generate(["p0", "p1"], seed=[1])
    with pytest.raises(ValueError, match="seed count must match prompt count"):
        inference.generate_with_lora(["p0", "p1"], request, seed=[1])
    with pytest.raises(ValueError, match="seed count must match prompt count"):
        inference.generate_with_loras(["p0", "p1"], requests, seed=[1])

    inference._llm = None
    inference._tokenizer = None
