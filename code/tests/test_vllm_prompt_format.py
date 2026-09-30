from __future__ import annotations

import pytest

from src.inference.vllm_server import (
    DEFAULT_INFERENCE_TEMPLATE_VERSION,
    GEMMA2_INFERENCE_TEMPLATE_VERSION,
    MISTRAL_INFERENCE_TEMPLATE_VERSION,
    MISTRAL_V03_INFERENCE_TEMPLATE_VERSION,
    PHI4_INFERENCE_TEMPLATE_VERSION,
    QWEN25_INFERENCE_TEMPLATE_VERSION,
    QWEN3_INSTRUCT_2507_TEMPLATE_VERSION,
    VLLMInference,
    inference_prompt_format_version,
)


class FakeMistralTokenizer:
    bos_token = "<s>"
    bos_token_id = 1

    def __init__(self) -> None:
        self.chat_template_calls = 0

    def apply_chat_template(self, *args, **kwargs):
        self.chat_template_calls += 1
        return "<s> [INST] legacy double-BOS path [/INST]"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids = [1] if text.lstrip().startswith(self.bos_token) else [17]
        return ([1] if add_special_tokens else []) + ids


class FakeGemma2Tokenizer:
    bos_token = "<bos>"
    bos_token_id = 2

    def __init__(self) -> None:
        self.chat_template_calls = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.chat_template_calls += 1
        content = messages[0]["content"]
        return (
            f"<bos><start_of_turn>user\n{content}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids = [2] if text.lstrip().startswith(self.bos_token) else [29]
        return ([2] if add_special_tokens else []) + ids


class FakeQwenTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.calls += 1
        return (
            f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


class FakePhiTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.calls += 1
        return f"<|user|>{messages[0]['content']}<|end|><|assistant|>"


class FakeMistralV03Tokenizer:
    bos_token = "<s>"
    bos_token_id = 1
    eos_token_id = 2
    truncation_side = "left"

    def __init__(self) -> None:
        self.calls = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert add_generation_prompt is True
        self.calls += 1
        if tokenize:
            content_ids = [100 + ord(char) % 50 for char in messages[0]["content"]]
            return [1, 3, *content_ids, 4]
        raise AssertionError("Mistral v0.3 must not render an unsafe text template")

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids = [1] if text.lstrip().startswith(self.bos_token) else [17]
        return ([1] if add_special_tokens else []) + ids


def _engine(model_name: str, model_type: str | None) -> VLLMInference:
    engine = VLLMInference(model_name, model_type=model_type)
    engine._tokenizer = FakeMistralTokenizer()
    return engine


def test_mistral_inference_uses_official_bos_free_inst_wrapper() -> None:
    engine = _engine("/models/local-snapshot", "mistral")

    formatted = engine._format_prompts_for_thinking(["Solve this."], enable_thinking=False)

    assert formatted == ["[INST] Solve this. [/INST]"]
    assert engine._tokenizer.chat_template_calls == 0
    assert engine._tokenizer.encode(formatted[0], add_special_tokens=True)[0:2] == [1, 17]
    assert engine._mistral_prompt_format_audited is True


def test_mistral_inference_removes_legacy_rendered_bos_and_is_idempotent() -> None:
    engine = _engine("/models/local-snapshot", "Mistral-7B")

    formatted = engine._format_prompts_for_thinking(
        ["<s> <s> [INST] Solve this. [/INST]"],
        enable_thinking=None,
    )

    assert formatted == ["[INST] Solve this. [/INST]"]


def test_mistral_model_name_is_used_only_when_explicit_type_is_absent() -> None:
    inferred = _engine("/models/Mistral-7B-Instruct-v0.2", None)
    explicit_other = _engine("/models/Mistral-7B-Instruct-v0.2", "llama3")

    assert inferred._format_prompts_for_thinking(["Q"], None) == ["[INST] Q [/INST]"]
    assert explicit_other._format_prompts_for_thinking(["Q"], None) == ["Q"]


def test_mistral_single_bos_audit_rejects_rendered_bos() -> None:
    engine = _engine("/models/local-snapshot", "mistral")

    with pytest.raises(RuntimeError, match="must not contain a rendered BOS"):
        engine._audit_mistral_single_bos("<s> [INST] Q [/INST]")


def test_mistral_v03_passes_native_token_ids_to_vllm_without_double_bos() -> None:
    engine = VLLMInference(
        "/models/Mistral-7B-Instruct-v0.3",
        model_type="mistral_v03",
    )
    engine._tokenizer = FakeMistralV03Tokenizer()

    prompts, kwargs = engine._prepare_generation_prompts(
        ["Question"], max_tokens=256, enable_thinking=False
    )

    assert prompts == [
        {"prompt_token_ids": [1, 3, 131, 117, 101, 115, 116, 105, 111, 110, 4]}
    ]
    assert kwargs == {}
    assert engine._tokenizer.calls == 1
    assert engine._mistral_v03_prompt_format_audited is True
    assert engine._format_prompts_for_thinking(["[INST] Question[/INST]"], False) == [
        "Question"
    ]


def test_mistral_v03_preserves_raw_user_whitespace() -> None:
    engine = VLLMInference(
        "/models/Mistral-7B-Instruct-v0.3",
        model_type="mistral_v03",
    )
    engine._tokenizer = FakeMistralV03Tokenizer()

    prompts, _ = engine._prepare_generation_prompts(
        [" Question "], max_tokens=256, enable_thinking=False
    )

    expected_content = [100 + ord(char) % 50 for char in " Question "]
    assert prompts == [{"prompt_token_ids": [1, 3, *expected_content, 4]}]


def test_mistral_v03_truncates_token_ids_while_preserving_control_prefix() -> None:
    engine = VLLMInference(
        "/models/Mistral-7B-Instruct-v0.3",
        model_type="mistral_v03",
        max_model_len=8,
    )
    engine._tokenizer = FakeMistralV03Tokenizer()

    prompts, _ = engine._prepare_generation_prompts(
        ["abcdefghij"], max_tokens=2, enable_thinking=False
    )

    full = [1, 3, *[100 + ord(char) % 50 for char in "abcdefghij"], 4]
    assert prompts == [{"prompt_token_ids": full[:2] + full[-4:]}]


def test_read_only_left_truncation_side_is_already_compatible() -> None:
    class ReadOnlyLeftTokenizer:
        @property
        def truncation_side(self) -> str:
            return "left"

    VLLMInference._configure_left_truncation(ReadOnlyLeftTokenizer())


def test_read_only_non_left_truncation_side_fails_clearly() -> None:
    class ReadOnlyRightTokenizer:
        @property
        def truncation_side(self) -> str:
            return "right"

    with pytest.raises(RuntimeError, match="required left-side truncation"):
        VLLMInference._configure_left_truncation(ReadOnlyRightTokenizer())


def test_non_mistral_prompt_path_is_unchanged() -> None:
    engine = _engine("/models/local-snapshot", "llama3")

    assert engine._format_prompts_for_thinking(["Q"], None) == ["Q"]
    assert engine._tokenizer.chat_template_calls == 0


def test_qwen3_instruct_2507_uses_native_non_thinking_template() -> None:
    engine = VLLMInference(
        "/models/Qwen3-4B-Instruct-2507",
        model_type="qwen3",
    )
    engine._tokenizer = FakeQwenTokenizer()

    formatted = engine._format_prompts_for_thinking(["Question"], False)

    assert formatted == [
        "<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n"
    ]
    assert engine._tokenizer.calls == 1


def test_qwen25_uses_native_non_thinking_template() -> None:
    engine = VLLMInference(
        "/models/Qwen2.5-3B-Instruct",
        model_type="qwen2.5",
    )
    engine._tokenizer = FakeQwenTokenizer()

    formatted = engine._format_prompts_for_thinking(["Question"], False)

    assert formatted == [
        "<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n"
    ]
    assert engine._tokenizer.calls == 1
    assert engine._format_prompts_for_thinking(formatted, False) == formatted


def test_phi4_uses_native_tokenizer_template_and_is_idempotent() -> None:
    engine = VLLMInference(
        "/models/Phi-4-mini-instruct",
        model_type="phi4",
    )
    engine._tokenizer = FakePhiTokenizer()

    formatted = engine._format_prompts_for_thinking(["Question"], False)

    assert formatted == ["<|user|>Question<|end|><|assistant|>"]
    assert engine._tokenizer.calls == 1
    assert engine._format_prompts_for_thinking(formatted, False) == formatted


def test_generation_reserves_context_for_output_and_rejects_impossible_budget() -> None:
    engine = VLLMInference(
        "/models/Phi-4-mini-instruct",
        model_type="phi4",
        max_model_len=4096,
    )

    assert engine._generation_tokenization_kwargs(1024) == {
        "max_length": 3072,
        "truncation": True,
    }
    with pytest.raises(ValueError, match="leave room for at least one input token"):
        engine._generation_tokenization_kwargs(4096)


def test_gemma2_inference_keeps_chat_controls_but_removes_rendered_bos() -> None:
    engine = VLLMInference("/models/local-snapshot", model_type="gemma2")
    engine._tokenizer = FakeGemma2Tokenizer()

    formatted = engine._format_prompts_for_thinking(["Solve this."], False)

    assert formatted == [
        "<start_of_turn>user\nSolve this.<end_of_turn>\n<start_of_turn>model\n"
    ]
    assert engine._tokenizer.chat_template_calls == 1
    assert engine._tokenizer.encode(formatted[0], add_special_tokens=True)[:2] == [2, 29]
    assert engine._gemma2_prompt_format_audited is True


def test_gemma2_inference_format_is_idempotent() -> None:
    engine = VLLMInference("/models/local-snapshot", model_type="gemma-2")
    engine._tokenizer = FakeGemma2Tokenizer()
    rendered = (
        "<bos><start_of_turn>user\nQ<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    assert engine._format_prompts_for_thinking([rendered], None) == [
        "<start_of_turn>user\nQ<end_of_turn>\n<start_of_turn>model\n"
    ]
    assert engine._tokenizer.chat_template_calls == 0


def test_inference_prompt_format_version_respects_explicit_model_type() -> None:
    assert inference_prompt_format_version("mistral", "/models/local") == (
        MISTRAL_INFERENCE_TEMPLATE_VERSION
    )
    assert inference_prompt_format_version(None, "/models/Mistral-7B") == (
        MISTRAL_INFERENCE_TEMPLATE_VERSION
    )
    assert inference_prompt_format_version(
        "mistral_v03", "/models/Mistral-7B-Instruct-v0.3"
    ) == MISTRAL_V03_INFERENCE_TEMPLATE_VERSION
    assert inference_prompt_format_version(
        None, "/models/Mistral-7B-Instruct-v0.3"
    ) == MISTRAL_V03_INFERENCE_TEMPLATE_VERSION
    assert inference_prompt_format_version("llama3", "/models/Mistral-7B") == (
        DEFAULT_INFERENCE_TEMPLATE_VERSION
    )
    assert inference_prompt_format_version("gemma2", "/models/local") == (
        GEMMA2_INFERENCE_TEMPLATE_VERSION
    )
    assert inference_prompt_format_version(None, "/models/gemma-2-2b-it") == (
        GEMMA2_INFERENCE_TEMPLATE_VERSION
    )
    assert inference_prompt_format_version("llama3", "/models/gemma-2-2b-it") == (
        DEFAULT_INFERENCE_TEMPLATE_VERSION
    )
    assert inference_prompt_format_version(
        "qwen3", "/models/Qwen3-4B-Instruct-2507"
    ) == QWEN3_INSTRUCT_2507_TEMPLATE_VERSION
    assert inference_prompt_format_version(
        "qwen2.5", "/models/Qwen2.5-3B-Instruct"
    ) == QWEN25_INFERENCE_TEMPLATE_VERSION
    assert inference_prompt_format_version(
        None, "/models/Qwen2.5-3B-Instruct"
    ) == QWEN25_INFERENCE_TEMPLATE_VERSION
    assert inference_prompt_format_version(
        "llama3", "/models/Qwen2.5-3B-Instruct"
    ) == DEFAULT_INFERENCE_TEMPLATE_VERSION
    assert inference_prompt_format_version(
        "phi4", "/models/local"
    ) == PHI4_INFERENCE_TEMPLATE_VERSION
    assert inference_prompt_format_version(
        None, "/models/Phi-4-mini-instruct"
    ) == PHI4_INFERENCE_TEMPLATE_VERSION
    assert inference_prompt_format_version(
        "llama3", "/models/Phi-4-mini-instruct"
    ) == DEFAULT_INFERENCE_TEMPLATE_VERSION
