from __future__ import annotations

from src.training.chat_format import (
    GEMMA2_TRAINING_TEMPLATE_VERSION,
    MISTRAL_TRAINING_TEMPLATE_VERSION,
    MISTRAL_V03_TRAINING_TEMPLATE_VERSION,
    PHI4_TRAINING_TEMPLATE_VERSION,
    QWEN25_TRAINING_TEMPLATE_VERSION,
    QWEN3_TRAINING_TEMPLATE_VERSION,
    format_preference_example,
    format_training_prompt,
    training_prompt_format_version,
    uses_gemma2_chat_template,
    uses_mistral_instruction_template,
    uses_mistral_v03_chat_template,
    uses_phi4_chat_template,
    uses_qwen25_chat_template,
    uses_qwen3_chat_template,
)


class FakeGemmaTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        content = messages[0]["content"]
        return (
            f"<bos><start_of_turn>user\n{content}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )


class FakeQwenTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        content = messages[0]["content"]
        return (
            f"<|im_start|>user\n{content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


class FakePhiTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        content = messages[0]["content"]
        return f"<|user|>{content}<|end|><|assistant|>"


class FakeMistralV03Tokenizer:
    bos_token = "<s>"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<s>[INST] {messages[0]['content']}[/INST]"


def test_mistral_training_prompt_matches_released_inst_shape() -> None:
    assert format_training_prompt("Solve this.", model_type="mistral") == (
        "[INST] Solve this. [/INST]"
    )
    assert training_prompt_format_version("mistral_7b") == (
        MISTRAL_TRAINING_TEMPLATE_VERSION
    )
    assert uses_mistral_instruction_template("mistral") is True
    assert uses_mistral_instruction_template("gemma2") is False
    assert uses_mistral_instruction_template("llama3") is False


def test_mistral_training_prompt_format_is_idempotent() -> None:
    formatted = "[INST] Solve this. [/INST]"
    assert format_training_prompt(formatted, model_type="Mistral-7B") == formatted


def test_mistral_v03_training_uses_native_spacing_without_rendered_bos() -> None:
    tokenizer = FakeMistralV03Tokenizer()
    expected = "[INST] Solve this.[/INST]"

    assert format_training_prompt(
        "Solve this.",
        model_type="mistral_v03",
        tokenizer=tokenizer,
    ) == expected
    assert format_training_prompt(
        expected,
        model_type="mistral_v03",
        tokenizer=tokenizer,
    ) == expected
    assert format_training_prompt(
        " Solve this. ",
        model_type="mistral_v03",
        tokenizer=tokenizer,
    ) == "[INST]  Solve this. [/INST]"
    assert training_prompt_format_version("mistral_v03") == (
        MISTRAL_V03_TRAINING_TEMPLATE_VERSION
    )
    assert uses_mistral_v03_chat_template("mistral-v03") is True
    assert uses_mistral_instruction_template("mistral_v03") is False


def test_non_mistral_training_prompt_is_unchanged() -> None:
    prompt = "  Preserve whitespace.  "
    assert format_training_prompt(prompt, model_type="llama3") == prompt
    assert training_prompt_format_version("llama3") == "raw_prompt_v1"
    assert training_prompt_format_version("gemma3") == "raw_prompt_v1"


def test_gemma2_uses_its_tokenizer_chat_template_only() -> None:
    tokenizer = FakeGemmaTokenizer()
    expected = (
        "<bos><start_of_turn>user\nQuestion<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    assert format_training_prompt(
        "Question",
        model_type="gemma2",
        tokenizer=tokenizer,
    ) == expected
    assert training_prompt_format_version("gemma-2") == (
        GEMMA2_TRAINING_TEMPLATE_VERSION
    )
    assert uses_gemma2_chat_template("gemma2") is True
    assert uses_gemma2_chat_template("mistral") is False
    assert uses_gemma2_chat_template("llama3") is False
    assert uses_gemma2_chat_template("gemma3") is False


def test_gemma2_requires_a_real_chat_template_tokenizer() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires its tokenizer chat template"):
        format_training_prompt("Question", model_type="gemma2")


def test_qwen3_training_uses_native_chat_template_and_is_idempotent() -> None:
    tokenizer = FakeQwenTokenizer()
    expected = (
        "<|im_start|>user\nQuestion<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    assert format_training_prompt(
        "Question",
        model_type="qwen3",
        tokenizer=tokenizer,
    ) == expected
    assert format_training_prompt(
        expected,
        model_type="qwen3",
        tokenizer=tokenizer,
    ) == expected
    assert training_prompt_format_version("qwen3") == QWEN3_TRAINING_TEMPLATE_VERSION
    assert uses_qwen3_chat_template("qwen3") is True
    assert uses_qwen3_chat_template("llama3") is False


def test_qwen25_training_uses_native_chat_template_and_is_isolated() -> None:
    tokenizer = FakeQwenTokenizer()
    expected = (
        "<|im_start|>user\nQuestion<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    assert format_training_prompt(
        "Question",
        model_type="qwen2.5",
        tokenizer=tokenizer,
    ) == expected
    assert format_training_prompt(
        expected,
        model_type="qwen2_5_3b",
        tokenizer=tokenizer,
    ) == expected
    assert training_prompt_format_version("qwen2.5") == QWEN25_TRAINING_TEMPLATE_VERSION
    assert uses_qwen25_chat_template("qwen2.5") is True
    assert uses_qwen25_chat_template("qwen2_5_3b") is True
    assert uses_qwen25_chat_template("qwen3") is False
    assert uses_qwen3_chat_template("qwen2.5") is False


def test_qwen3_requires_a_real_chat_template_tokenizer() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires its tokenizer chat template"):
        format_training_prompt("Question", model_type="qwen3")


def test_qwen25_requires_a_real_chat_template_tokenizer() -> None:
    import pytest

    with pytest.raises(ValueError, match="Qwen 2.5.*requires its tokenizer chat template"):
        format_training_prompt("Question", model_type="qwen2.5")


def test_phi4_training_uses_native_chat_template_and_phi3_lora_targets() -> None:
    from src.training.lora_config import MODEL_TARGET_MODULES

    tokenizer = FakePhiTokenizer()
    expected = "<|user|>Question<|end|><|assistant|>"

    assert format_training_prompt(
        "Question",
        model_type="phi4",
        tokenizer=tokenizer,
    ) == expected
    assert format_training_prompt(
        expected,
        model_type="phi_4_mini",
        tokenizer=tokenizer,
    ) == expected
    assert training_prompt_format_version("phi4") == PHI4_TRAINING_TEMPLATE_VERSION
    assert uses_phi4_chat_template("phi-4-mini") is True
    assert uses_phi4_chat_template("mistral") is False
    assert MODEL_TARGET_MODULES["phi4"] == [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]


def test_phi4_requires_a_real_chat_template_tokenizer() -> None:
    import pytest

    with pytest.raises(ValueError, match="Phi-4.*requires its tokenizer chat template"):
        format_training_prompt("Question", model_type="phi4")


def test_preference_formatting_changes_only_prompt() -> None:
    example = {"prompt": "Question", "chosen": "A", "rejected": "B"}
    assert format_preference_example(example, model_type="mistral") == {
        "prompt": "[INST] Question [/INST]"
    }


def test_mistral_v03_dpo_adds_native_assistant_separator() -> None:
    example = {
        "prompt": "Question",
        "chosen": "  Answer A  ",
        "rejected": "\nAnswer B\n",
    }

    assert format_preference_example(
        example,
        model_type="mistral_v03",
        tokenizer=FakeMistralV03Tokenizer(),
    ) == {
        "prompt": "[INST] Question[/INST]",
        "chosen": " Answer A",
        "rejected": " Answer B",
    }
    assert "v2_dpo_assistant_space" in MISTRAL_V03_TRAINING_TEMPLATE_VERSION


def test_dpo_dataset_applies_mistral_v03_assistant_separator() -> None:
    from src.training._dpo_runner import _format_preference_dataset

    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        def map(self, function, *, desc):
            assert MISTRAL_V03_TRAINING_TEMPLATE_VERSION in desc
            return FakeDataset([{**row, **function(row)} for row in self.rows])

    original = FakeDataset([
        {"prompt": "Question", "chosen": "A", "rejected": "B"}
    ])
    formatted = _format_preference_dataset(
        original,
        model_type="mistral_v03",
        tokenizer=FakeMistralV03Tokenizer(),
    )

    assert formatted.rows == [
        {
            "prompt": "[INST] Question[/INST]",
            "chosen": " A",
            "rejected": " B",
        }
    ]


def test_dpo_dataset_formats_mistral_prompts_without_changing_completions() -> None:
    from src.training._dpo_runner import _format_preference_dataset

    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        def map(self, function, *, desc):
            assert "mistral_inst_v1" in desc
            return FakeDataset([{**row, **function(row)} for row in self.rows])

    original = FakeDataset([
        {"prompt": "Question", "chosen": "A", "rejected": "B"}
    ])
    formatted = _format_preference_dataset(original, model_type="mistral")

    assert formatted.rows == [
        {
            "prompt": "[INST] Question [/INST]",
            "chosen": "A",
            "rejected": "B",
        }
    ]
    assert _format_preference_dataset(original, model_type="llama3") is original


def test_dpo_dataset_uses_gemma2_tokenizer_chat_template() -> None:
    from src.training._dpo_runner import _format_preference_dataset

    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        def map(self, function, *, desc):
            assert GEMMA2_TRAINING_TEMPLATE_VERSION in desc
            return FakeDataset([{**row, **function(row)} for row in self.rows])

    dataset = FakeDataset([
        {"prompt": "Question", "chosen": "A", "rejected": "B"}
    ])
    formatted = _format_preference_dataset(
        dataset,
        model_type="gemma2",
        tokenizer=FakeGemmaTokenizer(),
    )

    assert formatted.rows[0] == {
        "prompt": (
            "<bos><start_of_turn>user\nQuestion<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
        "chosen": "A",
        "rejected": "B",
    }


def test_dpo_dataset_uses_qwen25_tokenizer_chat_template() -> None:
    from src.training._dpo_runner import _format_preference_dataset

    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        def map(self, function, *, desc):
            assert QWEN25_TRAINING_TEMPLATE_VERSION in desc
            return FakeDataset([{**row, **function(row)} for row in self.rows])

    dataset = FakeDataset([
        {"prompt": "Question", "chosen": "A", "rejected": "B"}
    ])
    formatted = _format_preference_dataset(
        dataset,
        model_type="qwen2.5",
        tokenizer=FakeQwenTokenizer(),
    )

    assert formatted.rows[0] == {
        "prompt": (
            "<|im_start|>user\nQuestion<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "chosen": "A",
        "rejected": "B",
    }
