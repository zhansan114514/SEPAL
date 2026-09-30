"""Model-specific prompt formatting shared by SFT and preference training."""

from __future__ import annotations

MISTRAL_TRAINING_TEMPLATE_VERSION = "mistral_inst_v1"
MISTRAL_V03_TRAINING_TEMPLATE_VERSION = (
    "mistral_v03_tokenizer_chat_template_v2_dpo_assistant_space"
)
GEMMA2_TRAINING_TEMPLATE_VERSION = "gemma2_tokenizer_chat_template_v1"
QWEN25_TRAINING_TEMPLATE_VERSION = "qwen2_5_tokenizer_chat_template_v1"
QWEN3_TRAINING_TEMPLATE_VERSION = "qwen3_tokenizer_chat_template_v1"
PHI4_TRAINING_TEMPLATE_VERSION = "phi4_tokenizer_chat_template_v1"


def uses_mistral_instruction_template(model_type: str | None) -> bool:
    """Return whether training prompts must use Mistral's ``[INST]`` format."""
    normalized = str(model_type or "").strip().lower().replace("-", "_")
    return "mistral" in normalized and not uses_mistral_v03_chat_template(model_type)


def uses_mistral_v03_chat_template(model_type: str | None) -> bool:
    """Match Mistral-Instruct v0.3, whose released spacing differs from v0.2."""
    normalized = str(model_type or "").strip().lower().replace("-", "_")
    return normalized in {"mistral_v03", "mistral_v0.3", "mistral_v0_3"}


def uses_gemma2_chat_template(model_type: str | None) -> bool:
    """Match Gemma 2 only; do not change Llama, Mistral, or later Gemma families."""
    normalized = str(model_type or "").strip().lower().replace("-", "_")
    return (
        normalized == "gemma2"
        or normalized.startswith("gemma2_")
        or normalized == "gemma_2"
        or normalized.startswith("gemma_2_")
    )


def uses_qwen3_chat_template(model_type: str | None) -> bool:
    """Return whether Qwen 3's tokenizer-owned ChatML template is required."""
    normalized = str(model_type or "").strip().lower().replace("-", "_")
    return normalized == "qwen3" or normalized.startswith("qwen3_")


def uses_qwen25_chat_template(model_type: str | None) -> bool:
    """Match Qwen 2.5 only; keep its training path separate from Qwen 3."""
    normalized = str(model_type or "").strip().lower().replace("-", "_")
    return (
        normalized == "qwen2.5"
        or normalized.startswith("qwen2.5_")
        or normalized == "qwen2_5"
        or normalized.startswith("qwen2_5_")
        or normalized == "qwen25"
        or normalized.startswith("qwen25_")
    )


def uses_phi4_chat_template(model_type: str | None) -> bool:
    """Match Phi-4/Phi-4-mini without changing older raw-prompt families."""
    normalized = str(model_type or "").strip().lower().replace("-", "_")
    return (
        normalized == "phi4"
        or normalized.startswith("phi4_")
        or normalized == "phi_4"
        or normalized.startswith("phi_4_")
    )


def training_prompt_format_version(model_type: str | None) -> str:
    """Return a stable identifier suitable for caches and training provenance."""
    if uses_mistral_v03_chat_template(model_type):
        return MISTRAL_V03_TRAINING_TEMPLATE_VERSION
    if uses_mistral_instruction_template(model_type):
        return MISTRAL_TRAINING_TEMPLATE_VERSION
    if uses_gemma2_chat_template(model_type):
        return GEMMA2_TRAINING_TEMPLATE_VERSION
    if uses_qwen25_chat_template(model_type):
        return QWEN25_TRAINING_TEMPLATE_VERSION
    if uses_qwen3_chat_template(model_type):
        return QWEN3_TRAINING_TEMPLATE_VERSION
    if uses_phi4_chat_template(model_type):
        return PHI4_TRAINING_TEMPLATE_VERSION
    return "raw_prompt_v1"


def format_training_prompt(
    prompt: str,
    *,
    model_type: str | None,
    tokenizer=None,
) -> str:
    """Format one raw instruction exactly as the corresponding model sees it.

    Mistral-Instruct uses ``[INST] ... [/INST]`` here; its tokenizer adds the
    single BOS token. Gemma 2 uses tokenizer-owned ``<start_of_turn>`` control
    tokens. Qwen 2.5 and Qwen 3 each use their tokenizer-owned ChatML controls
    so SFT/DPO sees exactly the same prompt shape as inference. Their branches
    remain deliberately separate, and other model families preserve their
    existing raw-prompt behavior.
    """
    original = str(prompt or "")
    raw = original.strip()
    if uses_mistral_v03_chat_template(model_type):
        # Tokenizer V3 preserves user-message whitespace.  This matters for
        # prompts that deliberately end in a separator such as
        # ``"My Response: "``; trimming it changes the token immediately
        # before ``[/INST]``.
        v03_prompt = original
        normalized_v03_prompt = v03_prompt.strip()
        if normalized_v03_prompt.startswith("[INST]") and normalized_v03_prompt.endswith(
            "[/INST]"
        ):
            return normalized_v03_prompt
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise ValueError(
                "Mistral v0.3 training prompt formatting requires its tokenizer chat template"
            )
        rendered = str(
            apply_chat_template(
                [{"role": "user", "content": v03_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        bos_token = str(getattr(tokenizer, "bos_token", "") or "")
        while bos_token and rendered.lstrip().startswith(bos_token):
            rendered = rendered.lstrip()[len(bos_token) :]
        return rendered.lstrip()
    if uses_mistral_instruction_template(model_type):
        if raw.startswith("[INST]") and raw.endswith("[/INST]"):
            return raw
        return f"[INST] {raw} [/INST]"
    if uses_gemma2_chat_template(model_type):
        if "<start_of_turn>user\n" in raw and "<start_of_turn>model\n" in raw:
            return raw
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise ValueError(
                "Gemma 2 training prompt formatting requires its tokenizer chat template"
            )
        return str(
            apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    if uses_qwen25_chat_template(model_type) or uses_qwen3_chat_template(model_type):
        if "<|im_start|>user\n" in raw and "<|im_start|>assistant" in raw:
            return original
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            family = "Qwen 2.5" if uses_qwen25_chat_template(model_type) else "Qwen 3"
            raise ValueError(
                f"{family} training prompt formatting requires its tokenizer chat template"
            )
        return str(
            apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    if uses_phi4_chat_template(model_type):
        if "<|user|>" in raw and "<|assistant|>" in raw:
            return original
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise ValueError(
                "Phi-4 training prompt formatting requires its tokenizer chat template"
            )
        return str(
            apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return original


def format_preference_example(
    example: dict[str, object],
    *,
    model_type: str | None,
    tokenizer=None,
) -> dict[str, str]:
    """Return model-specific DPO text fields used by Hugging Face ``Dataset.map``.

    TRL tokenizes plain-text preference rows by concatenating ``prompt`` with
    each completion.  Mistral v0.3's V3 template places exactly one space
    between ``[/INST]`` and assistant content, so that separator has to live
    at the start of each completion.  Other model families keep their existing
    prompt-only update.
    """
    update = {
        "prompt": format_training_prompt(
            str(example.get("prompt") or ""),
            model_type=model_type,
            tokenizer=tokenizer,
        )
    }
    if uses_mistral_v03_chat_template(model_type):
        for field in ("chosen", "rejected"):
            if field not in example:
                continue
            completion = str(example.get(field) or "").strip()
            update[field] = f" {completion}" if completion else completion
    return update
