"""LoRA inference adapters for the paired pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LoRAError(RuntimeError):
    """Raised when a LoRA adapter cannot be loaded."""


_LORA_ID_COUNTER = 0
_LORA_ID_CACHE: dict[str, int] = {}


def resolve_adapter_path(path: str) -> str:
    p = Path(path)
    if (p / "adapter_config.json").exists():
        return str(p)
    suffixed = Path(str(p) + "_adapter")
    if (suffixed / "adapter_config.json").exists():
        return str(suffixed)
    raise LoRAError(
        f"LoRA adapter not found. Tried {p}/adapter_config.json and "
        f"{suffixed}/adapter_config.json."
    )


def _stable_lora_id(lora_path: str) -> int:
    global _LORA_ID_COUNTER
    resolved = resolve_adapter_path(lora_path)
    if resolved not in _LORA_ID_CACHE:
        _LORA_ID_COUNTER += 1
        _LORA_ID_CACHE[resolved] = _LORA_ID_COUNTER
    return _LORA_ID_CACHE[resolved]


def load_lora_request(engine: Any, lora_path: str) -> Any:
    resolved = resolve_adapter_path(lora_path)
    supports = getattr(engine, "supports_lora", None)
    if supports is False:
        raise LoRAError("Engine was created without LoRA support.")
    try:
        try:
            from vllm.lora.request import LoRARequest
        except ImportError:
            from vllm import LoRARequest
        return LoRARequest(Path(resolved).name, _stable_lora_id(resolved), resolved)
    except Exception as e:  # pragma: no cover - depends on vLLM version
        raise LoRAError(f"Failed to create LoRARequest for {resolved}: {e}") from e


class LoRAModelAdapter:
    """Small generation wrapper around VLLMInference and one LoRA request."""

    def __init__(self, engine: Any, lora_request: Any = None):
        self.engine = engine
        self.lora_request = lora_request

    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 0.9,
        enable_thinking: bool | None = None,
    ) -> list[str]:
        if self.lora_request is None:
            return self.engine.generate(
                prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )
        if hasattr(self.engine, "generate_with_lora"):
            return self.engine.generate_with_lora(
                prompts,
                self.lora_request,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )
        if hasattr(self.engine, "_llm") and self.engine._llm is not None:
            from vllm import SamplingParams
            params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            if hasattr(self.engine, "_format_prompts_for_thinking"):
                prompts = self.engine._format_prompts_for_thinking(prompts, enable_thinking)
            outputs = self.engine._llm.generate(prompts, params, lora_request=self.lora_request)
            return [choice.text for out in outputs for choice in out.outputs]
        raise LoRAError("Engine does not support LoRA generation.")

    def generate_single(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        enable_thinking: bool | None = None,
    ) -> str:
        result = self.generate(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        return result[0] if result else ""


def build_lora_adapters(engine: Any, lora_paths: dict[str, str]) -> dict[str, LoRAModelAdapter]:
    adapters: dict[str, LoRAModelAdapter] = {}
    for name, path in lora_paths.items():
        adapters[name] = LoRAModelAdapter(engine, load_lora_request(engine, path))
        logger.info("Loaded LoRA adapter for %s: %s", name, path)
    return adapters


def generate_lora_batches(
    adapters: dict[str, LoRAModelAdapter],
    prompts_by_name: dict[str, list[str]],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float = 0.9,
    enable_thinking: bool | None = None,
) -> dict[str, list[str]]:
    """Generate one vLLM batch with a per-prompt LoRA request."""
    outputs_by_name = {name: [] for name in prompts_by_name}
    flat_prompts: list[str] = []
    flat_loras: list[Any] = []
    flat_keys: list[str] = []

    for name, prompts in prompts_by_name.items():
        if not prompts:
            continue
        adapter = adapters[name]
        flat_prompts.extend(prompts)
        flat_loras.extend([adapter.lora_request] * len(prompts))
        flat_keys.extend([name] * len(prompts))

    if not flat_prompts:
        return outputs_by_name

    engine = next(iter(adapters.values())).engine
    if not hasattr(engine, "generate_with_loras"):
        raise LoRAError("Engine does not support batched per-prompt LoRA generation.")
    flat_outputs = engine.generate_with_loras(
        flat_prompts,
        flat_loras,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
    )
    if len(flat_outputs) != len(flat_keys):
        raise LoRAError(
            "vLLM returned an unexpected number of outputs: "
            f"{len(flat_outputs)} != {len(flat_keys)}"
        )

    for name, output in zip(flat_keys, flat_outputs):
        outputs_by_name[name].append(output)
    return outputs_by_name
