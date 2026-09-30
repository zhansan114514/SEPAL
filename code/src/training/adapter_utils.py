"""Generic LoRA adapter-path resolution shared by training runners."""

from __future__ import annotations

from pathlib import Path


class AdapterPathError(FileNotFoundError):
    """Raised when a requested LoRA adapter cannot be resolved."""


def resolve_lora_adapter_path(path: str | Path) -> str:
    """Resolve either an adapter directory or its ``_adapter`` training suffix.

    Training runners intentionally validate the PEFT config rather than a pipeline-specific
    success marker. Pipeline entry points remain responsible for requiring durable ``_SUCCESS``
    markers before they pass an adapter to this generic layer.
    """
    target = Path(path)
    candidates = [target, Path(str(target) + "_adapter")]
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file():
            return str(candidate)
    tried = ", ".join(str(candidate / "adapter_config.json") for candidate in candidates)
    raise AdapterPathError(f"LoRA adapter not found; tried: {tried}")
