"""Role policies and LoRA loading for the paper-original two-agent team."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.acccollab.config import ACCCollabConfig
from src.inference.vllm_server import build_inference_engine
from src.utils.artifacts import completed_adapter_path

logger = logging.getLogger(__name__)


class ACCCollabPolicyError(RuntimeError):
    """Raised when an Actor/Critic policy cannot be constructed."""


def resolve_adapter_path(path: str | Path, *, require_success: bool = True) -> str:
    """Resolve a direct or ``_adapter`` LoRA directory.

    Experiment stages require a durable ``_SUCCESS`` marker by default. Tests
    and low-level utilities may disable that check when inspecting a raw LoRA.
    """
    if require_success:
        completed = completed_adapter_path(path)
        if completed:
            return completed
        raise ACCCollabPolicyError(f"No completed LoRA adapter found for {path}")

    target = Path(path)
    candidates = [target, Path(str(target) + "_adapter")]
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file():
            return str(candidate)
    raise ACCCollabPolicyError(f"LoRA adapter_config.json not found for {path}")


def _lora_id(path: str) -> int:
    """Return a stable positive vLLM LoRA id for one resolved path."""
    digest = hashlib.sha256(str(Path(path).resolve()).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 2_000_000_000 + 1


def load_lora_request(path: str | Path) -> Any:
    """Create a vLLM ``LoRARequest`` without importing the paired pipeline."""
    resolved = resolve_adapter_path(path)
    try:
        try:
            from vllm.lora.request import LoRARequest
        except ImportError:  # pragma: no cover - vLLM-version dependent
            from vllm import LoRARequest
        return LoRARequest(Path(resolved).name, _lora_id(resolved), resolved)
    except Exception as exc:  # pragma: no cover - requires vLLM runtime
        raise ACCCollabPolicyError(
            f"Failed to create LoRARequest for {resolved}: {exc}"
        ) from exc


class RolePolicy:
    """Generation facade for either the Actor or Critic role."""

    def __init__(self, engine: Any, *, role: str, lora_request: Any | None = None):
        if role not in {"actor", "critic"}:
            raise ValueError(f"role must be actor or critic, got {role!r}")
        self.engine = engine
        self.role = role
        self.lora_request = lora_request

    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool | None,
        seed: int | Sequence[int] | None = None,
    ) -> list[str]:
        """Generate one completion per prompt with this role's policy."""
        kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "enable_thinking": enable_thinking,
        }
        if seed is not None:
            kwargs["seed"] = (
                [int(item) for item in seed]
                if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes, bytearray))
                else int(seed)
            )
        if self.lora_request is None:
            return list(self.engine.generate(prompts, **kwargs))
        if not hasattr(self.engine, "generate_with_lora"):
            raise ACCCollabPolicyError("Inference engine has no LoRA generation API")
        return list(
            self.engine.generate_with_lora(
                prompts,
                self.lora_request,
                **kwargs,
            )
        )


@dataclass
class PolicyBundle:
    """One shared base engine with independently selectable role adapters."""

    engine: Any
    actor: RolePolicy
    critic: RolePolicy
    actor_adapter: str | None
    critic_adapter: str | None

    def generation_stats(self) -> dict[str, Any]:
        getter = getattr(self.engine, "generation_stats", None)
        return getter() if callable(getter) else {"schema_version": 1, "by_max_tokens": {}}

    def cleanup(self) -> None:
        cleanup = getattr(self.engine, "cleanup", None)
        if callable(cleanup):
            cleanup()

    def __enter__(self) -> "PolicyBundle":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_type, exc_value, traceback
        self.cleanup()
        return False


def build_policy_bundle(
    config: ACCCollabConfig,
    *,
    actor_adapter: str | Path | None,
    critic_adapter: str | Path | None,
    device: int | None = None,
) -> PolicyBundle:
    """Build the two policies through the repository's shared vLLM factory."""
    resolved_actor = resolve_adapter_path(actor_adapter) if actor_adapter else None
    resolved_critic = resolve_adapter_path(critic_adapter) if critic_adapter else None
    adapter_paths = {path for path in (resolved_actor, resolved_critic) if path}
    enable_lora = bool(adapter_paths)
    engine = build_inference_engine(
        config.inference_args(device),
        enable_lora=enable_lora,
        max_loras=max(1, len(adapter_paths)) if enable_lora else None,
        max_lora_rank=config.training.lora.r if enable_lora else None,
        disable_lora_cudagraph=config.runtime.disable_lora_cudagraph,
    )
    actor_request = load_lora_request(resolved_actor) if resolved_actor else None
    critic_request = load_lora_request(resolved_critic) if resolved_critic else None
    logger.info(
        "Built ACC-Collab policies (actor=%s, critic=%s)",
        resolved_actor or "base",
        resolved_critic or "base",
    )
    return PolicyBundle(
        engine=engine,
        actor=RolePolicy(engine, role="actor", lora_request=actor_request),
        critic=RolePolicy(engine, role="critic", lora_request=critic_request),
        actor_adapter=resolved_actor,
        critic_adapter=resolved_critic,
    )
