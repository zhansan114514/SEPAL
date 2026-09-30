"""Strict configuration schema for the original two-agent ACC-Collab reproduction."""

from __future__ import annotations

import argparse
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ACCCollabConfigError(ValueError):
    """Raised when an ACC-Collab config violates the public schema or paper invariants."""


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    name: str
    type: str = "llama3"
    dtype: str = "bfloat16"
    max_model_len: int = 8192
    language_model_only: bool = False


@dataclass(frozen=True)
class MethodConfig:
    """Structural choices fixed by ACC-Collab §3–§4 and Algorithm 1."""

    alternating_iterations: int = 1
    deliberation_rounds: int = 5
    training_order: str = "critic_then_actor"
    reward_estimator: str = "one_step_mc"
    pair_rule: str = "paper_eq5_if_elif"


@dataclass(frozen=True)
class SplitConfig:
    split: str
    samples: int | None = None
    strategy: str = "random"
    expected_samples: int | None = None
    # Optional data-selection seed. When omitted, paper-original configs keep
    # using run.seed; multi-role runs set this once across all roles so their
    # stochastic policies remain distinct while questions stay aligned.
    sampling_seed: int | None = None


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    preference: SplitConfig
    eval: SplitConfig
    mmlu_load_mode: str = "by_subject"
    benchmark_data_dir: str | None = None
    # Independent stochastic trajectories per selected training question.
    # This is separate from evaluation.trials; repeated evaluation remains off.
    preference_trials: int = 1


@dataclass(frozen=True)
class RuntimeBatchConfig:
    preference_generation: int = 4
    evaluation: int = 16


@dataclass(frozen=True)
class RuntimeConfig:
    generation_devices: list[int] = field(default_factory=lambda: [0])
    training_devices: list[int] = field(default_factory=lambda: [0])
    gpu_memory_utilization: float = 0.80
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    disable_lora_cudagraph: bool = True
    gdn_prefill_backend: str | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    batch: RuntimeBatchConfig = field(default_factory=RuntimeBatchConfig)


@dataclass(frozen=True)
class TokenConfig:
    actor: int = 1024
    critic: int = 1024
    dpo_max_length: int = 4096
    dpo_prompt: int = 3072
    dpo_completion: int = 1024


@dataclass(frozen=True)
class ThinkingConfig:
    train: bool = False
    eval: bool = False


@dataclass(frozen=True)
class TruncationConfig:
    warn_rate: float = 0.001
    fail_rate: float = 0.01
    fail_on_excess: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    train_temperature: float = 0.7
    eval_temperature: float = 0.7
    top_p: float = 0.9
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    truncation: TruncationConfig = field(default_factory=TruncationConfig)


@dataclass(frozen=True)
class RewardConfig:
    # The paper only says "multiple" one-step simulations. Ten is taken from
    # the released Critic data generator; it is explicitly an implementation choice.
    rollouts: int = 10
    # The paper does not publish epsilon. 0.6 matches released DPO.py/data.py.
    epsilon: float = 0.6
    min_pairs_per_stage: int = 256


@dataclass(frozen=True)
class LoraConfig:
    # Appendix A explicitly reports LoRA rank 256. Alpha=2*r follows the release.
    r: int = 256
    alpha: int = 512


@dataclass(frozen=True)
class DPOConfig:
    learning_rate: float = 1.41e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    epochs: int = 3
    beta: float = 0.1
    loss_type: str = "sigmoid"
    # Appendix A explicitly gives chosen-completion NLL weight 1.
    nll_weight: float = 1.0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    max_grad_norm: float = 0.3
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = True
    timeout_per_1k: int = 3600
    checkpoint_steps: int = 50
    checkpoint_total_limit: int = 1
    resume_from_checkpoint: bool = True
    resume_optimizer_state: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    lora: LoraConfig = field(default_factory=LoraConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)


@dataclass(frozen=True)
class EvaluationConfig:
    trials: int = 5
    # Optional source run whose final Actor/Critic registry is evaluated.
    policy_output_dir: str | None = None


@dataclass(frozen=True)
class InitializationConfig:
    """Optional policies that replace the shared base at iteration zero.

    The paper-original configs omit this section and therefore remain byte-for-byte
    equivalent to starting both roles from the base model.  The multi-role extension
    uses a role-specific SFT adapter for the Actor only; the Critic deliberately keeps
    the original base-model initialization.
    """

    actor_adapter: str | None = None
    critic_adapter: str | None = None


@dataclass(frozen=True)
class PromptRoleConfig:
    """Optional specialization prefix applied without changing original prompts."""

    name: str = "original"
    actor_instruction: str = ""
    critic_instruction: str = ""
    implementation_version: str = "acccollab_role_prefix_v1"


@dataclass(frozen=True)
class ACCCollabPaths:
    """All outputs for the isolated reproduction, derived from one root."""

    output_dir: Path

    @property
    def marker_dir(self) -> Path:
        return self.output_dir / "markers"

    @property
    def registry_dir(self) -> Path:
        return self.output_dir / "registry"

    @property
    def eval_dir(self) -> Path:
        return self.output_dir / "eval"

    @staticmethod
    def iteration_tag(iteration: int) -> str:
        if iteration < 1:
            raise ValueError(f"iteration must be >= 1, got {iteration}")
        return f"iteration_{iteration:02d}"

    def data_dir(self, iteration: int, agent: str) -> Path:
        if agent not in {"actor", "critic"}:
            raise ValueError(f"agent must be actor or critic, got {agent!r}")
        return self.output_dir / "data" / self.iteration_tag(iteration) / f"{agent}_dpo"

    def training_output(self, iteration: int, agent: str) -> Path:
        if agent not in {"actor", "critic"}:
            raise ValueError(f"agent must be actor or critic, got {agent!r}")
        return self.output_dir / "adapters" / self.iteration_tag(iteration) / agent

    def adapter_dir(self, iteration: int, agent: str) -> Path:
        return Path(str(self.training_output(iteration, agent)) + "_adapter")

    def iteration_registry(self, iteration: int) -> Path:
        return self.registry_dir / f"{self.iteration_tag(iteration)}.json"

    @property
    def final_registry(self) -> Path:
        return self.registry_dir / "final.json"


@dataclass(frozen=True)
class ACCCollabConfig:
    run: RunConfig
    model: ModelConfig
    method: MethodConfig
    data: DataConfig
    runtime: RuntimeConfig
    tokens: TokenConfig
    generation: GenerationConfig
    reward: RewardConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    prompt_role: PromptRoleConfig = field(default_factory=PromptRoleConfig)

    @property
    def paths(self) -> ACCCollabPaths:
        return ACCCollabPaths(Path(self.run.output_dir))

    @property
    def pair_rounds(self) -> int:
        """Algorithm 1 branches at t=1..T after the natural t=0 round."""
        return self.method.deliberation_rounds - 1

    def validate_iteration(self, iteration: int) -> None:
        if not 1 <= int(iteration) <= self.method.alternating_iterations:
            raise ACCCollabConfigError(
                f"iteration must be in [1, {self.method.alternating_iterations}], got {iteration}"
            )

    def actor_adapter_before(self, iteration: int) -> Path | None:
        self.validate_iteration(iteration)
        if iteration == 1:
            initial = self.initialization.actor_adapter
            return Path(initial) if initial else None
        return self.paths.adapter_dir(iteration - 1, "actor")

    def critic_adapter_before(self, iteration: int) -> Path | None:
        self.validate_iteration(iteration)
        if iteration == 1:
            initial = self.initialization.critic_adapter
            return Path(initial) if initial else None
        return self.paths.adapter_dir(iteration - 1, "critic")

    def inference_args(self, device: int | None = None) -> argparse.Namespace:
        selected = (
            int(device)
            if device is not None
            else int(self.runtime.generation_devices[0])
        )
        return argparse.Namespace(
            model_name=self.model.name,
            model_type=self.model.type,
            device=selected,
            dtype=self.model.dtype,
            seed=int(self.run.seed),
            gpu_memory_utilization=float(self.runtime.gpu_memory_utilization),
            max_model_len=int(self.model.max_model_len),
            enable_prefix_caching=bool(self.runtime.enable_prefix_caching),
            max_num_batched_tokens=self.runtime.max_num_batched_tokens,
            max_num_seqs=self.runtime.max_num_seqs,
            enforce_eager=bool(self.runtime.enforce_eager),
            language_model_only=bool(self.model.language_model_only),
            gdn_prefill_backend=self.runtime.gdn_prefill_backend,
        )

    def validate(self) -> None:
        """Validate both generic ranges and the paper-original structural invariants."""
        if self.method.alternating_iterations not in {1, 2}:
            raise ACCCollabConfigError(
                "method.alternating_iterations must be 1 (ACC-Collab) or "
                "2 (ACC-Collab+)"
            )
        if self.method.deliberation_rounds != 5:
            raise ACCCollabConfigError(
                "method.deliberation_rounds must be 5 for the paper-original protocol"
            )
        expected_method_values = {
            "training_order": "critic_then_actor",
            "reward_estimator": "one_step_mc",
            "pair_rule": "paper_eq5_if_elif",
        }
        for name, expected in expected_method_values.items():
            actual = getattr(self.method, name)
            if actual != expected:
                raise ACCCollabConfigError(
                    f"method.{name} must be {expected!r}, got {actual!r}"
                )

        if self.training.lora.r != 256:
            raise ACCCollabConfigError(
                "training.lora.r must be 256 (ACC-Collab Appendix A)"
            )
        if abs(self.training.dpo.nll_weight - 1.0) > 1e-12:
            raise ACCCollabConfigError(
                "training.dpo.nll_weight must be 1.0 (ACC-Collab Appendix A)"
            )

        allowed_strategies = {"full", "random", "stratified_by_subject"}
        for path, split in {
            "data.preference": self.data.preference,
            "data.eval": self.data.eval,
        }.items():
            if split.strategy not in allowed_strategies:
                raise ACCCollabConfigError(
                    f"{path}.strategy must be one of {sorted(allowed_strategies)}"
                )
            if split.strategy == "full" and split.samples is not None:
                raise ACCCollabConfigError(
                    f"{path}.samples must be null when strategy is 'full'"
                )
            if split.samples is not None and split.samples <= 0:
                raise ACCCollabConfigError(f"{path}.samples must be positive or null")
            if split.expected_samples is not None and split.expected_samples <= 0:
                raise ACCCollabConfigError(
                    f"{path}.expected_samples must be positive or null"
                )
            if split.sampling_seed is not None and split.sampling_seed < 0:
                raise ACCCollabConfigError(
                    f"{path}.sampling_seed must be non-negative or null"
                )

        if self.data.mmlu_load_mode not in {"all", "by_subject"}:
            raise ACCCollabConfigError(
                "data.mmlu_load_mode must be 'all' or 'by_subject'"
            )
        if self.data.benchmark_data_dir is not None and not str(
            self.data.benchmark_data_dir
        ).strip():
            raise ACCCollabConfigError(
                "data.benchmark_data_dir must be a non-empty string or null"
            )
        if self.data.preference_trials < 1:
            raise ACCCollabConfigError("data.preference_trials must be positive")
        for name, devices in {
            "runtime.generation_devices": self.runtime.generation_devices,
            "runtime.training_devices": self.runtime.training_devices,
        }.items():
            if not devices or any(device < 0 for device in devices):
                raise ACCCollabConfigError(f"{name} must contain non-negative GPU ids")
        if not 0 < self.runtime.gpu_memory_utilization <= 1:
            raise ACCCollabConfigError(
                "runtime.gpu_memory_utilization must be in (0, 1]"
            )
        if min(
            self.runtime.batch.preference_generation,
            self.runtime.batch.evaluation,
        ) <= 0:
            raise ACCCollabConfigError("runtime batch sizes must be positive")

        if min(self.tokens.actor, self.tokens.critic) <= 0:
            raise ACCCollabConfigError("tokens.actor and tokens.critic must be positive")
        if min(
            self.tokens.dpo_max_length,
            self.tokens.dpo_prompt,
            self.tokens.dpo_completion,
        ) <= 0:
            raise ACCCollabConfigError("all DPO token budgets must be positive")
        if self.tokens.dpo_prompt + self.tokens.dpo_completion > self.tokens.dpo_max_length:
            raise ACCCollabConfigError(
                "tokens.dpo_prompt + tokens.dpo_completion must not exceed "
                "tokens.dpo_max_length"
            )

        if self.reward.rollouts < 2:
            raise ACCCollabConfigError(
                "reward.rollouts must be >= 2 because §4.2 specifies multiple simulations"
            )
        if not 0 <= self.reward.epsilon <= 1:
            raise ACCCollabConfigError("reward.epsilon must be in [0, 1]")
        if self.reward.min_pairs_per_stage < 1:
            raise ACCCollabConfigError("reward.min_pairs_per_stage must be positive")
        for name, value in {
            "generation.train_temperature": self.generation.train_temperature,
            "generation.eval_temperature": self.generation.eval_temperature,
        }.items():
            if value < 0:
                raise ACCCollabConfigError(f"{name} must be non-negative")
        if not 0 < self.generation.top_p <= 1:
            raise ACCCollabConfigError("generation.top_p must be in (0, 1]")
        if not 0 <= self.generation.truncation.warn_rate <= 1:
            raise ACCCollabConfigError("generation.truncation.warn_rate must be in [0, 1]")
        if not 0 <= self.generation.truncation.fail_rate <= 1:
            raise ACCCollabConfigError("generation.truncation.fail_rate must be in [0, 1]")
        if (
            self.generation.truncation.warn_rate
            > self.generation.truncation.fail_rate
        ):
            raise ACCCollabConfigError(
                "generation.truncation.warn_rate must not exceed fail_rate"
            )
        if self.evaluation.trials < 1:
            raise ACCCollabConfigError("evaluation.trials must be positive")
        if self.evaluation.policy_output_dir is not None and (
            not isinstance(self.evaluation.policy_output_dir, str)
            or not self.evaluation.policy_output_dir.strip()
        ):
            raise ACCCollabConfigError(
                "evaluation.policy_output_dir must be a non-empty string or null"
            )

        for name, adapter in {
            "initialization.actor_adapter": self.initialization.actor_adapter,
            "initialization.critic_adapter": self.initialization.critic_adapter,
        }.items():
            if adapter is not None and (
                not isinstance(adapter, str) or not adapter.strip()
            ):
                raise ACCCollabConfigError(f"{name} must be a non-empty string or null")
        role = self.prompt_role
        if not role.name.strip():
            raise ACCCollabConfigError("prompt_role.name must be non-empty")
        if role.name == "original":
            if role.actor_instruction.strip() or role.critic_instruction.strip():
                raise ACCCollabConfigError(
                    "prompt_role original must not define specialization instructions"
                )
        elif not role.actor_instruction.strip() or not role.critic_instruction.strip():
            raise ACCCollabConfigError(
                "specialized prompt_role requires both actor_instruction and "
                "critic_instruction"
            )
        if not role.implementation_version.strip():
            raise ACCCollabConfigError(
                "prompt_role.implementation_version must be non-empty"
            )

        dpo = self.training.dpo
        if min(
            dpo.learning_rate,
            dpo.batch_size,
            dpo.gradient_accumulation_steps,
            dpo.epochs,
        ) <= 0:
            raise ACCCollabConfigError(
                "DPO learning rate, batch sizes, and epochs must be positive"
            )
        if dpo.beta <= 0:
            raise ACCCollabConfigError("training.dpo.beta must be positive")
        if dpo.loss_type not in {"sigmoid", "hinge", "ipo"}:
            raise ACCCollabConfigError(
                "training.dpo.loss_type must be sigmoid, hinge, or ipo"
            )


def load_acccollab_config(
    config_path: str,
    overrides: list[str] | None = None,
) -> ACCCollabConfig:
    """Load one isolated ACC-Collab YAML and reject unknown or missing keys."""
    resolved = resolve_acccollab_config_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    payload = OmegaConf.load(resolved)
    if overrides:
        payload = OmegaConf.merge(payload, OmegaConf.from_dotlist(overrides))
    raw = OmegaConf.to_container(payload, resolve=True, throw_on_missing=False)
    if not isinstance(raw, dict):
        raise ACCCollabConfigError(f"Config root must be a mapping: {resolved}")
    config = _build_dataclass(ACCCollabConfig, raw, "config")
    config.validate()
    return config


def resolve_acccollab_config_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return PROJECT_ROOT / candidate


def _build_dataclass(cls: type[Any], data: Any, path: str) -> Any:
    if not is_dataclass(cls) or not isinstance(data, dict):
        raise ACCCollabConfigError(f"{path} must be a mapping")
    field_map = {item.name: item for item in fields(cls)}
    unknown = sorted(set(data) - set(field_map))
    if unknown:
        raise ACCCollabConfigError(
            f"Unknown config key(s) under {path}: {', '.join(unknown)}"
        )
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in data:
            if item.default is MISSING and item.default_factory is MISSING:
                raise ACCCollabConfigError(f"Missing required config key: {path}.{item.name}")
            continue
        value = data[item.name]
        hint = hints.get(item.name, Any)
        nested = _nested_dataclass_type(hint)
        if nested is not None:
            value = _build_dataclass(nested, value, f"{path}.{item.name}")
        else:
            value = _coerce(value, hint, f"{path}.{item.name}")
        kwargs[item.name] = value
    return cls(**kwargs)


def _nested_dataclass_type(hint: Any) -> type[Any] | None:
    if isinstance(hint, type) and is_dataclass(hint):
        return hint
    args = [arg for arg in get_args(hint) if arg is not type(None)]
    if len(args) == 1 and isinstance(args[0], type) and is_dataclass(args[0]):
        return args[0]
    return None


def _coerce(value: Any, hint: Any, path: str) -> Any:
    if hint is bool:
        if isinstance(value, bool):
            return value
        raise ACCCollabConfigError(f"{path} must be a boolean")
    if hint is int:
        if isinstance(value, bool):
            raise ACCCollabConfigError(f"{path} must be an integer")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ACCCollabConfigError(f"{path} must be an integer") from exc
    if hint is float:
        if isinstance(value, bool):
            raise ACCCollabConfigError(f"{path} must be a number")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ACCCollabConfigError(f"{path} must be a number") from exc
    if hint is str:
        if not isinstance(value, str):
            raise ACCCollabConfigError(f"{path} must be a string")
        return value
    origin = get_origin(hint)
    if origin is list:
        if not isinstance(value, list):
            raise ACCCollabConfigError(f"{path} must be a list")
        args = get_args(hint)
        item_hint = args[0] if args else Any
        return [_coerce(item, item_hint, f"{path}[{idx}]") for idx, item in enumerate(value)]
    return value
