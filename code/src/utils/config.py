"""Experiment-level YAML config for the paired Actor/Critic pipeline."""

from __future__ import annotations

import argparse
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from omegaconf import OmegaConf

from src.paired import ACTOR_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigKeyError(Exception):
    """Raised when a config file is missing required keys or has unknown keys."""


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    name: str
    dtype: str = "bfloat16"
    max_model_len: int = 8192
    language_model_only: bool = True


@dataclass(frozen=True)
class AgentsConfig:
    actors: list[str] = field(default_factory=lambda: list(ACTOR_NAMES))


@dataclass(frozen=True)
class SplitConfig:
    split: str
    samples: int | None = None
    strategy: str = "random"
    expected_samples: int | None = None


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    train: SplitConfig
    eval: SplitConfig
    dpo: SplitConfig | None = None
    mmlu_load_mode: str = "by_subject"

    @property
    def dpo_source(self) -> SplitConfig:
        """Split to use for DPO data generation (defaults to train)."""
        return self.dpo or self.train


@dataclass(frozen=True)
class RuntimeBatchConfig:
    sft_candidates: int = 32
    dpo_generation: int = 4
    evaluation: int = 16


@dataclass(frozen=True)
class RuntimeConfig:
    # Generation uses independent one-GPU vLLM workers. The pipeline shards
    # generation stages over these physical GPU ids.
    generation_devices: list[int] = field(default_factory=lambda: [0])
    # Fine-tuning launches one isolated process per actor/critic and assigns
    # them round-robin to these physical GPU ids.
    training_devices: list[int] = field(default_factory=lambda: [0])
    gpu_memory_utilization: float = 0.75
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    disable_lora_cudagraph: bool = True
    gdn_prefill_backend: str | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    batch: RuntimeBatchConfig = field(default_factory=RuntimeBatchConfig)


@dataclass(frozen=True)
class TokenConfig:
    actor: int = 4096
    critic: int = 2048
    summary: int = 1024
    judge: int = 2048
    train_sequence: int = 4096
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
    sft_temperatures: list[float] = field(default_factory=lambda: [0.4, 0.7, 1.0])
    deliberation_temperature: float = 0.7
    summary_temperature: float = 0.0
    eval_temperature: float = 0.0
    top_p: float = 0.9
    truncation: TruncationConfig = field(default_factory=TruncationConfig)
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)


@dataclass(frozen=True)
class RewardConfig:
    # ACC-Collab §4.2: reward is a one-step Monte-Carlo roll-out averaged over
    # `rollouts` simulations. With 3 rollouts the reward is quantized to thirds,
    # so noise filtering is handled by epsilon rather than the rollout count.
    rollouts: int = 3
    # Must be >= 3: the candidate occupies 1 forced round and needs at least 1
    # natural continuation round to be a genuine roll-out (matches paper). At 2
    # the reward degenerates to the candidate's own correctness.
    final_rounds: int = 3
    # ACC-Collab Eq. 5 keeps a pair only when the reward delta >= epsilon.
    # 0.33 admits the smallest positive delta (1/3 at 3 rollouts); scales to a
    # >=2/5 margin if rollouts is later raised to 5. See configs/paired/*.yaml.
    epsilon: float = 0.33
    # ACC-Collab Algorithm 1 collects guided-trajectory pairs at every round
    # t in [1, T]. trajectory_rounds=1 (legacy) collects only round-1 pairs;
    # raising it toward the paper's T makes pair collection multi-round at a
    # roughly linear cost in reward-estimation compute.
    trajectory_rounds: int = 1
    min_pairs_per_agent: int = 256


@dataclass(frozen=True)
class LoraTrainingConfig:
    r: int = 128
    alpha: int = 256


@dataclass(frozen=True)
class SFTTrainingConfig:
    min_examples_per_actor: int = 256
    learning_rate: float = 5.0e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    epochs: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    checkpoint_steps: int = 100
    checkpoint_total_limit: int = 2
    resume_from_checkpoint: bool = True


@dataclass(frozen=True)
class DPOTrainingConfig:
    learning_rate: float = 5.0e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    beta: float = 0.1
    # ACC-Collab Appendix A applies an NLL regularizer (weight 1) for DPO of BOTH
    # actors and critics. The critic default previously differed (0.0), leaving
    # the noisiest-reward agent unanchored; it now matches the actor.
    actor_nll_weight: float = 0.3
    critic_nll_weight: float = 0.3
    optim: str = "paged_adamw_32bit"
    timeout_per_1k: int = 3600
    checkpoint_steps: int = 100
    checkpoint_total_limit: int = 2
    resume_from_checkpoint: bool = True
    resume_optimizer_state: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    lora: LoraTrainingConfig = field(default_factory=LoraTrainingConfig)
    sft: SFTTrainingConfig = field(default_factory=SFTTrainingConfig)
    dpo: DPOTrainingConfig = field(default_factory=DPOTrainingConfig)


@dataclass(frozen=True)
class EvaluationConfig:
    rounds: int = 3


@dataclass(frozen=True)
class ExperimentPaths:
    output_dir: Path

    @property
    def marker_dir(self) -> Path:
        return self.output_dir / "markers"

    @property
    def actor_sft_data_dir(self) -> Path:
        return self.output_dir / "data" / "actor_sft"

    @property
    def actors_sft_dir(self) -> Path:
        return self.output_dir / "adapters" / "actors_sft"

    @property
    def actor_registry(self) -> Path:
        return self.actors_sft_dir / "actor_registry.json"

    @property
    def critic_dpo_data_dir(self) -> Path:
        return self.output_dir / "data" / "critic_dpo"

    @property
    def critics_dpo_dir(self) -> Path:
        return self.output_dir / "adapters" / "critics_dpo"

    @property
    def critic_registry(self) -> Path:
        return self.critics_dpo_dir / "critic_registry.json"

    @property
    def actor_dpo_data_dir(self) -> Path:
        return self.output_dir / "data" / "actor_dpo"

    @property
    def paired_dpo_dir(self) -> Path:
        return self.output_dir / "adapters" / "paired_dpo"

    @property
    def final_registry(self) -> Path:
        return self.paired_dpo_dir / "final_agent_registry.json"

    @property
    def eval_dir(self) -> Path:
        return self.output_dir / "eval"


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    model: ModelConfig
    agents: AgentsConfig
    data: DataConfig
    runtime: RuntimeConfig
    tokens: TokenConfig
    generation: GenerationConfig
    reward: RewardConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    @property
    def paths(self) -> ExperimentPaths:
        return ExperimentPaths(Path(self.run.output_dir))

    @property
    def actor_names(self) -> list[str]:
        return list(self.agents.actors)

    @property
    def max_lora_rank(self) -> int:
        return int(self.training.lora.r)

    @property
    def all_agent_max_loras(self) -> int:
        return len(self.actor_names) * 2

    def validate(self) -> None:
        expected = list(ACTOR_NAMES)
        if self.actor_names != expected:
            raise ConfigKeyError(f"agents.actors must be exactly {expected}, got {self.actor_names}")

        allowed_strategies = {"full", "random", "stratified_by_subject"}
        for key, split in {
            "data.train": self.data.train,
            "data.dpo": self.data.dpo_source,
            "data.eval": self.data.eval,
        }.items():
            if split.strategy not in allowed_strategies:
                raise ConfigKeyError(
                    f"{key}.strategy must be one of {sorted(allowed_strategies)}, "
                    f"got {split.strategy!r}"
                )
            if split.strategy == "full" and split.samples is not None:
                raise ConfigKeyError(
                    f"{key}.samples must be null when strategy is 'full'; "
                    "use expected_samples to assert the full split size"
                )
            if split.samples is not None and int(split.samples) <= 0:
                raise ConfigKeyError(f"{key}.samples must be positive or null")
            if split.expected_samples is not None and int(split.expected_samples) <= 0:
                raise ConfigKeyError(f"{key}.expected_samples must be positive or null")

        for key, devices in {
            "runtime.generation_devices": self.runtime.generation_devices,
            "runtime.training_devices": self.runtime.training_devices,
        }.items():
            if not devices:
                raise ConfigKeyError(f"{key} must not be empty")
            if any(int(device) < 0 for device in devices):
                raise ConfigKeyError(f"{key} must contain non-negative GPU ids, got {devices}")
            if len(set(devices)) != len(devices):
                raise ConfigKeyError(f"{key} must not contain duplicate GPU ids, got {devices}")

        if not 0.0 < float(self.runtime.gpu_memory_utilization) < 1.0:
            raise ConfigKeyError("runtime.gpu_memory_utilization must be between 0 and 1")
        if not 0.0 < float(self.generation.top_p) <= 1.0:
            raise ConfigKeyError("generation.top_p must be in (0, 1]")
        warn_rate = float(self.generation.truncation.warn_rate)
        fail_rate = float(self.generation.truncation.fail_rate)
        if not 0.0 <= warn_rate <= 1.0 or not 0.0 <= fail_rate <= 1.0:
            raise ConfigKeyError("generation.truncation warn/fail rates must be in [0, 1]")
        if warn_rate > fail_rate:
            raise ConfigKeyError(
                "generation.truncation.warn_rate must be <= truncation.fail_rate"
            )

        if not self.generation.sft_temperatures:
            raise ConfigKeyError("generation.sft_temperatures must not be empty")
        if any(float(item) < 0.0 for item in self.generation.sft_temperatures):
            raise ConfigKeyError("generation.sft_temperatures must be non-negative")

        positive_values = {
            "model.max_model_len": self.model.max_model_len,
            "runtime.batch.sft_candidates": self.runtime.batch.sft_candidates,
            "runtime.batch.dpo_generation": self.runtime.batch.dpo_generation,
            "runtime.batch.evaluation": self.runtime.batch.evaluation,
            "tokens.actor": self.tokens.actor,
            "tokens.critic": self.tokens.critic,
            "tokens.summary": self.tokens.summary,
            "tokens.judge": self.tokens.judge,
            "tokens.train_sequence": self.tokens.train_sequence,
            "tokens.dpo_prompt": self.tokens.dpo_prompt,
            "tokens.dpo_completion": self.tokens.dpo_completion,
            "reward.rollouts": self.reward.rollouts,
            "reward.final_rounds": self.reward.final_rounds,
            "evaluation.rounds": self.evaluation.rounds,
            "training.sft.batch_size": self.training.sft.batch_size,
            "training.sft.gradient_accumulation_steps": (
                self.training.sft.gradient_accumulation_steps
            ),
            "training.dpo.batch_size": self.training.dpo.batch_size,
            "training.dpo.gradient_accumulation_steps": (
                self.training.dpo.gradient_accumulation_steps
            ),
        }
        for key, value in positive_values.items():
            if int(value) <= 0:
                raise ConfigKeyError(f"{key} must be positive, got {value}")

        for key, value in {
            "tokens.actor": self.tokens.actor,
            "tokens.critic": self.tokens.critic,
            "tokens.summary": self.tokens.summary,
            "tokens.judge": self.tokens.judge,
            "tokens.train_sequence": self.tokens.train_sequence,
        }.items():
            if int(value) >= int(self.model.max_model_len):
                raise ConfigKeyError(
                    f"{key} must be smaller than model.max_model_len="
                    f"{self.model.max_model_len}, got {value}"
                )

        if self.tokens.dpo_prompt + self.tokens.dpo_completion > self.tokens.train_sequence:
            raise ConfigKeyError(
                "tokens.dpo_prompt + tokens.dpo_completion must be <= tokens.train_sequence"
            )
        if int(self.reward.final_rounds) < 3:
            raise ConfigKeyError(
                "reward.final_rounds must be >= 3: the candidate occupies 1 forced "
                "round and needs >= 1 natural continuation round to be a genuine "
                "ACC-Collab one-step roll-out (final_rounds=2 collapses the reward "
                "to the candidate's own correctness)."
            )
        if float(self.reward.epsilon) < 0.0:
            raise ConfigKeyError("reward.epsilon must be non-negative")
        if float(self.reward.epsilon) >= 1.0:
            raise ConfigKeyError(
                "reward.epsilon must be < 1.0 (rewards are 0/1-averaged deltas)"
            )
        if int(self.reward.trajectory_rounds) < 1:
            raise ConfigKeyError("reward.trajectory_rounds must be >= 1")

    def actor_sft_data_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            **self._vllm_args(),
            dataset=self.data.dataset,
            output_dir=str(self.paths.actor_sft_data_dir),
            source_split=self.data.train.split,
            max_samples=self.data.train.samples,
            sample_strategy=self.data.train.strategy,
            expected_samples=self.data.train.expected_samples,
            actor_names=self.actor_names,
            temperatures=list(self.generation.sft_temperatures),
            top_p=float(self.generation.top_p),
            max_tokens=int(self.tokens.actor),
            truncation_warn_rate=float(self.generation.truncation.warn_rate),
            truncation_fail_rate=float(self.generation.truncation.fail_rate),
            truncation_fail_on_excess=bool(self.generation.truncation.fail_on_excess),
            batch_size=int(self.runtime.batch.sft_candidates),
            enable_thinking=bool(self.generation.thinking.train),
            mmlu_load_mode=self.data.mmlu_load_mode,
        )

    def train_actors_sft_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            model_name=self.model.name,
            input_dir=str(self.paths.actor_sft_data_dir),
            output_dir=str(self.paths.actors_sft_dir),
            actor_names=self.actor_names,
            min_examples_per_actor=int(self.training.sft.min_examples_per_actor),
            **self._sft_training_args(),
        )

    def critic_dpo_data_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            **self._vllm_args(),
            dataset=self.data.dataset,
            actor_registry=str(self.paths.actor_registry),
            output_dir=str(self.paths.critic_dpo_data_dir),
            source_split=self.data.dpo_source.split,
            max_samples=self.data.dpo_source.samples,
            sample_strategy=self.data.dpo_source.strategy,
            expected_samples=self.data.dpo_source.expected_samples,
            actor_names=self.actor_names,
            actor_max_tokens=int(self.tokens.actor),
            critic_max_tokens=int(self.tokens.critic),
            summary_max_tokens=int(self.tokens.summary),
            temperature=float(self.generation.deliberation_temperature),
            summary_temperature=float(self.generation.summary_temperature),
            top_p=float(self.generation.top_p),
            truncation_warn_rate=float(self.generation.truncation.warn_rate),
            truncation_fail_rate=float(self.generation.truncation.fail_rate),
            truncation_fail_on_excess=bool(self.generation.truncation.fail_on_excess),
            batch_size=int(self.runtime.batch.dpo_generation),
            reward_rollouts=int(self.reward.rollouts),
            reward_final_rounds=int(self.reward.final_rounds),
            reward_epsilon=float(self.reward.epsilon),
            reward_trajectory_rounds=int(self.reward.trajectory_rounds),
            min_pairs_per_agent=int(self.reward.min_pairs_per_agent),
            enable_thinking=bool(self.generation.thinking.train),
            max_lora_rank=self.max_lora_rank,
            disable_lora_cudagraph=bool(self.runtime.disable_lora_cudagraph),
            mmlu_load_mode=self.data.mmlu_load_mode,
        )

    def train_critics_dpo_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            model_name=self.model.name,
            actor_registry=str(self.paths.actor_registry),
            critic_dpo_data_dir=str(self.paths.critic_dpo_data_dir),
            output_dir=str(self.paths.critics_dpo_dir),
            actor_names=self.actor_names,
            critic_nll_weight=float(self.training.dpo.critic_nll_weight),
            **self._dpo_training_args(),
        )

    def actor_dpo_data_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            **self._vllm_args(),
            actor_registry=str(self.paths.actor_registry),
            critic_registry=str(self.paths.critic_registry),
            output_dir=str(self.paths.actor_dpo_data_dir),
            dataset=self.data.dataset,
            source_split=self.data.dpo_source.split,
            max_samples=self.data.dpo_source.samples,
            sample_strategy=self.data.dpo_source.strategy,
            expected_samples=self.data.dpo_source.expected_samples,
            actor_names=self.actor_names,
            actor_max_tokens=int(self.tokens.actor),
            critic_max_tokens=int(self.tokens.critic),
            summary_max_tokens=int(self.tokens.summary),
            temperature=float(self.generation.deliberation_temperature),
            summary_temperature=float(self.generation.summary_temperature),
            top_p=float(self.generation.top_p),
            truncation_warn_rate=float(self.generation.truncation.warn_rate),
            truncation_fail_rate=float(self.generation.truncation.fail_rate),
            truncation_fail_on_excess=bool(self.generation.truncation.fail_on_excess),
            batch_size=int(self.runtime.batch.dpo_generation),
            reward_rollouts=int(self.reward.rollouts),
            reward_final_rounds=int(self.reward.final_rounds),
            reward_epsilon=float(self.reward.epsilon),
            reward_trajectory_rounds=int(self.reward.trajectory_rounds),
            min_pairs_per_agent=int(self.reward.min_pairs_per_agent),
            enable_thinking=bool(self.generation.thinking.train),
            max_loras=self.all_agent_max_loras,
            max_lora_rank=self.max_lora_rank,
            disable_lora_cudagraph=bool(self.runtime.disable_lora_cudagraph),
            mmlu_load_mode=self.data.mmlu_load_mode,
        )

    def train_actors_dpo_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            model_name=self.model.name,
            actor_registry=str(self.paths.actor_registry),
            critic_registry=str(self.paths.critic_registry),
            actor_dpo_data_dir=str(self.paths.actor_dpo_data_dir),
            output_dir=str(self.paths.paired_dpo_dir),
            actor_names=self.actor_names,
            actor_nll_weight=float(self.training.dpo.actor_nll_weight),
            **self._dpo_training_args(),
        )

    def evaluate_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            **self._vllm_args(),
            registry=str(self.paths.final_registry),
            output_dir=str(self.paths.eval_dir),
            dataset=self.data.dataset,
            source_split=self.data.eval.split,
            max_samples=self.data.eval.samples,
            sample_strategy=self.data.eval.strategy,
            expected_samples=self.data.eval.expected_samples,
            actor_names=self.actor_names,
            eval_rounds=int(self.evaluation.rounds),
            actor_max_tokens=int(self.tokens.actor),
            critic_max_tokens=int(self.tokens.critic),
            summary_max_tokens=int(self.tokens.summary),
            judge_max_tokens=int(self.tokens.judge),
            temperature=float(self.generation.eval_temperature),
            summary_temperature=float(self.generation.summary_temperature),
            top_p=float(self.generation.top_p),
            truncation_warn_rate=float(self.generation.truncation.warn_rate),
            truncation_fail_rate=float(self.generation.truncation.fail_rate),
            truncation_fail_on_excess=bool(self.generation.truncation.fail_on_excess),
            batch_size=int(self.runtime.batch.evaluation),
            max_loras=self.all_agent_max_loras,
            max_lora_rank=self.max_lora_rank,
            disable_lora_cudagraph=bool(self.runtime.disable_lora_cudagraph),
            enable_thinking=bool(self.generation.thinking.eval),
            mmlu_load_mode=self.data.mmlu_load_mode,
        )

    def _vllm_args(self) -> dict[str, Any]:
        return {
            "model_name": self.model.name,
            "model_type": getattr(self.model, "type", None),
            "seed": int(self.run.seed),
            "device": self.runtime.generation_devices[0],
            "dtype": self.model.dtype,
            "gpu_memory_utilization": float(self.runtime.gpu_memory_utilization),
            "max_model_len": int(self.model.max_model_len),
            "enable_prefix_caching": bool(self.runtime.enable_prefix_caching),
            "max_num_batched_tokens": self.runtime.max_num_batched_tokens,
            "max_num_seqs": self.runtime.max_num_seqs,
            "enforce_eager": bool(self.runtime.enforce_eager),
            "language_model_only": bool(self.model.language_model_only),
            "gdn_prefill_backend": self.runtime.gdn_prefill_backend,
        }

    def _sft_training_args(self) -> dict[str, Any]:
        return {
            "lora_r": int(self.training.lora.r),
            "lora_alpha": int(self.training.lora.alpha),
            "learning_rate": float(self.training.sft.learning_rate),
            "batch_size": int(self.training.sft.batch_size),
            "gradient_accumulation_steps": int(self.training.sft.gradient_accumulation_steps),
            "num_epochs": int(self.training.sft.epochs),
            "max_length": int(self.tokens.train_sequence),
            "warmup_ratio": float(self.training.sft.warmup_ratio),
            "weight_decay": float(self.training.sft.weight_decay),
            "max_grad_norm": float(self.training.sft.max_grad_norm),
            "checkpoint_steps": int(self.training.sft.checkpoint_steps),
            "checkpoint_total_limit": int(self.training.sft.checkpoint_total_limit),
            "resume_from_checkpoint": bool(self.training.sft.resume_from_checkpoint),
            "seed": int(self.run.seed),
            "device": self._training_device(),
        }

    def _dpo_training_args(self) -> dict[str, Any]:
        return {
            "lora_r": int(self.training.lora.r),
            "lora_alpha": int(self.training.lora.alpha),
            "learning_rate": float(self.training.dpo.learning_rate),
            "batch_size": int(self.training.dpo.batch_size),
            "gradient_accumulation_steps": int(self.training.dpo.gradient_accumulation_steps),
            "num_epochs": int(self.training.dpo.epochs),
            "beta": float(self.training.dpo.beta),
            "max_length": int(self.tokens.train_sequence),
            "max_prompt_length": int(self.tokens.dpo_prompt),
            "max_completion_length": int(self.tokens.dpo_completion),
            "optim": self.training.dpo.optim,
            "timeout_per_1k": int(self.training.dpo.timeout_per_1k),
            "checkpoint_steps": int(self.training.dpo.checkpoint_steps),
            "checkpoint_total_limit": int(self.training.dpo.checkpoint_total_limit),
            "resume_from_checkpoint": bool(self.training.dpo.resume_from_checkpoint),
            "resume_optimizer_state": bool(self.training.dpo.resume_optimizer_state),
            "seed": int(self.run.seed),
            "device": self._training_device(),
        }

    def _training_device(self) -> int:
        if not self.runtime.training_devices:
            raise ConfigKeyError("runtime.training_devices must not be empty")
        return int(self.runtime.training_devices[0])


def load_experiment_config(
    config_path: str,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load one experiment config and validate the public schema."""
    resolved = resolve_config_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    config = OmegaConf.load(resolved)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=False)
    if not isinstance(raw, dict):
        raise ConfigKeyError(f"Config root must be a mapping: {resolved}")

    cfg = _build_dataclass(ExperimentConfig, raw, "config")
    cfg.validate()
    return cfg


def resolve_config_path(path: str) -> Path:
    """Resolve config paths relative to cwd first, then the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return PROJECT_ROOT / candidate


def _build_dataclass(cls: type[Any], data: Any, path: str) -> Any:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")
    if not isinstance(data, dict):
        raise ConfigKeyError(f"{path} must be a mapping")

    field_map = {item.name: item for item in fields(cls)}
    unknown = sorted(set(data) - set(field_map))
    if unknown:
        raise ConfigKeyError(f"Unknown config key(s) under {path}: {', '.join(unknown)}")

    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in data:
            if item.default is MISSING and item.default_factory is MISSING:
                raise ConfigKeyError(f"Missing required config key: {path}.{item.name}")
            continue

        value = data[item.name]
        type_hint = type_hints.get(item.name)

        # Unwrap Optional[T] / T | None so nested dataclasses are built correctly.
        origin = get_origin(type_hint)
        if origin is not None:
            args = [a for a in get_args(type_hint) if a is not type(None)]
            if len(args) == 1 and isinstance(args[0], type) and is_dataclass(args[0]):
                type_hint = args[0]

        if isinstance(type_hint, type) and is_dataclass(type_hint):
            value = _build_dataclass(type_hint, value, f"{path}.{item.name}")
        else:
            value = _coerce_config_value(value, type_hint, f"{path}.{item.name}")
        kwargs[item.name] = value
    return cls(**kwargs)


def _coerce_config_value(value: Any, type_hint: Any, path: str) -> Any:
    if type_hint is bool:
        return _coerce_bool(value, path)
    if type_hint is int:
        return _coerce_int(value, path)
    if type_hint is float:
        return _coerce_float(value, path)
    if type_hint is str:
        if not isinstance(value, str):
            raise ConfigKeyError(f"{path} must be a string, got {type(value).__name__}")
        return value

    origin = get_origin(type_hint)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigKeyError(f"{path} must be a list, got {type(value).__name__}")
        args = get_args(type_hint)
        item_hint = args[0] if args else Any
        return [
            _coerce_config_value(item, item_hint, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    return value


def _coerce_bool(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ConfigKeyError(f"{path} must be a boolean, got {value!r}")


def _coerce_int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise ConfigKeyError(f"{path} must be an integer, got boolean {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigKeyError(f"{path} must be an integer, got {value!r}") from exc


def _coerce_float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ConfigKeyError(f"{path} must be a number, got boolean {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigKeyError(f"{path} must be a number, got {value!r}") from exc
