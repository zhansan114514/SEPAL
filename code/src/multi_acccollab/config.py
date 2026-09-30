"""Strict configuration and derived role configs for multi-role ACC-Collab."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from omegaconf import OmegaConf

from src.acccollab.config import (
    ACCCollabConfig,
    InitializationConfig,
    PromptRoleConfig,
    load_acccollab_config,
)
from src.multi_acccollab import MULTI_ACCCOLLAB_VERSION
from src.utils.artifacts import (
    completed_adapter_path,
    file_sha256,
    path_identity,
    stable_fingerprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROLES = ("direct", "evidence", "verification")


class MultiACCCollabConfigError(ValueError):
    """Raised when the multi-role experiment config is incomplete or unfair."""


@dataclass(frozen=True)
class MultiRunConfig:
    name: str
    output_dir: str
    seed: int = 42


@dataclass(frozen=True)
class RoleConfig:
    name: str
    actor_instruction: str
    critic_instruction: str
    device: int
    seed_offset: int


@dataclass(frozen=True)
class SFTDataConfig:
    split: str = "train"
    samples: int = 10000
    strategy: str = "random"
    expected_samples: int = 10000


@dataclass(frozen=True)
class SFTGenerationConfig:
    temperatures: list[float] = field(default_factory=lambda: [0.4, 0.7, 1.0])
    top_p: float = 0.9
    max_tokens: int = 1024
    enable_thinking: bool = False


@dataclass(frozen=True)
class SFTTrainingConfig:
    min_examples_per_role: int = 256
    learning_rate: float = 5.0e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    max_length: int = 4096
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    timeout_per_1k: int = 3600
    checkpoint_steps: int = 100
    checkpoint_total_limit: int = 1
    resume_from_checkpoint: bool = True


@dataclass(frozen=True)
class SFTConfig:
    data: SFTDataConfig = field(default_factory=SFTDataConfig)
    generation: SFTGenerationConfig = field(default_factory=SFTGenerationConfig)
    training: SFTTrainingConfig = field(default_factory=SFTTrainingConfig)


@dataclass(frozen=True)
class MultiRuntimeConfig:
    sft_generation_devices: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    sft_generation_batch_size: int = 64
    parallel_role_pipelines: bool = True


@dataclass(frozen=True)
class MajorityEvaluationConfig:
    decision_rule: str = "final_round_majority_then_fixed_direct"
    fallback_role: str = "direct"
    use_judge: bool = False
    policy_source_output_dir: str | None = None


@dataclass(frozen=True)
class MultiACCCollabPaths:
    output_dir: Path

    @property
    def sft_data_dir(self) -> Path:
        return self.output_dir / "sft" / "data"

    @property
    def sft_adapter_root(self) -> Path:
        return self.output_dir / "sft" / "adapters"

    def sft_training_output(self, role_name: str) -> Path:
        return self.sft_adapter_root / role_name

    def sft_adapter(self, role_name: str) -> Path:
        return Path(str(self.sft_training_output(role_name)) + "_adapter")

    @property
    def sft_registry(self) -> Path:
        return self.output_dir / "sft" / "registry.json"

    @property
    def role_config_dir(self) -> Path:
        return self.output_dir / "resolved_role_configs"

    def role_config(self, role_name: str) -> Path:
        return self.role_config_dir / f"{role_name}.yaml"

    def role_output(self, role_name: str) -> Path:
        return self.output_dir / "roles" / role_name

    @property
    def role_manifest(self) -> Path:
        return self.role_config_dir / "manifest.json"

    @property
    def majority_eval_dir(self) -> Path:
        return self.output_dir / "eval_majority"

    @property
    def pipeline_marker_dir(self) -> Path:
        return self.output_dir / "pipeline_markers"


@dataclass(frozen=True)
class MultiACCCollabConfig:
    run: MultiRunConfig
    base_acccollab_config: str
    roles: list[RoleConfig]
    sft: SFTConfig
    runtime: MultiRuntimeConfig
    evaluation: MajorityEvaluationConfig
    profile: str = "primary"
    implementation_version: str = MULTI_ACCCOLLAB_VERSION

    @property
    def paths(self) -> MultiACCCollabPaths:
        return MultiACCCollabPaths(Path(self.run.output_dir))

    def role(self, name: str) -> RoleConfig:
        for role in self.roles:
            if role.name == name:
                return role
        raise KeyError(f"Unknown multi-role ACC-Collab role: {name}")

    def base_config(self) -> ACCCollabConfig:
        return load_acccollab_config(str(resolve_config_path(self.base_acccollab_config)))

    def validate(self) -> None:
        if not self.run.name.strip() or not self.run.output_dir.strip():
            raise MultiACCCollabConfigError("run.name and run.output_dir must be non-empty")
        if self.implementation_version != MULTI_ACCCOLLAB_VERSION:
            raise MultiACCCollabConfigError(
                f"implementation_version must be {MULTI_ACCCOLLAB_VERSION!r}"
            )
        if self.profile not in {
            "primary",
            "cross_eval",
            "smoke_train",
            "smoke_eval",
        }:
            raise MultiACCCollabConfigError(
                "profile must be primary, cross_eval, smoke_train, or smoke_eval"
            )
        names = [role.name for role in self.roles]
        if tuple(names) != EXPECTED_ROLES:
            raise MultiACCCollabConfigError(
                f"roles must be ordered exactly as {list(EXPECTED_ROLES)}, got {names}"
            )
        if any(role.device < 0 for role in self.roles):
            raise MultiACCCollabConfigError("role devices must be non-negative")
        # Roles may share a physical device when the machine has fewer GPUs
        # than roles. The orchestrators schedule same-device jobs in separate
        # batches, while still running conflict-free roles concurrently.
        if len({role.seed_offset for role in self.roles}) != len(self.roles):
            raise MultiACCCollabConfigError("role seed_offset values must be distinct")
        for role in self.roles:
            if not role.actor_instruction.strip() or not role.critic_instruction.strip():
                raise MultiACCCollabConfigError(
                    f"role {role.name!r} needs non-empty Actor and Critic instructions"
                )

        if self.profile == "primary" and (
            self.sft.data.split != "train"
            or self.sft.data.samples != 10000
            or self.sft.data.expected_samples != 10000
        ):
            raise MultiACCCollabConfigError(
                "The primary method requires exactly 10000 sampled MMLU train questions"
            )
        if self.profile == "smoke_train" and (
            self.sft.data.split != "train"
            or self.sft.data.samples < 1
            or self.sft.data.expected_samples != self.sft.data.samples
        ):
            raise MultiACCCollabConfigError(
                "smoke_train requires a positive, exact MMLU train sample count"
            )
        if self.sft.data.strategy not in {"random", "stratified_by_subject"}:
            raise MultiACCCollabConfigError("sft.data.strategy must be random or stratified")
        if not self.sft.generation.temperatures or any(
            value < 0 for value in self.sft.generation.temperatures
        ):
            raise MultiACCCollabConfigError("SFT temperatures must be non-empty and non-negative")
        if len(set(self.sft.generation.temperatures)) != len(
            self.sft.generation.temperatures
        ):
            raise MultiACCCollabConfigError("SFT temperatures must be unique")
        if not 0 < self.sft.generation.top_p <= 1:
            raise MultiACCCollabConfigError("sft.generation.top_p must be in (0, 1]")
        if min(
            self.sft.generation.max_tokens,
            self.sft.training.min_examples_per_role,
            self.sft.training.batch_size,
            self.sft.training.gradient_accumulation_steps,
            self.sft.training.epochs,
            self.sft.training.max_length,
            self.runtime.sft_generation_batch_size,
        ) <= 0:
            raise MultiACCCollabConfigError(
                "SFT sizes, token budgets, and epochs must be positive"
            )
        devices = self.runtime.sft_generation_devices
        if (
            not devices
            or any(device < 0 for device in devices)
            or len(set(devices)) != len(devices)
        ):
            raise MultiACCCollabConfigError(
                "runtime.sft_generation_devices must be unique non-negative ids"
            )
        if self.evaluation.decision_rule != "final_round_majority_then_fixed_direct":
            raise MultiACCCollabConfigError(
                "evaluation.decision_rule must be final_round_majority_then_fixed_direct"
            )
        if self.evaluation.fallback_role != "direct":
            raise MultiACCCollabConfigError("evaluation.fallback_role must be direct")
        if self.evaluation.use_judge:
            raise MultiACCCollabConfigError("The primary method forbids Judge evaluation")
        source_output = self.evaluation.policy_source_output_dir
        external_evaluation = self.profile in {"cross_eval", "smoke_eval"}
        if external_evaluation:
            if not isinstance(source_output, str) or not source_output.strip():
                raise MultiACCCollabConfigError(
                    f"{self.profile} requires evaluation.policy_source_output_dir"
                )
        elif source_output is not None:
            raise MultiACCCollabConfigError(
                "policy_source_output_dir is only valid for cross_eval or smoke_eval"
            )

        base = self.base_config()
        if not external_evaluation and base.data.dataset != "mmlu":
            raise MultiACCCollabConfigError("base ACC-Collab config must use MMLU")
        preference = base.data.preference
        if self.profile == "primary" and (
            preference.split != "validation"
            or preference.strategy != "full"
            or preference.samples is not None
            or preference.expected_samples != 1531
        ):
            raise MultiACCCollabConfigError(
                "base preference data must be the full 1531-example MMLU validation split"
            )
        if base.method.alternating_iterations != 1:
            raise MultiACCCollabConfigError(
                "base config must run one original ACC-Collab alternation"
            )
        if self.profile == "primary" and base.evaluation.trials != 1:
            raise MultiACCCollabConfigError("primary evaluation requires one trial")
        if self.profile == "cross_eval":
            expected_sizes = {
                "boolq": 3270,
                "mmlu": 14042,
                "bbh": 1260,
                "sciq": 1000,
                "arc": 3548,
            }
            expected_size = expected_sizes.get(base.data.dataset)
            evaluation = base.data.eval
            if expected_size is None:
                raise MultiACCCollabConfigError(
                    "cross_eval supports BoolQ, MMLU, BBH, SciQ, and ARC"
                )
            if (
                evaluation.strategy != "full"
                or evaluation.samples is not None
                or evaluation.expected_samples != expected_size
                or base.evaluation.trials != 1
            ):
                raise MultiACCCollabConfigError(
                    "cross_eval requires the full benchmark split and one trial"
                )
        if base.training.lora.r != 256 or base.training.lora.alpha != 512:
            raise MultiACCCollabConfigError("base config must retain paper LoRA r=256 alpha=512")
        if self.sft.training.max_length > base.model.max_model_len:
            raise MultiACCCollabConfigError("SFT max_length exceeds base model context")


def resolve_config_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return PROJECT_ROOT / candidate


def load_multi_acccollab_config(path: str) -> MultiACCCollabConfig:
    resolved = resolve_config_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Multi ACC-Collab config not found: {resolved}")
    raw = OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)
    if not isinstance(raw, dict):
        raise MultiACCCollabConfigError("Multi ACC-Collab config root must be a mapping")
    config = _build_dataclass(MultiACCCollabConfig, raw, "config")
    config.validate()
    return config


def build_role_acccollab_config(
    config: MultiACCCollabConfig,
    role: RoleConfig,
    *,
    require_sft_adapter: bool = True,
) -> ACCCollabConfig:
    """Derive one independent original-training run from the shared base config."""
    base = config.base_config()
    source_output = config.evaluation.policy_source_output_dir
    source_paths = (
        MultiACCCollabPaths(Path(source_output)) if source_output else config.paths
    )
    requested = source_paths.sft_training_output(role.name)
    completed = completed_adapter_path(requested)
    if require_sft_adapter and not completed:
        raise RuntimeError(f"Missing completed Actor SFT adapter for {role.name}: {requested}")
    actor_adapter = completed or str(config.paths.sft_adapter(role.name))
    derived = replace(
        base,
        run=replace(
            base.run,
            name=f"{config.run.name}_{role.name}",
            output_dir=str(config.paths.role_output(role.name)),
            seed=int(config.run.seed + role.seed_offset),
        ),
        runtime=replace(
            base.runtime,
            generation_devices=[int(role.device)],
            training_devices=[int(role.device)],
        ),
        data=replace(
            base.data,
            preference=replace(
                base.data.preference,
                sampling_seed=int(config.run.seed),
            ),
            eval=replace(
                base.data.eval,
                sampling_seed=int(config.run.seed),
            ),
        ),
        evaluation=replace(
            base.evaluation,
            policy_output_dir=(
                str(source_paths.role_output(role.name)) if source_output else None
            ),
        ),
        initialization=InitializationConfig(
            actor_adapter=actor_adapter,
            critic_adapter=None,
        ),
        prompt_role=PromptRoleConfig(
            name=role.name,
            actor_instruction=role.actor_instruction,
            critic_instruction=role.critic_instruction,
            implementation_version=(
                f"acccollab_role_prefix_v1+src_"
                f"{stable_fingerprint(_source_tree_identity())[:16]}"
            ),
        ),
    )
    derived.validate()
    return derived


def write_role_configs(config: MultiACCCollabConfig) -> dict[str, Any]:
    """Materialize exact per-role configs and a provenance manifest."""
    from src.acccollab.io import write_json

    output = config.paths.role_config_dir
    output.mkdir(parents=True, exist_ok=True)
    roles: dict[str, Any] = {}
    for role in config.roles:
        derived = build_role_acccollab_config(config, role, require_sft_adapter=True)
        path = config.paths.role_config(role.name)
        OmegaConf.save(config=OmegaConf.create(asdict(derived)), f=path)
        roles[role.name] = {
            "config_path": str(path),
            "config_sha256": file_sha256(path),
            "actor_sft_adapter": path_identity(
                derived.initialization.actor_adapter,
                hash_weights=True,
            ),
            "critic_initialization": "base_model",
            "output_dir": derived.run.output_dir,
            "device": role.device,
            "seed": derived.run.seed,
        }
    base_path = resolve_config_path(config.base_acccollab_config)
    manifest = {
        "schema_version": 1,
        "pipeline": "multi_acccollab",
        "implementation_version": config.implementation_version,
        "profile": config.profile,
        "config_fingerprint": stable_fingerprint(config_snapshot(config)),
        "base_config": {
            "path": str(base_path),
            "sha256": file_sha256(base_path),
        },
        "roles": roles,
        "training_protocol": {
            "actor_iteration_zero": (
                "external_trained_role_policy"
                if config.evaluation.policy_source_output_dir
                else f"role_sft_mmlu_train_{config.sft.data.samples}"
            ),
            "critic_iteration_zero": "base_model",
            "preference_split": (
                "external_policy_evaluation_only"
                if config.evaluation.policy_source_output_dir
                else (
                    f"{config.base_config().data.dataset}_"
                    f"{config.base_config().data.preference.split}_"
                    f"{config.base_config().data.preference.expected_samples}"
                )
            ),
            "acccollab_protocol": "paper_original_critic_then_actor",
        },
        "evaluation": asdict(config.evaluation),
    }
    write_json(config.paths.role_manifest, manifest)
    return manifest


def config_snapshot(config: MultiACCCollabConfig) -> dict[str, Any]:
    base_path = resolve_config_path(config.base_acccollab_config)
    return {
        **asdict(config),
        "resolved_base_acccollab": {
            "path": str(base_path),
            "sha256": file_sha256(base_path),
            "config": asdict(config.base_config()),
        },
        "source_tree": _source_tree_identity(),
    }


def _source_tree_identity() -> dict[str, str]:
    """Hash executable Python sources because this checkout has no Git metadata."""
    roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "scripts" / "acccollab",
        PROJECT_ROOT / "scripts" / "multi_acccollab",
    ]
    result: dict[str, str] = {}
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            result[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = file_sha256(path)
    return result


def _build_dataclass(cls: type[Any], data: Any, path: str) -> Any:
    if not is_dataclass(cls) or not isinstance(data, dict):
        raise MultiACCCollabConfigError(f"{path} must be a mapping")
    field_map = {item.name: item for item in fields(cls)}
    unknown = sorted(set(data) - set(field_map))
    if unknown:
        raise MultiACCCollabConfigError(
            f"Unknown config key(s) under {path}: {', '.join(unknown)}"
        )
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in data:
            if item.default is MISSING and item.default_factory is MISSING:
                raise MultiACCCollabConfigError(f"Missing config key: {path}.{item.name}")
            continue
        hint = hints.get(item.name, Any)
        kwargs[item.name] = _coerce(data[item.name], hint, f"{path}.{item.name}")
    return cls(**kwargs)


def _coerce(value: Any, hint: Any, path: str) -> Any:
    if isinstance(hint, type) and is_dataclass(hint):
        return _build_dataclass(hint, value, path)
    origin = get_origin(hint)
    if origin is list:
        if not isinstance(value, list):
            raise MultiACCCollabConfigError(f"{path} must be a list")
        item_hint = get_args(hint)[0] if get_args(hint) else Any
        return [_coerce(item, item_hint, f"{path}[{index}]") for index, item in enumerate(value)]
    if hint is bool:
        if not isinstance(value, bool):
            raise MultiACCCollabConfigError(f"{path} must be a boolean")
        return value
    if hint is int:
        if isinstance(value, bool):
            raise MultiACCCollabConfigError(f"{path} must be an integer")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise MultiACCCollabConfigError(f"{path} must be an integer") from exc
    if hint is float:
        if isinstance(value, bool):
            raise MultiACCCollabConfigError(f"{path} must be a number")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise MultiACCCollabConfigError(f"{path} must be a number") from exc
    if hint is str:
        if not isinstance(value, str):
            raise MultiACCCollabConfigError(f"{path} must be a string")
        return value
    return value
