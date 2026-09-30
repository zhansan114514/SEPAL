"""Strict materialization and validation for reusable ablation policy states."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from src.acccollab.config import (
    ACCCollabConfig,
    InitializationConfig,
    load_acccollab_config,
)
from src.acccollab.io import read_json, write_json
from src.utils.artifacts import (
    completed_adapter_path,
    file_sha256,
    path_identity,
    stable_fingerprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLE_NAMES = ("direct", "evidence", "verification")
DATASET_ORDER = ("boolq", "sciq", "bbh", "arc", "mmlu")
POLICY_VARIANTS = ("sft_only", "sft_base_critic", "sft_trained_critic")
POLICY_VARIANT_DATASETS = {
    variant: DATASET_ORDER for variant in POLICY_VARIANTS
}
DATASET_EXPECTED_SAMPLES = {
    "boolq": 3270,
    "sciq": 1000,
    "bbh": 1260,
    "arc": 3548,
    "mmlu": 14042,
}
# These are the logical shard layouts used by the completed four-card main
# experiments. A server with fewer physical GPUs runs the same logical shards
# in waves so sample partitions and per-batch seeds remain comparable.
DATASET_LOGICAL_SHARDS = {
    "boolq": 3,
    "sciq": 1,
    "bbh": 1,
    "arc": 3,
    "mmlu": 4,
}
ABLATION_MANIFEST_VERSION = "multi_acccollab_component_ablations_v1"


@dataclass(frozen=True)
class ModelLayout:
    key: str
    slug: str
    model_type: str
    default_model_path: str
    primary_output_dir: str | None = None
    eval_output_template: str | None = None

    @property
    def primary_multi_output(self) -> Path:
        if self.primary_output_dir is not None:
            return Path(self.primary_output_dir)
        return (
            Path("output/multi_acccollab")
            / f"{self.slug}_mmlu_sft10k"
        )

    def dataset_multi_output(self, dataset_name: str) -> Path:
        if dataset_name == "mmlu":
            return self.primary_multi_output
        if self.eval_output_template is not None:
            return Path(self.eval_output_template.format(dataset=dataset_name))
        return Path("output/multi_acccollab") / f"{self.slug}_eval_{dataset_name}"


MODEL_LAYOUTS = {
    "llama3": ModelLayout(
        key="llama3",
        slug="llama3_8b_instruct",
        model_type="llama3",
        default_model_path="PROJECT_ROOT/models/Meta-Llama-3-8B-Instruct",
    ),
    "gemma2": ModelLayout(
        key="gemma2",
        slug="gemma2_2b_instruct",
        model_type="gemma2",
        default_model_path="PROJECT_ROOT/models/gemma-2-2b-it",
    ),
    "qwen25": ModelLayout(
        key="qwen25",
        slug="qwen2_5_3b_instruct",
        model_type="qwen2.5",
        default_model_path="PROJECT_ROOT/models/Qwen2.5-3B-Instruct",
    ),
    "phi4": ModelLayout(
        key="phi4",
        slug="phi4_mini_instruct",
        model_type="phi4",
        default_model_path="PROJECT_ROOT/models/Phi-4-mini-instruct",
    ),
    "mistral_v03": ModelLayout(
        key="mistral_v03",
        slug="mistral_7b_instruct_v03",
        model_type="mistral_v03",
        default_model_path="PROJECT_ROOT/models/Mistral-7B-Instruct-v0.3",
        primary_output_dir=(
            "output/multi_acccollab/"
            "mistral_7b_instruct_v03_mmlu_sft10k_trials5_singlebos"
        ),
        eval_output_template=(
            "output/multi_acccollab/"
            "mistral_7b_instruct_v03_eval_{dataset}_trials5_singlebos"
        ),
    ),
}


class AblationConfigError(ValueError):
    """Raised when an ablation manifest or source policy is incomplete."""


def materialize_ablation_manifest(
    model_key: str,
    *,
    output_root: str | Path,
    model_path: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Authenticate existing policies and write server-local ablation configs."""
    if model_key not in MODEL_LAYOUTS:
        raise AblationConfigError(
            f"Unknown model {model_key!r}; choices={sorted(MODEL_LAYOUTS)}"
        )
    root = Path(project_root).resolve(strict=False)
    layout = MODEL_LAYOUTS[model_key]
    resolved_model = Path(model_path or layout.default_model_path)
    if not resolved_model.is_absolute():
        resolved_model = root / resolved_model
    if not resolved_model.is_dir():
        raise FileNotFoundError(f"Base model is missing: {resolved_model}")

    output = Path(output_root)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    source_primary = root / layout.primary_multi_output
    sft_registry_path = source_primary / "sft/registry.json"
    if not sft_registry_path.is_file():
        raise FileNotFoundError(f"Completed SFT registry is missing: {sft_registry_path}")
    sft_registry = read_json(sft_registry_path)
    _validate_sft_registry(sft_registry, layout)

    role_policies: dict[str, dict[str, Any]] = {}
    for role in ROLE_NAMES:
        role_info = _require_mapping(
            _require_mapping(sft_registry.get("roles"), "SFT registry roles").get(role),
            f"SFT registry role {role}",
        )
        actor_adapter = _resolve_relocated_artifact(
            role_info.get("adapter"),
            root,
            fallback=source_primary / f"sft/adapters/{role}_adapter",
        )
        _validate_adapter_identity(
            actor_adapter,
            _require_mapping(role_info.get("identity"), f"SFT identity for {role}"),
            label=f"{role} SFT Actor",
        )
        final_registry_path = source_primary / f"roles/{role}/registry/final.json"
        final_registry = read_json(final_registry_path)
        state = _require_mapping(final_registry.get("state"), f"final state for {role}")
        critic_adapter = _resolve_relocated_artifact(
            state.get("critic_adapter"),
            root,
            fallback=(
                source_primary
                / f"roles/{role}/adapters/iteration_01/critic_adapter"
            ),
        )
        _validate_adapter_identity(
            critic_adapter,
            _require_mapping(
                final_registry.get("critic_identity"),
                f"final Critic identity for {role}",
            ),
            label=f"{role} trained Critic",
        )
        role_policies[role] = {
            "sft_actor_adapter": _portable_path(actor_adapter, root),
            "sft_actor_identity": path_identity(actor_adapter, hash_weights=True),
            "trained_critic_adapter": _portable_path(critic_adapter, root),
            "trained_critic_identity": path_identity(critic_adapter, hash_weights=True),
            "source_final_registry": _file_record(final_registry_path, root),
        }

    datasets: dict[str, Any] = {}
    resolved_config_root = output / "resolved_configs"
    for dataset_name in DATASET_ORDER:
        source_dataset = root / layout.dataset_multi_output(dataset_name)
        dataset_roles: dict[str, Any] = {}
        for role in ROLE_NAMES:
            source_config = source_dataset / f"resolved_role_configs/{role}.yaml"
            if not source_config.is_file():
                raise FileNotFoundError(
                    f"Source role config is missing for {dataset_name}/{role}: "
                    f"{source_config}"
                )
            config = load_acccollab_config(str(source_config))
            if config.data.dataset != dataset_name:
                raise AblationConfigError(
                    f"{source_config} uses dataset={config.data.dataset!r}, "
                    f"expected {dataset_name!r}"
                )
            expected = DATASET_EXPECTED_SAMPLES[dataset_name]
            if (
                config.data.eval.expected_samples != expected
                or config.evaluation.trials != 1
            ):
                raise AblationConfigError(
                    f"{source_config} must evaluate exactly {expected} samples once"
                )
            if config.prompt_role.name != role:
                raise AblationConfigError(
                    f"{source_config} prompt role is {config.prompt_role.name!r}, "
                    f"expected {role!r}"
                )
            resolved = replace(
                config,
                model=replace(
                    config.model,
                    name=str(resolved_model),
                    type=layout.model_type,
                ),
                data=replace(
                    config.data,
                    benchmark_data_dir=_server_benchmark_data_dir(config, root),
                ),
                evaluation=replace(config.evaluation, policy_output_dir=None),
                initialization=InitializationConfig(
                    actor_adapter=role_policies[role]["sft_actor_adapter"],
                    critic_adapter=None,
                ),
            )
            resolved.validate()
            config_path = resolved_config_root / dataset_name / f"{role}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config=OmegaConf.create(asdict(resolved)), f=config_path)
            dataset_roles[role] = {
                "config": _file_record(config_path, root),
                "source_config": _file_record(source_config, root),
            }
        datasets[dataset_name] = {
            "expected_samples": DATASET_EXPECTED_SAMPLES[dataset_name],
            "logical_shards": DATASET_LOGICAL_SHARDS[dataset_name],
            "roles": dataset_roles,
        }

    no_sft_configs: dict[str, Any] = {}
    for role in ROLE_NAMES:
        source_config_path = Path(
            str(datasets["mmlu"]["roles"][role]["config"]["path"])
        )
        source_config = load_acccollab_config(str(_resolve_artifact_path(source_config_path, root)))
        no_sft_config = replace(
            source_config,
            run=replace(
                source_config.run,
                name=f"ablation_no_sft_{layout.slug}_{role}",
                output_dir=str(output / f"no_sft/roles/{role}"),
            ),
            evaluation=replace(source_config.evaluation, policy_output_dir=None),
            initialization=InitializationConfig(actor_adapter=None, critic_adapter=None),
        )
        no_sft_config.validate()
        no_sft_path = output / f"no_sft/resolved_role_configs/{role}.yaml"
        no_sft_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=OmegaConf.create(asdict(no_sft_config)), f=no_sft_path)
        no_sft_configs[role] = _file_record(no_sft_path, root)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "implementation_version": ABLATION_MANIFEST_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": {
            "key": layout.key,
            "slug": layout.slug,
            "type": layout.model_type,
            "path": _portable_path(resolved_model, root),
            "identity": _model_identity(resolved_model),
        },
        "source": {
            "primary_multi_output": _portable_path(source_primary, root),
            "sft_registry": _file_record(sft_registry_path, root),
            "role_policies": role_policies,
        },
        "datasets": datasets,
        "variants": {
            "gpu": {
                **{
                    variant: list(datasets)
                    for variant, datasets in POLICY_VARIANT_DATASETS.items()
                },
                "no_sft_full": list(DATASET_ORDER),
            },
            "offline_from_full_records": [
                "actor_dpo_only_round0",
                "deliberation_rounds_0_to_4",
                "per_role",
                "majority",
                "oracle_any_role",
                "pair_agreement",
            ],
        },
        "no_sft_role_configs": no_sft_configs,
        "output_root": _portable_path(output, root),
    }
    payload["fingerprint"] = stable_fingerprint(payload)
    manifest_path = output / "manifest.json"
    write_json(manifest_path, payload)
    return manifest_path


def load_ablation_manifest(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a materialized ablation manifest."""
    manifest_path = Path(path)
    payload = read_json(manifest_path)
    if payload.get("schema_version") != 1:
        raise AblationConfigError("Ablation manifest schema_version must be 1")
    if payload.get("implementation_version") != ABLATION_MANIFEST_VERSION:
        raise AblationConfigError(
            "Ablation manifest implementation version does not match this code"
        )
    expected_fingerprint = payload.get("fingerprint")
    unsigned = {key: value for key, value in payload.items() if key != "fingerprint"}
    if expected_fingerprint != stable_fingerprint(unsigned):
        raise AblationConfigError("Ablation manifest fingerprint is stale or corrupted")
    model = _require_mapping(payload.get("model"), "manifest model")
    if model.get("key") not in MODEL_LAYOUTS:
        raise AblationConfigError("Ablation manifest names an unsupported model")
    datasets = _require_mapping(payload.get("datasets"), "manifest datasets")
    if set(datasets) != set(DATASET_ORDER):
        raise AblationConfigError(
            f"Ablation manifest datasets must be exactly {DATASET_ORDER}"
        )
    for dataset_name in DATASET_ORDER:
        dataset = _require_mapping(datasets.get(dataset_name), dataset_name)
        if int(dataset.get("expected_samples", -1)) != DATASET_EXPECTED_SAMPLES[dataset_name]:
            raise AblationConfigError(f"Unexpected coverage for {dataset_name}")
        if int(dataset.get("logical_shards", -1)) != DATASET_LOGICAL_SHARDS[dataset_name]:
            raise AblationConfigError(f"Unexpected logical shard count for {dataset_name}")
        roles = _require_mapping(dataset.get("roles"), f"{dataset_name} roles")
        if set(roles) != set(ROLE_NAMES):
            raise AblationConfigError(f"Unexpected roles for {dataset_name}")
    return payload


def resolve_manifest_path(value: Any, *, project_root: str | Path = PROJECT_ROOT) -> Path:
    """Resolve one portable manifest path under a server's project root."""
    if not isinstance(value, (str, Path)) or not str(value):
        raise AblationConfigError(f"Manifest artifact path is invalid: {value!r}")
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path


def _validate_sft_registry(payload: Mapping[str, Any], layout: ModelLayout) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("pipeline") != "multi_acccollab"
        or payload.get("stage") != "actor_sft_train"
        or payload.get("status") != "complete"
    ):
        raise AblationConfigError("SFT registry is incomplete or belongs to another pipeline")
    if set(_require_mapping(payload.get("roles"), "SFT roles")) != set(ROLE_NAMES):
        raise AblationConfigError("SFT registry must contain all three roles")
    base_name = str(payload.get("base_model") or "").lower()
    if layout.key == "qwen25" and "qwen2.5" not in base_name:
        raise AblationConfigError("Qwen2.5 SFT registry names the wrong base model")


def _validate_adapter_identity(
    adapter: Path,
    registered: Mapping[str, Any],
    *,
    label: str,
) -> None:
    completed = completed_adapter_path(adapter)
    if not completed:
        raise FileNotFoundError(f"{label} is not a completed adapter: {adapter}")
    actual = path_identity(completed, hash_weights=True)
    registered_selected = _require_mapping(
        registered.get("selected"),
        f"registered selected files for {label}",
    )
    if actual.get("selected") != registered_selected:
        raise AblationConfigError(f"{label} identity does not match its registry")


def _resolve_artifact_path(value: Any, root: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise AblationConfigError(f"Artifact path is invalid: {value!r}")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _resolve_relocated_artifact(
    value: Any,
    root: Path,
    *,
    fallback: Path,
) -> Path:
    registered = _resolve_artifact_path(value, root)
    if registered.exists():
        return registered
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Registered artifact is unavailable and relocation fallback is missing: "
        f"{registered} / {fallback}"
    )


def _server_benchmark_data_dir(config: ACCCollabConfig, root: Path) -> str | None:
    configured = config.data.benchmark_data_dir
    if configured:
        path = Path(configured)
        resolved = path if path.is_absolute() else root / path
        if resolved.is_dir():
            return _portable_path(resolved, root)
    fallback = root / "benchmark_data"
    if fallback.is_dir():
        return _portable_path(fallback, root)
    if configured:
        raise FileNotFoundError(
            f"Benchmark data is unavailable on this server: {configured}"
        )
    return None


def _portable_path(path: str | Path, root: Path) -> str:
    target = Path(path).resolve(strict=False)
    try:
        return str(target.relative_to(root.resolve(strict=False))).replace("\\", "/")
    except ValueError:
        return str(target)


def _file_record(path: str | Path, root: Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    return {
        "path": _portable_path(target, root),
        "sha256": file_sha256(target),
        "size": target.stat().st_size,
    }


def _model_identity(model_dir: Path) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for name in (
        "config.json",
        "tokenizer_config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ):
        path = model_dir / name
        if path.is_file():
            selected[name] = {
                "sha256": file_sha256(path),
                "size": path.stat().st_size,
            }
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"Model has no safetensors weights: {model_dir}")
    return {
        "path": str(model_dir),
        "selected": selected,
        "weight_files": [
            {"name": path.name, "size": path.stat().st_size} for path in weights
        ],
        "total_weight_bytes": sum(path.stat().st_size for path in weights),
    }


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AblationConfigError(f"{label} must be a mapping")
    return value
