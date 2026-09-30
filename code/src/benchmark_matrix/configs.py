"""Build strict, resolved ACC-Collab configs for the three-model matrix."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from omegaconf import OmegaConf

from src.acccollab.config import ACCCollabConfig, ModelConfig, SplitConfig, load_acccollab_config
from src.multi_acccollab.config import (
    MajorityEvaluationConfig,
    MultiACCCollabConfig,
    MultiRunConfig,
    load_multi_acccollab_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACCCOLLAB_TEMPLATE = PROJECT_ROOT / "configs/acccollab/llama3_8b_instruct_mmlu_validation1531.yaml"
MULTI_TEMPLATE = PROJECT_ROOT / "configs/multi_acccollab/llama3_8b_instruct_mmlu_sft10k.yaml"


@dataclass(frozen=True)
class ModelMatrixSpec:
    key: str
    slug: str
    model_path: str
    model_type: str
    datasets: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedModelConfigs:
    spec: ModelMatrixSpec
    original_primary: Path
    original_evaluations: dict[str, Path]
    multi_primary: Path
    multi_evaluations: dict[str, Path]
    original_output_dir: Path
    multi_output_dir: Path


MODEL_SPECS = {
    "llama3": ModelMatrixSpec(
        key="llama3",
        slug="llama3_8b_instruct",
        model_path="PROJECT_ROOT/models/Meta-Llama-3-8B-Instruct",
        model_type="llama3",
        datasets=("bbh", "arc"),
    ),
    "mistral": ModelMatrixSpec(
        key="mistral",
        slug="mistral_7b_instruct_v02",
        model_path="PROJECT_ROOT/models/Mistral-7B-Instruct-v0.2",
        model_type="mistral",
        datasets=("boolq", "sciq", "bbh", "arc", "mmlu"),
    ),
    "mistral_v03": ModelMatrixSpec(
        key="mistral_v03",
        slug="mistral_7b_instruct_v03",
        model_path="PROJECT_ROOT/models/Mistral-7B-Instruct-v0.3",
        model_type="mistral_v03",
        datasets=("boolq", "sciq", "bbh", "arc", "mmlu"),
    ),
    "gemma2": ModelMatrixSpec(
        key="gemma2",
        slug="gemma2_2b_instruct",
        model_path="PROJECT_ROOT/models/gemma-2-2b-it",
        model_type="gemma2",
        datasets=("boolq", "sciq", "bbh", "arc", "mmlu"),
    ),
    "qwen3": ModelMatrixSpec(
        key="qwen3",
        slug="qwen3_4b_instruct_2507",
        model_path="models/Qwen3-4B-Instruct-2507",
        model_type="qwen3",
        datasets=("boolq", "sciq", "bbh", "arc", "mmlu"),
    ),
    "qwen25": ModelMatrixSpec(
        key="qwen25",
        slug="qwen2_5_3b_instruct",
        model_path="PROJECT_ROOT/models/Qwen2.5-3B-Instruct",
        model_type="qwen2.5",
        datasets=("boolq", "sciq", "bbh", "arc", "mmlu"),
    ),
    "phi4mini": ModelMatrixSpec(
        key="phi4mini",
        slug="phi4_mini_instruct",
        model_path="PROJECT_ROOT/models/Phi-4-mini-instruct",
        model_type="phi4",
        datasets=("boolq", "sciq", "bbh", "arc", "mmlu"),
    ),
}


DATASET_SPLITS = {
    "boolq": ("validation", 3270),
    "mmlu": ("test", 14042),
    "bbh": ("test", 1260),
    "sciq": ("test", 1000),
    "arc": ("test", 3548),
}


def materialize_model_configs(
    model_key: str,
    *,
    output_root: str | Path,
    benchmark_data_dir: str | Path,
    devices: Sequence[int] = (0, 1, 2, 3),
) -> ResolvedModelConfigs:
    """Write exact model/dataset configs used by the resumable matrix runner."""
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model key {model_key!r}; choices={sorted(MODEL_SPECS)}")
    selected_devices = tuple(int(device) for device in devices)
    if not selected_devices or any(device < 0 for device in selected_devices):
        raise ValueError("devices must contain non-negative GPU ids")
    if len(set(selected_devices)) != len(selected_devices):
        raise ValueError("devices must not contain duplicates")
    spec = MODEL_SPECS[model_key]
    resolved_root = Path(output_root) / "resolved_configs" / spec.slug
    resolved_root.mkdir(parents=True, exist_ok=True)
    paper_data = str(Path(benchmark_data_dir).resolve(strict=False))

    original_primary_config = _original_primary(
        spec,
        paper_data,
        devices=selected_devices,
    )
    original_primary = resolved_root / "original_mmlu_train.yaml"
    _save_config(original_primary, original_primary_config)

    original_evaluations: dict[str, Path] = {}
    for dataset_name in DATASET_SPLITS:
        if dataset_name == "mmlu":
            original_evaluations[dataset_name] = original_primary
            continue
        config = _original_cross_eval(
            original_primary_config,
            dataset_name=dataset_name,
            policy_output_dir=original_primary_config.run.output_dir,
            paper_data=paper_data,
        )
        path = resolved_root / f"original_eval_{dataset_name}.yaml"
        _save_config(path, config)
        original_evaluations[dataset_name] = path

    multi_template = load_multi_acccollab_config(str(MULTI_TEMPLATE))
    # The controlled single-BOS smoke still yielded only 4.49% Critic pairs.
    # Restore the paper's five independent training trajectories for Mistral;
    # evaluation remains single-trial. Keep a new directory so no legacy
    # double-BOS or one-trajectory checkpoint can be reused.
    training_suffix = (
        "_trials5_singlebos" if spec.model_type.startswith("mistral") else ""
    )
    multi_output = (
        Path("output/multi_acccollab")
        / f"{spec.slug}_mmlu_sft10k{training_suffix}"
    )
    role_configs = [
        replace(role, device=selected_devices[index % len(selected_devices)])
        for index, role in enumerate(multi_template.roles)
    ]
    sft_config = multi_template.sft
    multi_runtime = replace(
        multi_template.runtime,
        sft_generation_devices=list(selected_devices),
    )
    if spec.key == "qwen3":
        sft_config = replace(
            sft_config,
            training=replace(
                sft_config.training,
                # Preserve the effective batch of 16 while removing the
                # avoidable second forward/backward accumulation step.
                batch_size=16,
                gradient_accumulation_steps=1,
            ),
        )
        multi_runtime = replace(
            multi_runtime,
            sft_generation_batch_size=256,
            parallel_role_pipelines=True,
        )
    elif spec.key in {"qwen25", "phi4mini"}:
        sft_config = replace(
            sft_config,
            training=replace(
                sft_config.training,
                # Keep the original effective batch of 16 while avoiding an
                # unnecessary accumulation step on the smaller 3B model.
                batch_size=16,
                gradient_accumulation_steps=1,
            ),
        )
        multi_runtime = replace(
            multi_runtime,
            sft_generation_batch_size=128,
            parallel_role_pipelines=True,
        )
    multi_primary_config = replace(
        multi_template,
        run=MultiRunConfig(
            name=f"multi_acccollab_{spec.slug}_mmlu_sft10k{training_suffix}",
            output_dir=str(multi_output),
            seed=42,
        ),
        base_acccollab_config=str(original_primary),
        roles=role_configs,
        sft=sft_config,
        runtime=multi_runtime,
        evaluation=MajorityEvaluationConfig(),
        profile="primary",
    )
    multi_primary_config.validate()
    multi_primary = resolved_root / "multi_mmlu_train.yaml"
    _save_config(multi_primary, multi_primary_config)

    multi_evaluations: dict[str, Path] = {}
    for dataset_name in DATASET_SPLITS:
        if dataset_name == "mmlu":
            multi_evaluations[dataset_name] = multi_primary
            continue
        base_acccollab_config = original_evaluations[dataset_name]
        if spec.key == "qwen3":
            # Keep completed original cross-evaluation fingerprints stable,
            # but give the not-yet-run three-role evaluations a deeper H100
            # queue.  The role derivation rewrites run/output policy fields,
            # so this private base config cannot overwrite original results.
            fast_base = _qwen_h100_fast_evaluation(
                load_acccollab_config(str(base_acccollab_config))
            )
            base_acccollab_config = resolved_root / f"multi_base_eval_{dataset_name}.yaml"
            _save_config(base_acccollab_config, fast_base)
        multi_config = replace(
            multi_primary_config,
            run=MultiRunConfig(
                name=f"multi_acccollab_{spec.slug}_eval_{dataset_name}",
                output_dir=str(
                    Path("output/multi_acccollab")
                    / f"{spec.slug}_eval_{dataset_name}{training_suffix}"
                ),
                seed=42,
            ),
            base_acccollab_config=str(base_acccollab_config),
            evaluation=MajorityEvaluationConfig(
                policy_source_output_dir=str(multi_output),
            ),
            profile="cross_eval",
        )
        multi_config.validate()
        path = resolved_root / f"multi_eval_{dataset_name}.yaml"
        _save_config(path, multi_config)
        multi_evaluations[dataset_name] = path

    return ResolvedModelConfigs(
        spec=spec,
        original_primary=original_primary,
        original_evaluations=original_evaluations,
        multi_primary=multi_primary,
        multi_evaluations=multi_evaluations,
        original_output_dir=Path(original_primary_config.run.output_dir),
        multi_output_dir=multi_output,
    )


def _original_primary(
    spec: ModelMatrixSpec,
    paper_data: str,
    *,
    devices: Sequence[int],
) -> ACCCollabConfig:
    template = load_acccollab_config(str(ACCCOLLAB_TEMPLATE))
    training_suffix = (
        "_trials5_singlebos" if spec.model_type.startswith("mistral") else ""
    )
    output = (
        Path("output/acccollab")
        / f"{spec.slug}_mmlu_validation1531{training_suffix}"
    )
    runtime = replace(
        template.runtime,
        generation_devices=list(devices),
        training_devices=[int(devices[0])],
    )
    training = template.training
    if spec.key == "qwen3":
        runtime = replace(
            runtime,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            max_num_batched_tokens=65536,
            max_num_seqs=256,
            batch=replace(
                runtime.batch,
                preference_generation=16,
                evaluation=256,
            ),
        )
        training = replace(
            training,
            dpo=replace(
                training.dpo,
                batch_size=4,
                gradient_accumulation_steps=1,
            ),
        )
    elif spec.key in {"qwen25", "phi4mini"}:
        runtime = replace(
            runtime,
            gpu_memory_utilization=0.90,
            max_num_batched_tokens=32768,
            max_num_seqs=256,
            batch=replace(
                runtime.batch,
                preference_generation=16,
                evaluation=128,
            ),
        )
        training = replace(
            training,
            dpo=replace(
                training.dpo,
                batch_size=4,
                gradient_accumulation_steps=1,
            ),
        )
    elif spec.key == "mistral_v03":
        # The three-A800 batch-8 recovery run completed without OOM and roughly
        # doubled preference-generation throughput.  Keep that validated
        # profile in the materialized config so a clean rerun does not fall
        # back to the original conservative batch-4 queue.
        runtime = replace(
            runtime,
            gpu_memory_utilization=0.90,
            max_num_batched_tokens=32768,
            max_num_seqs=256,
            batch=replace(runtime.batch, preference_generation=8),
        )
    config = replace(
        template,
        run=replace(
            template.run,
            name=f"acccollab_{spec.slug}_mmlu_validation1531{training_suffix}",
            output_dir=str(output),
            seed=42,
        ),
        model=ModelConfig(
            name=spec.model_path,
            type=spec.model_type,
            dtype="bfloat16",
            # Phi-4-mini applies LongRoPE beyond its original 4k context and
            # explicitly warns that this can degrade short-task performance.
            max_model_len=4096 if spec.key == "phi4mini" else 8192,
            language_model_only=False,
        ),
        runtime=runtime,
        training=training,
        data=replace(
            template.data,
            benchmark_data_dir=paper_data,
            preference_trials=5 if spec.model_type.startswith("mistral") else 1,
        ),
        evaluation=replace(template.evaluation, trials=1, policy_output_dir=None),
    )
    config.validate()
    return config


def _original_cross_eval(
    primary: ACCCollabConfig,
    *,
    dataset_name: str,
    policy_output_dir: str,
    paper_data: str,
) -> ACCCollabConfig:
    split, expected = DATASET_SPLITS[dataset_name]
    # Keep the output naming human-readable and stable across reruns.
    output = Path("output/acccollab") / f"{Path(policy_output_dir).name}_eval_{dataset_name}"
    runtime = primary.runtime
    if primary.model.type == "qwen3":
        # These four original cross-evaluations may already be complete from
        # the initial batch-64 run.  Preserve their semantic fingerprints so
        # an MMLU throughput tune never invalidates finished measurements.
        runtime = replace(
            runtime,
            max_num_batched_tokens=32768,
            batch=replace(runtime.batch, evaluation=64),
        )
    config = replace(
        primary,
        run=replace(
            primary.run,
            name=f"{primary.run.name}_eval_{dataset_name}",
            output_dir=str(output),
        ),
        data=replace(
            primary.data,
            dataset=dataset_name,
            benchmark_data_dir=paper_data,
            preference_trials=1,
            preference=SplitConfig(
                split=("validation" if dataset_name in {"boolq", "bbh", "arc"} else "train"),
                samples=1,
                strategy="random",
                expected_samples=1,
            ),
            eval=SplitConfig(
                split=split,
                samples=None,
                strategy="full",
                expected_samples=expected,
            ),
        ),
        runtime=runtime,
        evaluation=replace(
            primary.evaluation,
            trials=1,
            policy_output_dir=policy_output_dir,
        ),
    )
    config.validate()
    return config


def _qwen_h100_fast_evaluation(config: ACCCollabConfig) -> ACCCollabConfig:
    runtime = replace(
        config.runtime,
        max_num_batched_tokens=65536,
        batch=replace(config.runtime.batch, evaluation=256),
    )
    tuned = replace(config, runtime=runtime)
    tuned.validate()
    return tuned


def _save_config(path: Path, config: ACCCollabConfig | MultiACCCollabConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(asdict(config)), f=path)
