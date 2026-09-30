"""Dataset selection and deterministic sharding for ACC-Collab."""

from __future__ import annotations

from typing import Any

from src.data.loader import load_dataset
from src.data.sampler import SplitSamplingConfig, sample_split


def load_split_samples(
    *,
    dataset_name: str,
    split: str,
    max_samples: int | None,
    strategy: str,
    seed: int,
    mmlu_load_mode: str = "by_subject",
    expected_samples: int | None = None,
    benchmark_data_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Load, sample, and identify one experiment split.

    Sampling is delegated to :mod:`src.data.sampler`; no prefix slicing is used.
    ``sample_id`` is assigned after sampling so every sharded stage observes the
    same stable order for a fixed config and seed.
    """
    bundle = load_dataset(
        dataset_name,
        seed=seed,
        mmlu_load_mode=mmlu_load_mode,
        benchmark_data_dir=benchmark_data_dir,
    )
    if split not in bundle:
        raise ValueError(f"Split {split!r} is unavailable; choices: {sorted(bundle)}")

    selected = sample_split(
        [dict(sample) for sample in bundle[split]],
        SplitSamplingConfig(strategy=strategy, max_samples=max_samples, seed=seed),
    )
    if expected_samples is not None and len(selected) != int(expected_samples):
        raise RuntimeError(
            f"Dataset split size mismatch for {dataset_name}/{split}: "
            f"expected {expected_samples}, selected {len(selected)}"
        )

    for index, sample in enumerate(selected):
        sample.setdefault("sample_id", f"{dataset_name}_{split}_{index:08d}")
        sample.setdefault("source_split", split)
        sample["acccollab_sample_index"] = index
    return selected


def shard_samples(
    samples: list[dict[str, Any]],
    *,
    shard_idx: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    """Return a deterministic modulo shard while preserving source order."""
    if num_shards <= 0 or shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(
            f"Invalid shard placement: shard_idx={shard_idx}, num_shards={num_shards}"
        )
    return [sample for index, sample in enumerate(samples) if index % num_shards == shard_idx]


def expand_preference_trials(
    samples: list[dict[str, Any]],
    *,
    trials: int,
) -> list[dict[str, Any]]:
    """Repeat each selected question as independently generated trajectories.

    The source set remains unchanged (for example, MMLU validation still has
    1531 unique questions). Each expanded row gets a unique sample id/index so
    checkpointing, sharding, wrong-answer guidance, and pair keys cannot merge
    different stochastic trials accidentally.
    """
    if trials < 1:
        raise ValueError(f"trials must be positive, got {trials}")
    if trials == 1:
        return samples

    expanded: list[dict[str, Any]] = []
    for source_index, source in enumerate(samples):
        source_id = str(source.get("sample_id") or "")
        if not source_id:
            raise ValueError(f"Preference sample at index {source_index} has no sample_id")
        original_index = int(source.get("acccollab_sample_index", source_index))
        for trial_index in range(trials):
            sample = dict(source)
            sample["source_sample_id"] = source_id
            sample["acccollab_source_sample_index"] = original_index
            sample["acccollab_preference_trial"] = trial_index
            sample["acccollab_preference_trials"] = trials
            sample["sample_id"] = f"{source_id}__trajectory_{trial_index:02d}"
            sample["acccollab_sample_index"] = len(expanded)
            expanded.append(sample)
    return expanded


def sample_order(samples: list[dict[str, Any]]) -> dict[str, int]:
    """Map unique sample ids to their source order, rejecting malformed data."""
    order: dict[str, int] = {}
    for index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"Sample at index {index} has no sample_id")
        if sample_id in order:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        order[sample_id] = index
    return order
