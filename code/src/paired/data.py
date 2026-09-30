"""Data selection helpers for paired experiments."""

from __future__ import annotations

from typing import Any

from src.data.loader import load_dataset
from src.data.sampler import SplitSamplingConfig, sample_split


def load_split_samples(
    *,
    dataset_name: str,
    split: str,
    max_samples: int | None,
    seed: int,
    strategy: str = "random",
    mmlu_load_mode: str = "by_subject",
    expected_samples: int | None = None,
) -> list[dict[str, Any]]:
    data = load_dataset(
        dataset_name,
        seed=seed,
        mmlu_load_mode=mmlu_load_mode,
    )
    if split not in data:
        raise ValueError(f"Split {split!r} not available. Available: {sorted(data)}")
    selected = sample_split(
        [dict(sample) for sample in data[split]],
        SplitSamplingConfig(strategy=strategy, max_samples=max_samples, seed=seed),
    )
    if expected_samples is not None and len(selected) != int(expected_samples):
        raise RuntimeError(
            f"Dataset split size mismatch for {dataset_name}/{split}: "
            f"expected {expected_samples}, selected {len(selected)}. "
            "This usually means a subject failed to load or the configured sampling changed."
        )
    for idx, sample in enumerate(selected):
        sample.setdefault("sample_id", f"{dataset_name}_{split}_{idx}")
        sample.setdefault("source_split", split)
    return selected
