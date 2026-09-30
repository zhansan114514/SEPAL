"""Sampling utilities: random, stratified, and full split sampling.

All data sampling MUST go through this module. No script should use
prefix slicing (data[:N]) directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class SplitSamplingConfig:
    """Configuration for sampling a single data split."""

    strategy: str = "full"           # "full" | "random" | "stratified_by_subject"
    max_samples: int | None = None   # None = use all samples
    seed: int = 42
    group_key: str = "subject"       # key for stratified grouping


def sample_split(
    samples: list[dict[str, Any]],
    config: SplitSamplingConfig,
) -> list[dict[str, Any]]:
    """Sample a data split according to the given configuration.

    Args:
        samples: Full list of samples for this split.
        config: Sampling configuration.

    Returns:
        Sampled (or full) list of samples.

    Raises:
        ValueError: If strategy is unknown or max_samples is invalid.
    """
    if not samples:
        return []

    strategy = config.strategy
    max_samples = config.max_samples

    # Full strategy or no limit: return everything
    if strategy == "full" or max_samples is None:
        return list(samples)

    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"max_samples must be positive or None, got {max_samples}")

    if len(samples) <= max_samples:
        return list(samples)

    if strategy == "random":
        return _random_sample(samples, max_samples, config.seed)

    if strategy == "stratified_by_subject":
        return _stratified_sample(
            samples,
            max_samples,
            config.seed,
            group_key=config.group_key,
        )

    raise ValueError(f"Unknown sampling strategy: {strategy}")


def _random_sample(
    samples: list[dict[str, Any]],
    max_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Random sampling: shuffle and take first N."""
    rng = random.Random(seed)
    selected = list(samples)
    rng.shuffle(selected)
    return selected[:max_samples]


def _stratified_sample(
    samples: list[dict[str, Any]],
    max_samples: int,
    seed: int,
    group_key: str = "subject",
) -> list[dict[str, Any]]:
    """Stratified sampling: ensure all groups are represented."""
    rng = random.Random(seed)

    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        key = sample.get(group_key) or "unknown"
        groups.setdefault(key, []).append(sample)

    if len(groups) <= 1:
        return _random_sample(samples, max_samples, seed)

    result: list[dict[str, Any]] = []
    per_group = max(1, max_samples // len(groups))

    for _, group_samples in sorted(groups.items()):
        group_samples = list(group_samples)
        rng.shuffle(group_samples)
        result.extend(group_samples[:per_group])

    # Fill remaining slots from unused samples
    if len(result) < max_samples:
        used = {id(x) for x in result}
        rest = [x for x in samples if id(x) not in used]
        rng.shuffle(rest)
        result.extend(rest[: max_samples - len(result)])

    rng.shuffle(result)
    return result[:max_samples]


def apply_sampling(
    data: dict[str, list[dict[str, Any]]],
    sampling: dict[str, dict[str, Any]],
    base_seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Apply per-split sampling configuration to a dataset bundle.

    Args:
        data: Dict mapping split names to sample lists.
        sampling: Dict mapping split names to sampling config dicts,
                  e.g. {"train": {"strategy": "random", "max_samples": 100, "seed_offset": 0}}.
        base_seed: Base random seed; split seed = base_seed + seed_offset.

    Returns:
        New dict with sampled splits.
    """
    result: dict[str, list[dict[str, Any]]] = {}

    for split_name, samples in data.items():
        split_cfg = sampling.get(split_name, {})
        strategy = split_cfg.get("strategy", "full")
        max_val = split_cfg.get("max_samples")
        seed_offset = split_cfg.get("seed_offset", 0)

        # If strategy is "full" and no max_samples set, skip sampling
        if strategy == "full" and max_val is None:
            result[split_name] = list(samples)
            continue

        cfg = SplitSamplingConfig(
            strategy=strategy,
            max_samples=max_val,
            seed=base_seed + seed_offset,
            group_key=split_cfg.get("group_key", "subject"),
        )
        result[split_name] = sample_split(samples, cfg)

    return result
