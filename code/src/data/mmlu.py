"""MMLU-specific dataset loader.

MMLU is special because:
- The training split is called ``auxiliary_train`` (not ``train``).
- ``dev`` contains 5-shot examples per subject (not training data).
- ``validation`` and ``test`` have subject labels when loaded per-subject.
- ``auxiliary_train`` appears in every subject config; loading it once from
  ``all`` avoids duplication.

Internal split mapping:
    train       <- auxiliary_train (loaded once from config ``all``)
    validation  <- validation (per-subject configs, preserving subject)
    test        <- test (per-subject configs, preserving subject)
    dev         <- dev (per-subject configs, preserving subject)
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import DatasetDict, get_dataset_config_names, load_dataset as hf_load_dataset

from src.data.preprocessor import standardize_sample

logger = logging.getLogger(__name__)

MMLU_HF_ID = "cais/mmlu"


def load_mmlu(
    cache_dir: str | None = None,
    load_mode: str = "by_subject",
) -> dict[str, list[dict[str, Any]]]:
    """Load MMLU with proper split mapping.

    Args:
        cache_dir: HuggingFace cache directory.
        load_mode: ``"all"`` loads from the ``all`` config (no subject info).
                   ``"by_subject"`` loads train from ``all`` and
                   validation/test/dev per-subject (preserving subject info).

    Returns:
        Dict with keys ``train``, ``validation``, ``test``, ``dev``.
    """
    if load_mode == "all":
        return _load_mmlu_all(cache_dir=cache_dir)
    if load_mode == "by_subject":
        return _load_mmlu_by_subject(cache_dir=cache_dir)
    raise ValueError(f"Unknown MMLU load_mode: {load_mode}")


def _load_mmlu_all(
    cache_dir: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load all MMLU splits from the ``all`` config."""
    raw = hf_load_dataset(MMLU_HF_ID, "all", cache_dir=cache_dir)

    result = {
        "train": _standardize_mmlu_split(raw, "auxiliary_train", subject="unknown"),
        "validation": _standardize_mmlu_split(raw, "validation", subject="unknown"),
        "test": _standardize_mmlu_split(raw, "test", subject="unknown"),
        "dev": _standardize_mmlu_split(raw, "dev", subject="unknown"),
    }

    _log_mmlu_summary(result)
    return result


def _load_mmlu_by_subject(
    cache_dir: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load MMLU with per-subject metadata.

    - train: from ``all/auxiliary_train`` loaded ONCE.
    - validation/test/dev: from per-subject configs to preserve subject info.
    """
    # 1. auxiliary_train loaded once from "all" config
    raw_all = hf_load_dataset(MMLU_HF_ID, "all", cache_dir=cache_dir)
    train = _standardize_mmlu_split(
        raw_all, "auxiliary_train", subject="auxiliary_train",
    )

    # 2. Collect subject names (exclude "all" and "auxiliary_train")
    subjects = [
        name for name in get_dataset_config_names(MMLU_HF_ID)
        if name not in {"all", "auxiliary_train"}
    ]
    logger.info(f"MMLU: loading {len(subjects)} subject configs for val/test/dev")

    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []

    for subject in subjects:
        try:
            raw_subject = hf_load_dataset(
                MMLU_HF_ID, subject, cache_dir=cache_dir,
            )
        except Exception as e:
            logger.warning(f"Failed to load MMLU subject '{subject}': {e}")
            continue

        validation.extend(
            _standardize_mmlu_split(raw_subject, "validation", subject=subject),
        )
        test.extend(
            _standardize_mmlu_split(raw_subject, "test", subject=subject),
        )
        dev.extend(
            _standardize_mmlu_split(raw_subject, "dev", subject=subject),
        )

    result = {
        "train": train,
        "validation": validation,
        "test": test,
        "dev": dev,
    }

    _log_mmlu_summary(result)
    return result


def _standardize_mmlu_split(
    raw: DatasetDict,
    split_name: str,
    subject: str,
) -> list[dict[str, Any]]:
    """Standardize a single MMLU split, preserving subject metadata."""
    if split_name not in raw:
        return []

    result: list[dict[str, Any]] = []
    for i, sample in enumerate(raw[split_name]):
        sample = dict(sample)
        # Inject metadata before standardization
        sample.setdefault("subject", subject)
        sample["source_split"] = split_name
        sample["source_index"] = i
        sample["dataset"] = "mmlu"

        standardized = standardize_sample(sample, task_type="multiple_choice")
        result.append(standardized)

    return result


def _log_mmlu_summary(result: dict[str, list[dict[str, Any]]]) -> None:
    """Log MMLU split sizes."""
    logger.info(
        "MMLU loaded: train=%d, validation=%d, test=%d, dev=%d",
        len(result.get("train", [])),
        len(result.get("validation", [])),
        len(result.get("test", [])),
        len(result.get("dev", [])),
    )

    # Log subject coverage for splits that have subject info
    for split_name in ("validation", "test"):
        samples = result.get(split_name, [])
        if samples:
            subjects = {s.get("subject", "unknown") for s in samples}
            logger.info(f"  {split_name} subjects: {len(subjects)} unique")
