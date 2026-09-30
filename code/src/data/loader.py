"""Dataset loading for BoolQ, MMLU, BBH, SCIQ, ARC benchmarks.

MMLU is routed to its own loader (``src.data.mmlu``) for correct
``auxiliary_train -> train`` mapping and per-subject metadata.
All other datasets use the standard HuggingFace path.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from datasets import load_dataset as hf_load_dataset, DatasetDict

from src.data.preprocessor import standardize_sample

logger = logging.getLogger(__name__)

# Supported datasets with their HuggingFace IDs and task types
DATASET_REGISTRY = {
    "boolq": {
        "hf_id": "google/boolq",
        "task_type": "yes_no",
        "splits": ["train", "validation"],
        "loader": "standard",
    },
    "mmlu": {
        "hf_id": "cais/mmlu",
        "task_type": "multiple_choice",
        "splits": ["auxiliary_train", "validation", "test", "dev"],
        "config_name": "all",
        "loader": "mmlu",
    },
    "bbh": {
        "hf_id": "lukaemon/bbh",
        "task_type": "mixed",
        "splits": None,
        "loader": "bbh",
    },
    "sciq": {
        "hf_id": "allenai/sciq",
        "fallback_hf_ids": ["sciq"],
        "task_type": "multiple_choice",
        "splits": ["train", "validation", "test"],
        "loader": "standard",
    },
    "arc": {
        "hf_id": "allenai/ai2_arc",
        "task_type": "multiple_choice",
        "splits": ["train", "validation", "test"],
        "config_names": ["ARC-Easy", "ARC-Challenge"],
        "loader": "arc",
    },
    "math": {
        "hf_id": "EleutherAI/hendrycks_math",
        "task_type": "math",
        "splits": ["train", "test"],
        "config_name": None,
        "loader": "math",
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "task_type": "math",
        "splits": ["train", "test"],
        "loader": "standard",
    },
}


# The released ACC-Collab implementation evaluates the 22 BBH categories whose
# targets can be represented as multiple-choice or yes/no answers.  The five
# free-form categories are intentionally excluded, matching ``BBHQuestion`` in
# the authors' repository and the paper's roughly-5k-question description.
PAPER_BBH_MULTIPLE_CHOICE_TASKS = (
    "temporal_sequences",
    "logical_deduction_three_objects",
    "date_understanding",
    "hyperbaton",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "snarks",
    "tracking_shuffled_objects_seven_objects",
    "movie_recommendation",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "geometric_shapes",
    "disambiguation_qa",
    "tracking_shuffled_objects_five_objects",
    "ruin_names",
    "tracking_shuffled_objects_three_objects",
    "salient_translation_error_detection",
)
PAPER_BBH_YES_NO_TASKS = (
    "navigate",
    "causal_judgement",
    "boolean_expressions",
    "sports_understanding",
    "web_of_lies",
)
PAPER_BBH_TASKS = PAPER_BBH_MULTIPLE_CHOICE_TASKS + PAPER_BBH_YES_NO_TASKS
PAPER_BBH_TEST_SAMPLES = 1260
PAPER_BBH_VALIDATION_SAMPLES = 500
PAPER_ARC_SPLIT_SIZES = {"train": 3370, "validation": 869, "test": 3548}


def load_dataset(
    name: str,
    split_ratios: Optional[dict[str, float]] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    sampling: Optional[dict[str, dict[str, Any]]] = None,
    mmlu_load_mode: str = "by_subject",
    benchmark_data_dir: Optional[str] = None,
) -> dict[str, list[dict]]:
    """
    Load and standardize a benchmark dataset.

    Args:
        name: Dataset name (boolq, mmlu, bbh, sciq, arc, math, gsm8k).
        split_ratios: Custom split ratios (only for BBH).
        seed: Random seed for reproducibility.
        cache_dir: HuggingFace cache directory.
        sampling: Per-split sampling config, e.g.
            {"train": {"strategy": "random", "max_samples": 100, "seed_offset": 0}}.
        mmlu_load_mode: ``"all"`` or ``"by_subject"`` (default).
        benchmark_data_dir: Optional root containing the paper release's
            ``BBH/`` JSON files and ``ARC/`` parquet files.

    Returns:
        Dictionary with split names as keys, each containing a list of
        standardized samples.

    Raises:
        ValueError: If dataset name is not recognized or data is invalid.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Supported: {list(DATASET_REGISTRY.keys())}"
        )

    config = DATASET_REGISTRY[name]
    task_type = config["task_type"]
    loader_type = config.get("loader", "standard")

    logger.info(f"Loading dataset: {name} (task_type={task_type}, loader={loader_type})")

    # Route to specialized loaders
    if loader_type == "mmlu":
        from src.data.mmlu import load_mmlu
        data = load_mmlu(cache_dir=cache_dir, load_mode=mmlu_load_mode)
    elif loader_type == "math":
        raw = _load_math_all(config["hf_id"], cache_dir)
        data = _standardize_splits(raw, task_type, dataset_name=name)
    elif loader_type == "bbh":
        paper_root = _paper_benchmark_root(benchmark_data_dir)
        if paper_root is not None and (paper_root / "BBH").is_dir():
            raw_bbh = _read_paper_bbh_json(paper_root / "BBH")
        else:
            kwargs = {"path": config["hf_id"]}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            raw_bbh = hf_load_dataset(**kwargs)
        data = _load_bbh(raw_bbh, task_type, split_ratios, seed)
    elif loader_type == "arc":
        paper_root = _paper_benchmark_root(benchmark_data_dir)
        if paper_root is not None and (paper_root / "ARC").is_dir():
            raw = _load_paper_arc_parquet(paper_root / "ARC", cache_dir=cache_dir)
        else:
            raw = _load_arc_easy_and_challenge(config, cache_dir=cache_dir)
        data = _standardize_splits(raw, task_type, dataset_name=name)
    else:
        # Standard datasets
        candidates = [config["hf_id"], *config.get("fallback_hf_ids", [])]
        last_error: Exception | None = None
        raw = None
        for candidate in candidates:
            kwargs = {"path": candidate}
            if config.get("config_name"):
                kwargs["name"] = config["config_name"]
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            try:
                raw = hf_load_dataset(**kwargs)
                if candidate != config["hf_id"]:
                    logger.warning(
                        "Loaded %s from fallback dataset id %s",
                        name,
                        candidate,
                    )
                break
            except Exception as exc:
                last_error = exc
                if candidate == candidates[-1]:
                    raise
                logger.warning(
                    "Could not load %s from %s; trying cached alias: %s",
                    name,
                    candidate,
                    exc,
                )
        if raw is None:
            raise RuntimeError(f"Could not load dataset {name}") from last_error
        data = _standardize_splits(raw, task_type, dataset_name=name)

    # Validate
    validate_dataset_bundle(name, data)

    # Apply sampling if configured
    if sampling:
        from src.data.sampler import apply_sampling
        data = apply_sampling(data, sampling, base_seed=seed)

    # Log final sizes
    for split_name, samples in data.items():
        logger.info(f"  {split_name}: {len(samples)} samples")

    return data


def _standardize_splits(
    raw: DatasetDict,
    task_type: str,
    dataset_name: str = "",
) -> dict[str, list[dict]]:
    """Standardize splits from a standard HuggingFace DatasetDict.

    Injects dataset, source_split, and source_index metadata before
    standardization, consistent with the MMLU loader.
    """
    result = {}
    for split_name in ["train", "validation", "test"]:
        if split_name in raw:
            standardized = []
            for i, sample in enumerate(raw[split_name]):
                sample = dict(sample)
                sample.setdefault("dataset", dataset_name)
                sample["source_split"] = split_name
                sample["source_index"] = i
                standardized.append(standardize_sample(sample, task_type))
            result[split_name] = standardized
    return result


def validate_dataset_bundle(
    name: str,
    data: dict[str, list[dict]],
) -> None:
    """Validate that a loaded dataset has the required splits and is non-empty.

    Raises:
        ValueError: If required splits are missing or empty.
    """
    if name == "mmlu":
        required = ["train", "validation", "test"]
    else:
        required = ["train"]

    for split in required:
        if split not in data:
            raise ValueError(
                f"Dataset '{name}': missing required split '{split}'. "
                f"Available: {list(data.keys())}"
            )
        if not data[split]:
            raise ValueError(
                f"Dataset '{name}': split '{split}' is empty. "
                f"Check that the data source is correct."
            )


def _load_math_all(
    hf_id: str,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """Load all MATH subconfigs and merge into a single DatasetDict."""
    from datasets import concatenate_datasets

    subconfigs = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
        "linear_algebra", "abstract_algebra", "college_mathematics",
        "miscellaneous",
    ]

    merged = {}
    for split in ["train", "test"]:
        parts = []
        for cfg in subconfigs:
            try:
                ds = hf_load_dataset(hf_id, cfg, split=split, cache_dir=cache_dir)
                parts.append(ds)
            except Exception:
                pass
        if parts:
            merged[split] = concatenate_datasets(parts)
            logger.info(f"  MATH {split}: {len(merged[split])} samples from {len(parts)} subconfigs")

    return DatasetDict(merged)


def _load_bbh(
    raw: Mapping[str, Sequence[Mapping[str, Any]]],
    task_type: str,
    split_ratios: Optional[dict[str, float]] = None,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Load BBH with custom train/val/test splits.

    Paper specifies: "roughly 25% and 10% of the questions from each category".
    This means we need to split each BBH task category separately.
    """
    import random

    if not isinstance(raw, Mapping):
        raise TypeError(f"BBH loader expects a task mapping, got {type(raw).__name__}")

    ratios = split_ratios or {"test": 0.25, "validation": 0.10}
    rng = random.Random(seed)

    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    test_samples: list[dict[str, Any]] = []

    task_names = [str(name).removesuffix(".json") for name in raw]
    task_sizes = {
        str(name).removesuffix(".json"): len(list(task_data))
        for name, task_data in raw.items()
    }
    paper_exact = (
        split_ratios is None
        and set(task_names) == set(PAPER_BBH_TASKS)
        and sum(task_sizes.values()) >= PAPER_BBH_TEST_SAMPLES + PAPER_BBH_VALIDATION_SAMPLES
    )
    if paper_exact:
        test_counts = _allocate_proportionally(task_sizes, PAPER_BBH_TEST_SAMPLES)
        remaining_sizes = {
            name: task_sizes[name] - test_counts[name] for name in task_sizes
        }
        val_counts = _allocate_proportionally(
            remaining_sizes,
            PAPER_BBH_VALIDATION_SAMPLES,
        )
    else:
        test_counts = {
            name: int(size * ratios["test"]) for name, size in task_sizes.items()
        }
        val_counts = {
            name: int(size * ratios["validation"]) for name, size in task_sizes.items()
        }

    for raw_task_name, task_data in raw.items():
        task_name = str(raw_task_name).removesuffix(".json")
        task_samples = []
        for source_index, source_sample in enumerate(task_data):
            sample = dict(source_sample)
            sample["bbh_task"] = task_name
            sample["source_index"] = source_index
            task_samples.append(sample)

        rng.shuffle(task_samples)

        n_test = test_counts[task_name]
        n_val = val_counts[task_name]

        task_test = task_samples[:n_test]
        task_val = task_samples[n_test : n_test + n_val]
        task_train = task_samples[n_test + n_val :]

        test_samples.extend(task_test)
        val_samples.extend(task_val)
        train_samples.extend(task_train)

        logger.info(
            f"  Task {task_name}: train={len(task_train)}, "
            f"val={len(task_val)}, test={len(task_test)}"
        )

    result = {}
    for split_name, samples in [
        ("test", test_samples),
        ("validation", val_samples),
        ("train", train_samples),
    ]:
        standardized = []
        for i, sample in enumerate(samples):
            sample.setdefault("dataset", "bbh")
            sample["source_split"] = split_name
            sample["source_index"] = i
            # Preserve bbh_task as subject for per-group analysis
            sample.setdefault("subject", sample.get("bbh_task", "unknown"))
            standardized.append(_standardize_bbh_sample(sample, fallback_task_type=task_type))
        result[split_name] = standardized

    logger.info(
        f"  BBH total split: train={len(result['train'])}, "
        f"val={len(result['validation'])}, test={len(result['test'])}"
    )

    return result


def _paper_benchmark_root(configured: Optional[str]) -> Path | None:
    """Resolve an explicit paper-data root without silently using other files."""
    if not configured:
        return None
    root = Path(configured).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise FileNotFoundError(f"benchmark_data_dir does not exist: {root}")
    return root


def _read_paper_bbh_json(directory: Path) -> dict[str, list[dict[str, Any]]]:
    """Read the exact 22 categories supported by the ACC-Collab release."""
    result: dict[str, list[dict[str, Any]]] = {}
    for task_name in PAPER_BBH_TASKS:
        path = directory / f"{task_name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ACC-Collab BBH category: {path}")
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        examples = payload.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"Invalid or empty BBH category file: {path}")
        result[task_name] = [dict(example) for example in examples]
    return result


def _parse_bbh_options(text: str) -> tuple[str, list[str], list[str]]:
    """Split a released BBH multiple-choice input into question/options."""
    marker = "\nOptions:\n"
    if marker not in text:
        raise ValueError("BBH multiple-choice input is missing the Options section")
    question, option_block = text.rsplit(marker, 1)
    matches = list(re.finditer(r"(?m)^\(([A-R])\)\s*", option_block))
    if not matches:
        raise ValueError("BBH multiple-choice input has no parsed options")
    labels = [match.group(1) for match in matches]
    choices = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(option_block)
        choices.append(option_block[match.end():end].strip())
    return question.strip(), choices, labels


def _standardize_bbh_sample(
    sample: Mapping[str, Any],
    *,
    fallback_task_type: str,
) -> dict[str, Any]:
    """Reproduce the release's per-category BBH prompt/answer representation."""
    raw = dict(sample)
    task_name = str(raw.get("bbh_task") or "").removesuffix(".json")
    raw["dataset"] = "bbh"
    raw["subject"] = task_name
    raw_input = str(raw.get("input", raw.get("question", "")))
    target = str(raw.get("target", raw.get("answer", ""))).strip()
    raw["bbh_raw_target"] = target
    raw["bbh_source_index"] = raw.get("source_index")

    if task_name in PAPER_BBH_MULTIPLE_CHOICE_TASKS:
        question, choices, labels = _parse_bbh_options(raw_input)
        target_match = re.fullmatch(r"\(([A-R])\)", target.upper())
        # Three records in the released BBH JSON have malformed comma-bearing
        # targets instead of an option label.  Preserve them for provenance but
        # make them explicitly unscorable rather than silently taking the first
        # letter of the free-text target.
        normalized_target = target_match.group(1) if target_match else "__UNSCORABLE__"
        raw.update(
            question=question,
            choices={"text": choices, "label": labels},
            answer=normalized_target,
        )
        standardized = standardize_sample(raw, "multiple_choice")
        standardized["bbh_raw_target"] = target
        standardized["bbh_source_index"] = raw.get("bbh_source_index")
        if len(choices) < 2:
            standardized["bbh_source_warning"] = "fewer_than_two_options"
        return standardized

    if task_name in PAPER_BBH_YES_NO_TASKS:
        question = raw_input
        if task_name == "web_of_lies" and question.startswith("Question: "):
            question = question[len("Question: "):]
        if task_name == "boolean_expressions":
            expression = question[:-3] if question.endswith(" is") else question
            question = f"Is the following boolean expression True? {expression}"
        lowered = target.lower()
        if lowered in {"yes", "true"}:
            answer = True
        elif lowered in {"no", "false"}:
            answer = False
        else:
            raise ValueError(f"Unexpected BBH yes/no target {target!r} for {task_name}")
        raw.update(question=question, passage="", answer=answer)
        return standardize_sample(raw, "yes_no")

    return standardize_sample(raw, fallback_task_type)


def _allocate_proportionally(
    sizes: Mapping[str, int],
    target_total: int,
) -> dict[str, int]:
    """Allocate an exact total with the largest-remainder method."""
    total = sum(int(size) for size in sizes.values())
    if target_total < 0 or target_total > total:
        raise ValueError(f"Cannot allocate {target_total} samples from {total}")
    if total == 0:
        return {name: 0 for name in sizes}
    quotas = {name: int(size) * target_total / total for name, size in sizes.items()}
    allocated = {name: int(quota) for name, quota in quotas.items()}
    remaining = target_total - sum(allocated.values())
    order = sorted(
        sizes,
        key=lambda name: (-(quotas[name] - allocated[name]), name),
    )
    for name in order[:remaining]:
        allocated[name] += 1
    return allocated


def _load_paper_arc_parquet(directory: Path, cache_dir: Optional[str]) -> DatasetDict:
    """Load Easy+Challenge with the paper's explicit train/validation/test splits."""
    data_files: dict[str, list[str]] = {}
    suffixes = {"ARC-Easy": "2", "ARC-Challenge": "3"}
    for split in ("train", "validation", "test"):
        paths = [
            directory / f"{split}-00000-of-00001-{suffixes[name]}.parquet"
            for name in ("ARC-Easy", "ARC-Challenge")
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing ACC-Collab ARC parquet file(s): {missing}")
        data_files[split] = [str(path) for path in paths]
    kwargs: dict[str, Any] = {"path": "parquet", "data_files": data_files}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    raw: DatasetDict = hf_load_dataset(**kwargs)
    actual = {split: len(raw[split]) for split in PAPER_ARC_SPLIT_SIZES}
    if actual != PAPER_ARC_SPLIT_SIZES:
        raise RuntimeError(
            f"ACC-Collab ARC split-size mismatch: expected {PAPER_ARC_SPLIT_SIZES}, got {actual}"
        )
    return raw


def _load_arc_easy_and_challenge(config: Mapping[str, Any], cache_dir: Optional[str]) -> DatasetDict:
    """Load and concatenate ARC-Easy and ARC-Challenge by official split."""
    from datasets import concatenate_datasets

    parts = []
    for config_name in config["config_names"]:
        kwargs: dict[str, Any] = {"path": config["hf_id"], "name": config_name}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        parts.append(hf_load_dataset(**kwargs))
    merged = {
        split: concatenate_datasets([part[split] for part in parts])
        for split in ("train", "validation", "test")
    }
    return DatasetDict(merged)
