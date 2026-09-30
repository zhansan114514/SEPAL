"""Data preprocessing utilities: answer extraction, normalization, format standardization."""

from __future__ import annotations

import re
import logging
import random
from typing import Any

from src.parsing.answer_extractor import extract_balanced_braces

logger = logging.getLogger(__name__)


def _standardize_yes_no(sample: dict[str, Any], result: dict[str, Any]) -> None:
    """BoolQ format."""
    result["question"] = sample.get("question", "")
    result["passage"] = sample.get("passage", "")
    answer = sample.get("answer", "")
    result["answer"] = "yes" if answer is True or str(answer).lower() == "true" else "no"
    result["choices"] = ["Yes", "No"]


def _standardize_mc(sample: dict[str, Any], result: dict[str, Any]) -> None:
    """MMLU/SCIQ/ARC format."""
    result["question"] = sample.get("question", "")
    result["passage"] = sample.get("passage", "")

    choices = sample.get("choices", [])
    source_labels: list[str] = []
    if isinstance(choices, dict):
        # ARC format: {"text": [...], "label": ["A", "B", ...]}
        choice_texts = list(choices.get("text", []))
        choice_labels = list(choices.get("label", []))
        result["choices"] = choice_texts
        source_labels = [str(label).strip().upper() for label in choice_labels]
        result["source_choice_labels"] = source_labels
        # ACC-Collab's released ARC wrapper discards source labels and presents
        # choices as A, B, C, ...; numeric ARC answer keys must therefore be
        # mapped by position before scoring.
        result["choice_labels"] = [chr(65 + i) for i in range(len(choice_texts))]
    elif isinstance(choices, list) and choices:
        result["choices"] = choices
        labels = [chr(65 + i) for i in range(len(choices))]
        result["choice_labels"] = labels
    else:
        # No choices provided: SCIQ format (correct_answer + distractor1/2/3)
        correct = sample.get("correct_answer", "")
        distractors = [
            sample.get(f"distractor{i}", "")
            for i in range(1, 4)
            if sample.get(f"distractor{i}")
        ]
        if correct and distractors:
            import random as _rng
            import hashlib
            # Deterministic seed from sample content so the same SCIQ
            # sample always produces the same option order and answer
            # label across processes and runs.
            seed_bytes = hashlib.md5(
                (correct + "".join(distractors)).encode()
            ).digest()
            seed_int = int.from_bytes(seed_bytes[:4], "little")
            _r = _rng.Random(seed_int)
            all_options = [correct] + distractors
            _r.shuffle(all_options)
            result["choices"] = all_options
            result["choice_labels"] = [chr(65 + i) for i in range(len(all_options))]
            correct_idx = all_options.index(correct)
            result["answer"] = chr(65 + correct_idx)

    # Handle answer as index or string (skip if already set by SCIQ handler)
    if not result["answer"]:
        answer = sample.get("answer", sample.get("answerKey", ""))
        if isinstance(answer, int):
            n_choices = len(result["choices"]) if result["choices"] else 4
            result["answer"] = chr(65 + answer) if answer < n_choices else ""
        elif isinstance(answer, str):
            cleaned = answer.upper().strip().strip("()")
            canonical = list(result.get("choice_labels") or [])
            if source_labels and cleaned in source_labels:
                result["answer"] = canonical[source_labels.index(cleaned)]
            elif cleaned in canonical:
                result["answer"] = cleaned
            else:
                choice_texts = [str(choice).strip().upper() for choice in result["choices"]]
                result["answer"] = (
                    canonical[choice_texts.index(cleaned)] if cleaned in choice_texts else cleaned
                )


def _standardize_math(sample: dict[str, Any], result: dict[str, Any]) -> None:
    """MATH/GSM8K format with \boxed{} or "#### number" answers."""
    result["question"] = sample.get("question", sample.get("problem", ""))
    result["passage"] = ""

    raw_answer = sample.get("answer", sample.get("solution", ""))
    if isinstance(raw_answer, str):
        # Try to extract from \boxed{} with balanced brace matching
        boxed_match = re.search(r'\\boxed\{', raw_answer)
        if boxed_match:
            brace_start = boxed_match.end() - 1
            content = extract_balanced_braces(raw_answer, brace_start)
            result["answer"] = content.strip() if content else raw_answer.strip()
        # Try GSM8K "#### number" format
        elif "####" in raw_answer:
            after_hash = raw_answer.split("####")[-1].strip()
            num_match = re.search(r'-?[0-9]+(?:\.[0-9]+)?', after_hash)
            result["answer"] = num_match.group(0) if num_match else after_hash
        else:
            # Extract last number as fallback
            nums = re.findall(r'-?[0-9]+(?:\.[0-9]+)?', raw_answer)
            result["answer"] = nums[-1] if nums else raw_answer.strip()
    else:
        result["answer"] = str(raw_answer).strip()

    result["choices"] = []


def _standardize_mixed(sample: dict[str, Any], result: dict[str, Any]) -> None:
    """BBH format: could be yes/no, multiple choice, or free-form."""
    result["question"] = sample.get("question", sample.get("input", ""))
    result["passage"] = ""

    # Try to extract choices if present
    choices = sample.get("choices", [])
    if isinstance(choices, list) and choices:
        result["choices"] = choices
        result["choice_labels"] = [chr(65 + i) for i in range(len(choices))]
    elif isinstance(choices, dict):
        choice_texts = choices.get("text", [])
        result["choices"] = choice_texts
        result["choice_labels"] = choices.get("label", [chr(65 + i) for i in range(len(choice_texts))])

    # Handle answer
    answer = sample.get("answer", sample.get("target", ""))
    if isinstance(answer, str):
        result["answer"] = answer.strip()
    elif isinstance(answer, bool):
        result["answer"] = "yes" if answer else "no"
        result["choices"] = ["Yes", "No"]
    else:
        result["answer"] = str(answer).strip()


_STANDARDIZERS = {
    "yes_no": _standardize_yes_no,
    "multiple_choice": _standardize_mc,
    "math": _standardize_math,
    "mixed": _standardize_mixed,
}

_DEFAULT_MC_LABELS = ["A", "B", "C", "D"]


def standardize_sample(
    sample: dict[str, Any],
    task_type: str = "yes_no",
) -> dict[str, Any]:
    """
    Convert dataset-specific format to unified format.

    Args:
        sample: Raw sample from a dataset.
        task_type: Type of task.

    Returns:
        Standardized sample with keys: question, passage, answer, choices.
    """
    result = {
        "dataset": sample.get("dataset", ""),
        "source_split": sample.get("source_split", ""),
        "source_index": sample.get("source_index", None),
        "subject": sample.get("subject", ""),
        "category": sample.get("category", ""),
        "question": "",
        "passage": "",
        "answer": "",
        "raw_answer": str(sample.get("answer", sample.get("answerKey", ""))),
        "choices": [],
        "task_type": task_type,
    }

    handler = _STANDARDIZERS.get(task_type)
    if handler is None:
        logger.warning(f"Unknown task_type: {task_type}")
        return result

    handler(sample, result)
    return result


def _wrong_math_answer(correct: str, rng: random.Random) -> str:
    try:
        num = float(correct)
    except ValueError:
        return "0"

    is_int = "." not in correct and "e" not in correct.lower() and num == int(num)
    if abs(num) < 1e-12:
        result = float(rng.choice([1, -1, 2, -2, 3, -3]))
    else:
        strategy = rng.choice(["negate", "perturb_up", "perturb_down", "add_offset"])
        if strategy == "negate":
            result = -num
        elif strategy == "perturb_up":
            result = round(num * rng.uniform(1.1, 1.5), 2)
        elif strategy == "perturb_down":
            result = round(num * rng.uniform(0.5, 0.9), 2)
        else:
            offset = rng.choice([-1, 1, 2, -2, 3, -3])
            result = int(num) + offset if is_int else num + offset

    if is_int:
        return str(int(result)) if result == int(result) else str(result)
    return str(result)


def _wrong_choice_label(correct: str, choices: list[str], rng: random.Random) -> str | None:
    correct_upper = correct.upper()
    if len(correct_upper) != 1 or not "A" <= correct_upper <= "Z":
        return None
    correct_idx = ord(correct_upper) - ord("A")
    wrong_labels = [chr(65 + i) for i in range(len(choices)) if i != correct_idx]
    return rng.choice(wrong_labels) if wrong_labels else None


def _wrong_choice_text(correct: str, choices: list[str], rng: random.Random) -> str | None:
    correct_upper = correct.upper()
    wrong_options = [choice for choice in choices if choice.upper() != correct_upper]
    return rng.choice(wrong_options) if wrong_options else None


def _wrong_default_choice(correct: str, rng: random.Random) -> str:
    wrong = [option for option in _DEFAULT_MC_LABELS if option != correct.upper()]
    if wrong:
        return rng.choice(wrong)
    return "A" if correct.upper() != "A" else "B"


def generate_wrong_answer(
    correct_answer: str,
    choices: list[str] | None = None,
    task_type: str = "multiple_choice",
    rng: random.Random | None = None,
) -> str:
    """
    Generate a wrong answer for guided trajectory (!y).

    For yes/no: flip the answer.
    For multiple choice: pick a random wrong option.
    For math: perturb the numeric answer.

    Args:
        correct_answer: The correct answer.
        choices: Available choices for multiple choice tasks.
        task_type: Task type (yes_no, multiple_choice, math).
        rng: Optional random.Random instance for reproducibility.

    Returns:
        A wrong answer string.
    """
    if rng is None:
        rng = random.Random()

    correct = correct_answer.strip()

    if task_type == "math":
        return _wrong_math_answer(correct, rng)

    if correct.lower() in ("yes", "no"):
        return "no" if correct.lower() == "yes" else "yes"

    if choices:
        return (
            _wrong_choice_label(correct, choices, rng)
            or _wrong_choice_text(correct, choices, rng)
            or _wrong_default_choice(correct, rng)
        )

    return _wrong_default_choice(correct, rng)
