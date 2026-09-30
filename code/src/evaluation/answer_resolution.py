"""Answer normalization and mixed-task evaluation helpers."""

from __future__ import annotations

from typing import Optional, Sequence

from src.algorithms.reward import math_answers_equal, normalize_answer


_YES_NO_LABELS = {"YES": "YES", "Y": "YES", "NO": "NO", "N": "NO"}
_MULTIPLE_CHOICE_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def normalize_task_answer(answer: Optional[str], task_type: str) -> Optional[str]:
    """Normalize an answer token for voting without collapsing yes/no labels."""
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    if task_type == "math":
        return text

    upper = text.strip("().").strip(".").upper()
    if task_type == "yes_no":
        return _YES_NO_LABELS.get(upper)

    if task_type in {"multiple_choice", "mixed"}:
        if upper in _MULTIPLE_CHOICE_LABELS:
            return upper
        return _YES_NO_LABELS.get(upper) if task_type == "mixed" else None

    return text


def answers_match(pred: Optional[str], label: Optional[str], task_type: str) -> bool:
    """Compare one prediction/label pair using its own task type."""
    pred_text = "" if pred is None else str(pred)
    label_text = "" if label is None else str(label)
    if task_type == "math":
        return math_answers_equal(pred_text, label_text)
    return normalize_answer(pred_text, task_type) == normalize_answer(label_text, task_type)


def compute_accuracy_mixed(
    predictions: Sequence[Optional[str]],
    labels: Sequence[Optional[str]],
    task_types: Sequence[str],
) -> float:
    """Compute accuracy with per-sample task types."""
    n = len(labels)
    if n == 0:
        return 0.0
    correct = sum(
        answers_match(
            predictions[i] if i < len(predictions) else None,
            labels[i],
            task_types[i] if i < len(task_types) else "yes_no",
        )
        for i in range(n)
    )
    return correct / n
