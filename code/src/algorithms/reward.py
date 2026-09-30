"""Reward and answer-normalization utilities.

Answer extraction lives in :mod:`src.parsing.answer_extractor`.  This module
keeps task normalization, math-answer comparison, and reward deltas.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.parsing.answer_extractor import ExtractedAnswer, extract_answer as _extract_answer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerExtraction:
    """Compatibility-shaped extraction result backed by the new parser."""

    answer: Optional[str]
    source: str
    confidence: float = 0.0
    raw_span: str = ""


def extract_answer(response: str, task_type: str = "yes_no") -> Optional[str]:
    """Extract a task answer from natural model output."""
    return _extract_answer(response, task_type).answer


def extract_answer_with_source(response: str, task_type: str = "yes_no") -> AnswerExtraction:
    """Extract an answer and return parser-source diagnostics."""
    extracted: ExtractedAnswer = _extract_answer(response, task_type)
    return AnswerExtraction(
        answer=extracted.answer,
        source=extracted.source,
        confidence=extracted.confidence,
        raw_span=extracted.raw_span,
    )


def normalize_answer(answer: str, task_type: str = "yes_no") -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    text = str(answer).strip()
    if task_type == "math":
        return _normalize_math_answer(text)

    upper = text.strip("()").strip(".").upper()
    if task_type == "yes_no":
        if upper in {"YES", "Y"}:
            return "Y"
        if upper in {"NO", "N"}:
            return "N"
        return upper[:1]
    if task_type == "mixed":
        if upper in {"YES", "Y"}:
            return "Y"
        if upper in {"NO", "N"}:
            return "N"
    return upper[:1]


def _normalize_math_answer(answer: str) -> str:
    """Normalize a math answer for comparison."""
    text = answer.strip()
    text_match = re.match(r"^\\text\{(.+)\}$", text)
    if text_match:
        text = text_match.group(1).strip()

    frac_match = re.match(r"^\\d?frac\{(.+?)\}\{(.+?)\}$", text)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den != 0:
                result = num / den
                if result == int(result):
                    return str(int(result))
                return str(result)
        except (ValueError, OverflowError):
            pass

    try:
        num = float(text)
        if num == int(num) and "." not in text and "e" not in text.lower():
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        pass

    return re.sub(r"\s+", " ", text)


def math_answers_equal(pred: str, label: str) -> bool:
    """Compare two math answers with a small numeric tolerance."""
    if not pred or not label:
        return pred == label
    norm_pred = _normalize_math_answer(pred.strip())
    norm_label = _normalize_math_answer(label.strip())
    if norm_pred == norm_label:
        return True
    try:
        return abs(float(norm_pred) - float(norm_label)) < 1e-6
    except (ValueError, OverflowError):
        return False
