"""Answer extraction for natural Actor responses.

Actors are asked to reason naturally and place a light answer anchor near the
end of the response.  This module makes extraction robust without turning
format obedience into the experiment target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from src.parsing.think_blocks import clean_model_response


AnswerSource = Literal[
    "final_result",
    "final_answer",
    "boxed",
    "tail_claim",
    "weak_tail",
    "none",
]


SOURCE_CONFIDENCE: dict[AnswerSource, float] = {
    "final_result": 1.00,
    "final_answer": 0.95,
    "boxed": 0.95,
    "tail_claim": 0.80,
    "weak_tail": 0.50,
    "none": 0.00,
}


@dataclass(frozen=True)
class ExtractedAnswer:
    """Structured answer extraction result."""

    answer: Optional[str]
    source: AnswerSource
    confidence: float
    raw_span: str = ""


FINAL_RESULT_MC = [
    re.compile(r"(?i)\bthe\s+final\s+result\s+is\s*\(?\s*([A-R])\s*\)?\b"),
]

ANSWER_ANCHOR_MC = [
    re.compile(r"(?i)\bfinal\s+answer\s*[:：]\s*\(?\s*([A-R])\s*\)?\b"),
    # "The final answer is (X)" / "The final answer is:\n\n(X)" -- extremely common
    # phrasing that the "final RESULT is" anchor above does not match.
    re.compile(r"(?i)\bfinal\s+answer\s+is\s*[:：.]?\s*\.{0,3}\s*\(?\s*\(?\s*([A-R])\s*\)?\s*\)?\b"),
]

TAIL_CLAIM_MC = [
    re.compile(r"(?i)\banswer\s*[:：]\s*\(?\s*([A-R])\s*\)?\b"),
    re.compile(r"(?i)\bthe\s+(?:(?:correct|best|right|final|likely)\s+)?(?:answer|option|choice|diagnosis|conclusion)\s+is\s*[:：.]?\s*\.{0,3}\s*\(?\s*\(?\s*([A-R])\s*\)?\s*\)?\b"),
    re.compile(r"(?i)\b(?:correct|best)\s+(?:answer|option|choice)\s+is\s*[:：.]?\s*\.{0,3}\s*\(?\s*([A-R])\s*\)?\b"),
    # "the most likely diagnosis is (X)" / "my conclusion is (X)" (no "the answer" noun).
    re.compile(r"(?i)\b(?:diagnosis|conclusion)\s+is\s*[:：.]?\s*\.{0,3}\s*\(?\s*([A-R])\b"),
    re.compile(r"(?i)\btherefore\s*,?\s*(?:the\s+)?(?:answer|option|choice)\s+is\s*\(?\s*\(?\s*([A-R])\s*\)?\s*\)?\b"),
    re.compile(r"(?i)\b(?:therefore|thus|so|hence|overall|finally)\s*,?\s*\(?\s*([A-R])\s*\)?\s*(?:is\s+(?:correct|right))?\.?\s*$"),
    # "I choose/I'd choose/I'll choose/I'm going to choose (X)" -- contractions use an
    # apostrophe after "I" (not whitespace), so the subject/modal must allow both.
    re.compile(r"(?i)\bI(?:\s+(?:would\s+|will\s+|am\s+)?(?:going\s+to\s+)?|'(?:d|ll|m)\s+(?:going\s+to\s+)?)(?:choose|select|pick)\s*(?:option\s+)?[:：.]?\s*\.{0,3}\s*\(?\s*([A-R])\b"),
    re.compile(r"(?i)\b(?:going\s+with|go\s+with)\s*[:：.]?\s*\.{0,3}\s*\(?\s*([A-R])\b"),
    re.compile(r"(?i)\bmy\s+(?:answer|choice|selection)\s+(?:is|would\s+be)\s*[:：.]?\s*\.{0,3}\s*\(?\s*([A-R])\b"),
    re.compile(r"(?i)\boption\s*\(?\s*([A-R])\s*\)?\s+is\s+(?:correct|right)\b"),
    re.compile(r"(?i)\(\s*([A-R])\s*\)\s+is\s+the\s+(?:right|correct)\s+answer\b"),
    re.compile(r"(?i)\b(?:correct|best)\s+(?:answer|option|choice)\s*[:：]?\s*\(?\s*([A-R])\s*\)?\b"),
]

WEAK_TAIL_MC = [
    re.compile(r"(?im)^\s*\(?\s*([A-R])\s*\)?\s*\.?\s*$"),
    re.compile(r"(?i)\boption\s*\(?\s*([A-R])\s*\)?\s*\.?\s*$"),
    # Trailing "(X)." at the end of a line (e.g. "...the Quakers (C).").
    re.compile(r"(?im)\(([A-R])\)\s*\.?\s*$"),
    re.compile(r"(?i)^\s*\(?\s*([A-R])\s*\)?\s+is\s+the\s+(?:right|correct)\s+answer\b"),
]

FINAL_RESULT_YN = [
    re.compile(r"(?i)\bthe\s+final\s+result\s+is\s*(yes|no)\b"),
]

ANSWER_ANCHOR_YN = [
    re.compile(
        r"(?i)\bfinal\s+answer\s*(?:[:：]|is\s*[:：.]?)\s*"
        r"(?:\(?[AB]\s*\)?\s*[:：.-]?\s*)?(yes|no)\b"
    ),
]

TAIL_CLAIM_YN = [
    # Peer debate often emits the explicit binary pair without an "Answer"
    # prefix, for example "A: Yes", "B (No)", or "... I conclude: B) No".
    re.compile(r"(?i)\b[AB]\s*(?:[):：.\-]\s*|\(\s*)(yes|no)\s*\)?\b"),
    re.compile(
        r"(?i)\banswer\s*[:：]\s*(?:\(?[AB]\s*\)?\s*[:：.-]?\s*)?(yes|no)\b"
    ),
    re.compile(
        r"(?i)\b(?:the|my)\s+(?:correct\s+)?answer\s+is\s*[:：.]?\s*"
        r"(?:\(?[AB]\s*\)?\s*[:：.-]?\s*)?(yes|no)\b"
    ),
    re.compile(
        r"(?i)\b(?:the|my)\s+(?:correct\s+)?answer\s+to\s+[^\n]{1,200}?"
        r"\s+is\s*[:：.]?\s*(?:\(?[AB]\s*\)?\s*[:：.-]?\s*)?(yes|no)\b"
    ),
    re.compile(r"(?i)\btherefore\s*,?\s*(?:the\s+)?answer\s+is\s*(yes|no)\b"),
    re.compile(r"(?i)\b(?:actually|finally|in\s+the\s+end)\s+(yes|no)\b"),
    re.compile(r"(?i)\bI\s+(?:choose|answer)\s*(yes|no)\b"),
]

WEAK_TAIL_YN = [
    re.compile(r"(?im)^\s*(yes|no)\s*[.!?。！？]?\s*$"),
    # Instruction-tuned models (including Qwen3) commonly put the requested
    # answer first and continue the justification on the same line:
    # "No. The passage states ...".
    # Require answer punctuation so ordinary prose containing yes/no is not
    # promoted to a weak answer merely because it begins a tail line.
    re.compile(r"(?im)^\s*(yes|no)\s*(?:[.!?。！？,，;:：]|[-–—])\s*.+$"),
]

# Some models adopt the de-facto binary-choice convention A=Yes/B=No during
# peer debate even though the prompt requests the words Yes/No.  These
# fallbacks are deliberately restricted to explicit answer anchors or a bare
# final line.  They are evaluated only after every textual Yes/No rule above,
# so an explicit word always wins if a response contains conflicting signals.
BINARY_FINAL_RESULT_YN = [
    re.compile(r"(?i)\bthe\s+final\s+result\s+is\s*\(?\s*([AB])\s*\)?\b"),
]

BINARY_ANSWER_ANCHOR_YN = [
    re.compile(
        r"(?i)\bfinal\s+answer\s*(?:[:：]|is\s*[:：.]?)\s*\(?\s*([AB])\s*\)?\b"
    ),
]

BINARY_TAIL_CLAIM_YN = [
    re.compile(r"(?i)\banswer\s*[:：]\s*\(?\s*([AB])\s*\)?\b"),
    re.compile(
        r"(?i)\b(?:the|my)\s+(?:correct\s+)?answer\s+is\s*[:：.]?"
        r"\s*\(?\s*([AB])\s*\)?\b"
    ),
    re.compile(r"(?i)\bI\s+(?:choose|answer)\s*(?:option\s+)?\(?\s*([AB])\s*\)?\b"),
]

BINARY_WEAK_TAIL_YN = [
    re.compile(r"(?im)^\s*\(?\s*([AB])\s*\)?\s*[.!?。！？]?\s*$"),
    # A bare binary label followed by its justification, e.g.
    # "B: The passage states the opposite.".
    re.compile(r"(?im)^\s*\(?\s*([AB])\s*\)?\s*[:：]\s*\S.+$"),
]

FINAL_RESULT_MATH = [
    re.compile(r"(?i)\bthe\s+final\s+result\s+is\s+(.+?)(?:\n|$)"),
]

ANSWER_ANCHOR_MATH = [
    re.compile(r"(?i)\bfinal\s+answer\s*[:：]\s*(.+?)(?:\n|$)"),
    re.compile(r"(?i)\bfinal\s+answer\s+is\s+(.+?)(?:\n|$)"),
]

TAIL_CLAIM_MATH = [
    re.compile(r"(?i)\banswer\s*[:：]\s*(.+?)(?:\n|$)"),
    re.compile(r"(?i)\btherefore\s*,?\s*(?:the\s+)?(?:answer|result)\s+is\s+(.+?)(?:\n|$)"),
    re.compile(r"(?i)\bthe\s+(?:answer|result)\s+is\s+(.+?)(?:\n|$)"),
    re.compile(r"=\s*(-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?)\s*\.?\s*$"),
]


def extract_answer(text: str, task_type: str = "yes_no") -> ExtractedAnswer:
    """Extract an answer token with source and parse-confidence metadata."""
    if not text or not text.strip():
        return _none()
    text = clean_model_response(text)
    if not text:
        return _none()

    if task_type == "multiple_choice":
        return extract_mc_answer(text)
    if task_type == "yes_no":
        return extract_yes_no_answer(text)
    if task_type == "math":
        return extract_math_answer(text)
    if task_type == "mixed":
        first = extract_mc_answer(text)
        if first.answer is not None:
            return first
        return extract_yes_no_answer(text)
    return _none()


def extract_answer_with_source(text: str, task_type: str = "yes_no") -> ExtractedAnswer:
    """Alias kept for callers that expect a source-bearing extraction result."""
    return extract_answer(text, task_type)


def extract_mc_answer(text: str) -> ExtractedAnswer:
    """Extract an A-R label from a multiple-choice response."""
    text = _normalize_answer_markup(_normalize_colons(text))
    found = _last_match(text, FINAL_RESULT_MC)
    if found:
        answer, span = found
        return _result(answer.upper(), "final_result", span)

    found = _last_match(text, ANSWER_ANCHOR_MC)
    if found:
        answer, span = found
        return _result(answer.upper(), "final_answer", span)

    tail = get_tail(text)
    found = _last_match(tail, TAIL_CLAIM_MC)
    if found:
        answer, span = found
        return _result(answer.upper(), "tail_claim", span)

    found = _last_match(tail, WEAK_TAIL_MC)
    if found:
        answer, span = found
        return _result(answer.upper(), "weak_tail", span)

    return _none()


def extract_yes_no_answer(text: str) -> ExtractedAnswer:
    """Extract a yes/no answer."""
    text = _normalize_answer_markup(_normalize_colons(text))
    found = _last_match(text, FINAL_RESULT_YN)
    if found:
        answer, span = found
        return _result(_normalize_yes_no(answer), "final_result", span)

    found = _last_match(text, ANSWER_ANCHOR_YN)
    if found:
        answer, span = found
        return _result(_normalize_yes_no(answer), "final_answer", span)

    tail = get_tail(text)
    found = _last_match(tail, TAIL_CLAIM_YN)
    if found:
        answer, span = found
        return _result(_normalize_yes_no(answer), "tail_claim", span)

    found = _last_match(tail, WEAK_TAIL_YN)
    if found:
        answer, span = found
        return _result(_normalize_yes_no(answer), "weak_tail", span)

    found = _last_match(text, BINARY_FINAL_RESULT_YN)
    if found:
        answer, span = found
        return _result(_normalize_binary_yes_no(answer), "final_result", span)

    found = _last_match(text, BINARY_ANSWER_ANCHOR_YN)
    if found:
        answer, span = found
        return _result(_normalize_binary_yes_no(answer), "final_answer", span)

    found = _last_match(tail, BINARY_TAIL_CLAIM_YN)
    if found:
        answer, span = found
        return _result(_normalize_binary_yes_no(answer), "tail_claim", span)

    found = _last_match(tail, BINARY_WEAK_TAIL_YN)
    if found:
        answer, span = found
        return _result(_normalize_binary_yes_no(answer), "weak_tail", span)

    return _none()


def extract_math_answer(text: str) -> ExtractedAnswer:
    """Extract a mathematical final result."""
    text = _normalize_colons(text)
    found = _last_match(text, FINAL_RESULT_MATH)
    if found:
        answer, span = found
        return _result(_clean_math_answer(answer), "final_result", span)

    boxed = _last_boxed(text)
    if boxed:
        answer, span = boxed
        return _result(answer.strip(), "boxed", span)

    found = _last_match(text, ANSWER_ANCHOR_MATH)
    if found:
        answer, span = found
        return _result(_clean_math_answer(answer), "final_answer", span)

    tail = get_tail(text)
    found = _last_match(tail, TAIL_CLAIM_MATH)
    if found:
        answer, span = found
        return _result(_clean_math_answer(answer), "tail_claim", span)

    weak = _last_numeric_tail(tail)
    if weak:
        answer, span = weak
        return _result(answer, "weak_tail", span)

    return _none()


def get_tail(text: str, n: int = 8) -> str:
    """Return the last non-empty lines for weak extraction."""
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    return "\n".join(lines[-n:])


def extract_balanced_braces(text: str, start: int) -> Optional[str]:
    """Extract content inside balanced curly braces starting at an opening brace."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:idx]
    return None


def _last_boxed(text: str) -> Optional[tuple[str, str]]:
    prefixes = [
        re.compile(r"\\boxed\s*\{"),
        re.compile(r"boxed\s*\{", re.IGNORECASE),
        re.compile(r"\{\\boxed\s*\{"),
    ]
    last: Optional[tuple[int, str, str]] = None
    for prefix in prefixes:
        for match in prefix.finditer(text):
            brace_start = match.end() - 1
            content = extract_balanced_braces(text, brace_start)
            if content is None:
                continue
            end = brace_start + len(content) + 2
            span = text[match.start():end]
            last = (match.start(), content, span)
    if last is None:
        return None
    return last[1], last[2]


def _last_numeric_tail(tail: str) -> Optional[tuple[str, str]]:
    patterns = [
        re.compile(r"(?im)^\s*(-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?)\s*\.?\s*$"),
        re.compile(r"(?i)(?:therefore|thus|so|equals?|is)\s+(-?[0-9]+(?:\.[0-9]+)?)\s*\.?\s*$"),
        re.compile(r"(?i)(?:therefore|thus)\s+(-?[0-9]+(?:\.[0-9]+)?)\s*\.?\s*$"),
        re.compile(r"(?i)^=\s*(-?[0-9]+(?:\.[0-9]+)?)\s*\.?\s*$"),
    ]
    return _last_match(tail, patterns)


def _last_match(text: str, patterns: list[re.Pattern[str]]) -> Optional[tuple[str, str]]:
    last: Optional[tuple[int, str, str]] = None
    for pattern in patterns:
        for match in pattern.finditer(text):
            if not match.groups():
                continue
            answer = match.group(1)
            span = match.group(0)
            if last is None or match.start() >= last[0]:
                last = (match.start(), answer, span)
    if last is None:
        return None
    return last[1], last[2]


def _result(answer: Optional[str], source: AnswerSource, raw_span: str) -> ExtractedAnswer:
    cleaned = _strip_answer_punctuation(answer)
    if not cleaned:
        return _none()
    return ExtractedAnswer(
        answer=cleaned,
        source=source,
        confidence=SOURCE_CONFIDENCE[source],
        raw_span=raw_span.strip(),
    )


def _none() -> ExtractedAnswer:
    return ExtractedAnswer(None, "none", SOURCE_CONFIDENCE["none"], "")


def _normalize_colons(text: str) -> str:
    return (text or "").replace("\uff1a", ":")


def _normalize_answer_markup(text: str) -> str:
    """Remove Markdown emphasis that can split an otherwise explicit anchor."""
    return (text or "").replace("**", "").replace("__", "").replace("*", "")


def _normalize_yes_no(answer: str) -> str:
    return "YES" if str(answer).strip().lower().startswith("y") else "NO"


def _normalize_binary_yes_no(answer: str) -> str:
    return "YES" if str(answer).strip().upper() == "A" else "NO"


def _strip_answer_punctuation(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    text = str(answer).strip()
    text = text.strip().rstrip(".。")
    return text.strip()


def _clean_math_answer(answer: str) -> str:
    text = _strip_answer_punctuation(answer) or ""
    # Avoid swallowing trailing explanation after the answer anchor.
    text = re.split(r"\s+(?:because|since|as|with)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    numeric = re.match(r"^\s*(-?(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?)\b", text)
    if numeric:
        return numeric.group(1)
    return text.strip()
