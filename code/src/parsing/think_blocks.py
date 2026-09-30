"""Helpers for removing model-internal thinking blocks before parsing."""

from __future__ import annotations

import re


def strip_think_blocks(text: str) -> str:
    """Remove complete and unterminated <think> blocks from generated text."""
    raw = text or ""
    raw = re.sub(r"(?is)<think\b[^>]*>.*?</think\s*>", "\n", raw)
    raw = re.sub(r"(?is)<think\b[^>]*>.*$", "\n", raw)
    raw = re.sub(r"(?is)</think\s*>", "\n", raw)
    return raw.strip()


def has_unclosed_think(text: str) -> bool:
    """True if ``text`` contains a ``<think>`` tag without a matching close tag.

    Used to flag responses that were truncated by ``max_tokens`` while the model
    was still in its reasoning block — these have no final answer and should be
    dropped from downstream data rather than kept as partial content.
    """
    raw = text or ""
    without_complete = re.sub(r"(?is)<think\b[^>]*>.*?</think\s*>", "", raw)
    return bool(re.search(r"(?is)<think\b", without_complete))


_CHOICE_LETTER_TAG_RE = re.compile(r"<\s*(\(?\s*([A-D])\s*\)?)\s*>", re.I)
_ANGLE_SPAN_RE = re.compile(r"<([^\s<>\n](?:[^<>\n]{0,118}[^\s<>\n])?)>")
_PLACEHOLDER_RE = re.compile(
    r"(?is)<\s*/?\s*(?:"
    r"\|endoftext\||"
    r"br\s*/?|eoa|eoc|sep|end|newline|line\s+break|space|"
    r"\.{2,}|"
    r"(?:[a-z][a-z0-9_/-]{1,}|"
    r"[a-z][a-z0-9_/-]*(?:[\s,;:()/+-]+[a-z0-9_/-]+)+)"
    r")\s*/?\s*>"
)
_PLACEHOLDER_HINT_RE = re.compile(
    r"(?i)\b(?:"
    r"answer|application|brief|choice|chosen|clarification|correct|"
    r"description|evidence|example|explanation|format|justification|"
    r"label|letter|needed|option|otherwise|phrase|reason|reasoning|"
    r"selected|selection|sentence|summary|template|topic"
    r")\b"
)
_CONTENT_ANCHOR_RE = re.compile(
    r"(?i)\b(?:"
    r"direct reason|key evidence|application|option analysis|elimination|"
    r"judgement|answer correct|suggested answer|confidence"
    r")\s*:|\bthe final result\s+is\b"
)
_TEMPLATE_LABEL_RE = re.compile(
    r"(?i)\b(?:"
    r"answer format|direct reason|key evidence|application|option analysis|"
    r"elimination|the final result is|the answer is|answer|reasoning|reason|"
    r"explanation|justification|selected option|correct option|correct answer"
    r")\b"
)


def clean_model_response(text: str) -> str:
    """Return the public response text used for parsing and downstream prompts.

    The cleaner intentionally discards hidden thinking blocks instead of using
    them as answer evidence.  It also removes no-think control-token echoes,
    drops template-remnant paragraphs, and collapses adjacent duplicated
    paragraphs commonly produced around stray ``</think>`` tags.
    """
    raw = strip_think_blocks(text)
    if not raw:
        return ""

    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"(?i)<\s*eoa\s*>", "", raw)
    raw = re.sub(r"(?im)^\s*/no_think\s*$", "", raw)
    raw = re.sub(r"(?i)\b/no_think\b", "", raw)

    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n+", raw)
        if part.strip()
    ]
    paragraphs = _drop_template_paragraphs(paragraphs)
    paragraphs = _collapse_adjacent_duplicates(paragraphs)

    cleaned = "\n\n".join(paragraphs) if paragraphs else raw
    cleaned = _drop_placeholder_fragments(cleaned)
    cleaned = _normalize_angle_spans(cleaned)
    cleaned = _drop_empty_answer_claims(cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _drop_template_paragraphs(paragraphs: list[str]) -> list[str]:
    if len(paragraphs) <= 1:
        return paragraphs
    non_template = [p for p in paragraphs if not _is_template_paragraph(p)]
    if not non_template:
        return paragraphs
    return non_template


def _is_template_paragraph(paragraph: str) -> bool:
    if not _has_placeholder_span(paragraph):
        return False
    without_choice_tags = _CHOICE_LETTER_TAG_RE.sub(r"\1", paragraph)
    without_placeholders = _remove_placeholder_spans(without_choice_tags)
    without_labels = _TEMPLATE_LABEL_RE.sub(" ", without_placeholders)
    remainder = re.sub(r"[\W_]+", "", without_labels)
    return not remainder


def _collapse_adjacent_duplicates(paragraphs: list[str]) -> list[str]:
    collapsed: list[str] = []
    for paragraph in paragraphs:
        norm = _normalize_for_duplicate_check(paragraph)
        if collapsed and norm == _normalize_for_duplicate_check(collapsed[-1]):
            continue
        collapsed.append(paragraph)
    return collapsed


def _drop_placeholder_fragments(text: str) -> str:
    """Drop visible template fragments such as ``The answer is <option letter>.``."""
    cleaned_paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", text or ""):
        kept_lines: list[str] = []
        for line in paragraph.splitlines():
            line = _normalize_choice_letter_tags(line)
            if not _has_placeholder_span(line):
                kept_lines.append(line)
                continue

            anchored_line = _content_anchor_after_placeholder(line)
            if anchored_line:
                kept_lines.append(anchored_line)
                continue

            fragments = re.split(r"(?<=[.!?])\s+", line)
            kept = []
            for fragment in fragments:
                fragment = fragment.strip()
                if not fragment:
                    continue
                if not _has_placeholder_span(fragment):
                    kept.append(fragment)
                    continue
                anchored_tail = _content_anchor_after_placeholder(fragment)
                if anchored_tail:
                    kept.append(anchored_tail)
            if kept:
                kept_lines.append(" ".join(kept))
        cleaned = "\n".join(line for line in kept_lines if line.strip()).strip()
        if cleaned:
            cleaned_paragraphs.append(cleaned)
    return "\n\n".join(cleaned_paragraphs)


def _content_anchor_after_placeholder(fragment: str) -> str:
    placeholder = _first_placeholder_match(fragment)
    if not placeholder:
        return fragment.strip()
    anchor = _CONTENT_ANCHOR_RE.search(fragment, placeholder.end())
    if not anchor:
        return ""
    return _normalize_angle_spans(fragment[anchor.start():]).strip()


def _first_placeholder_match(text: str) -> re.Match[str] | None:
    for match in _ANGLE_SPAN_RE.finditer(text or ""):
        if _is_placeholder_angle_span(match):
            return match
    return _PLACEHOLDER_RE.search(text or "")


def _has_placeholder_span(text: str) -> bool:
    return _first_placeholder_match(text) is not None


def _remove_placeholder_spans(text: str) -> str:
    text = _PLACEHOLDER_RE.sub(" ", text or "")
    return _ANGLE_SPAN_RE.sub(
        lambda match: " " if _is_placeholder_angle_span(match) else match.group(0),
        text,
    )


def _normalize_angle_spans(text: str) -> str:
    text = _normalize_choice_letter_tags(text or "")

    def replace(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        if _looks_like_placeholder_tag(content):
            return ""
        return content

    return _ANGLE_SPAN_RE.sub(replace, text)


def _normalize_choice_letter_tags(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        original = match.group(1).strip()
        letter = match.group(2).upper()
        return f"({letter})" if original.startswith("(") else letter

    return _CHOICE_LETTER_TAG_RE.sub(replace, text or "")


def _is_placeholder_angle_span(match: re.Match[str]) -> bool:
    return _looks_like_placeholder_tag(match.group(1).strip())


def _looks_like_placeholder_tag(content: str) -> bool:
    tag = (content or "").strip()
    lower = tag.lower()
    if not lower:
        return True
    if re.fullmatch(r"\(?\s*[a-d]\s*\)?", lower):
        return False
    if (
        "\\" in tag
        or "|" in tag
        or "..." in tag
        or "e.g" in lower
        or lower in {"br", "br /", "eoa", "eoc", "sep", "end", "newline", "line break", "space"}
    ):
        return True
    if _PLACEHOLDER_HINT_RE.search(lower):
        return True
    if re.fullmatch(r"[a-z_/-]+", lower):
        return True
    return False


def _drop_empty_answer_claims(text: str) -> str:
    cleaned_paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", text or ""):
        kept_lines: list[str] = []
        for line in paragraph.splitlines():
            line = re.sub(
                r"(?i)(?:\s+(?:but|so|therefore|maybe|alternatively)\s*,?)?"
                r"\s*\bthe answer is\s*$",
                "",
                line,
            ).rstrip()
            fragments = re.split(r"(?<=[.!?])\s+", line)
            kept = [
                fragment.strip()
                for fragment in fragments
                if fragment.strip()
                and not re.fullmatch(
                    r"(?is)\s*(?:the\s+answer\s+is|answer\s*:?)\s*[.;]?\s*",
                    fragment,
                )
            ]
            if kept:
                kept_lines.append(" ".join(kept))
        cleaned = "\n".join(kept_lines).strip()
        if cleaned:
            cleaned_paragraphs.append(cleaned)
    return "\n\n".join(cleaned_paragraphs)


def _normalize_for_duplicate_check(text: str) -> str:
    text = _normalize_angle_spans(text or "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text
