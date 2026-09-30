"""Atomic JSON and JSONL helpers for the isolated ACC-Collab workflow."""

from __future__ import annotations

import json
import heapq
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, TypeVar

from src.utils.artifacts import atomic_write_json

_Key = TypeVar("_Key")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one formatted JSON object."""
    atomic_write_json(path, payload)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read one JSON object and reject non-object roots."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Atomically write mappings as JSONL and return the row count."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
                count += 1
        os.replace(temporary, target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL and require every non-empty row to be an object."""
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield validated JSON objects from one JSONL file without buffering it."""
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield row


def count_jsonl(path: str | Path) -> int:
    """Count validated non-empty rows in one JSONL file."""
    return sum(1 for _row in iter_jsonl(path))


def merge_sorted_jsonl(
    paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    key: Callable[[Mapping[str, Any]], _Key],
) -> int:
    """Atomically k-way merge already-sorted JSONL files.

    Each source is validated to be non-decreasing under ``key`` while it is
    consumed. Only one row per source is retained in memory, which is important
    for full ACC-Collab trajectories containing all Monte Carlo simulations.
    Equal keys are stable by source-file order and then by row order.
    """
    sources = [Path(path) for path in paths]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    iterators = [iter_jsonl(path) for path in sources]
    previous_keys: list[_Key | None] = [None] * len(iterators)
    heap: list[tuple[_Key, int, int, dict[str, Any]]] = []
    serial = 0

    def push_next(source_index: int) -> None:
        nonlocal serial
        try:
            row = next(iterators[source_index])
        except StopIteration:
            return
        row_key = key(row)
        previous = previous_keys[source_index]
        if previous is not None and row_key < previous:
            raise ValueError(
                f"JSONL source is not sorted under the requested key: "
                f"{sources[source_index]} ({row_key!r} < {previous!r})"
            )
        previous_keys[source_index] = row_key
        heapq.heappush(heap, (row_key, source_index, serial, row))
        serial += 1

    count = 0
    try:
        for source_index in range(len(iterators)):
            push_next(source_index)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            while heap:
                _row_key, source_index, _serial, row = heapq.heappop(heap)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                push_next(source_index)
        os.replace(temporary, target)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise
    return count
