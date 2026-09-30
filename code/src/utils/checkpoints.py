"""Atomic, batch-granular checkpoints for long-running data generation stages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


class CheckpointMismatchError(RuntimeError):
    """Raised when an existing checkpoint belongs to a different run layout."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def _validate_jsonl(path: Path, expected_count: int) -> bool:
    """Return whether a JSONL file is readable and has the declared row count."""
    if not path.is_file() or expected_count < 0:
        return False
    count = 0
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    return False
                count += 1
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return count == expected_count


def checkpoint_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a stable fingerprint for settings that determine checkpoint output."""
    encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class JsonlBatchCheckpoint:
    """Store each generated input batch as an independently committed checkpoint.

    A batch becomes resumable only after its directory is atomically renamed
    into place. On every resume, the success marker, manifest, logical JSONL
    files, JSON validity, and declared row counts are checked. A killed or
    corrupted batch is therefore regenerated rather than mistaken for a valid
    result.
    """

    SCHEMA_VERSION = 2
    MANIFEST_SCHEMA_VERSION = 1
    _BATCH_RE = re.compile(r"^batch-(\d{8})$")

    def __init__(
        self,
        *,
        output_dir: str | Path,
        stage: str,
        shard_idx: int = 0,
        num_shards: int = 1,
        fingerprint: str,
    ) -> None:
        if shard_idx < 0 or num_shards <= 0 or shard_idx >= num_shards:
            raise ValueError(
                f"Invalid shard placement: shard_idx={shard_idx}, num_shards={num_shards}"
            )
        self.stage = stage
        self.shard_idx = shard_idx
        self.num_shards = num_shards
        self.fingerprint = fingerprint
        self.root = (
            Path(output_dir)
            / "checkpoints"
            / stage
            / f"shard-{shard_idx:03d}-of-{num_shards:03d}"
        )
        self.batches_dir = self.root / "batches"
        self.state_path = self.root / "state.json"
        self._initialize()

    def _initialize(self) -> None:
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            try:
                with open(self.state_path, encoding="utf-8") as handle:
                    state = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise CheckpointMismatchError(
                    f"Checkpoint state is unreadable: {self.state_path}"
                ) from exc
            expected = {
                "schema_version": self.SCHEMA_VERSION,
                "stage": self.stage,
                "shard_idx": self.shard_idx,
                "num_shards": self.num_shards,
                "fingerprint": self.fingerprint,
            }
            mismatch = {
                key: (state.get(key), value)
                for key, value in expected.items()
                if state.get(key) != value
            }
            if mismatch:
                raise CheckpointMismatchError(
                    "Existing generation checkpoint does not match this run. "
                    f"Delete {self.root} only if you intentionally want a fresh restart. "
                    f"Differences: {mismatch}"
                )
            self._update_state()
            return
        _atomic_json(
            self.state_path,
            {
                "schema_version": self.SCHEMA_VERSION,
                "stage": self.stage,
                "shard_idx": self.shard_idx,
                "num_shards": self.num_shards,
                "fingerprint": self.fingerprint,
                "completed_batches": [],
            },
        )

    @staticmethod
    def _batch_name(batch_index: int) -> str:
        if batch_index < 0:
            raise ValueError(f"batch_index must be non-negative, got {batch_index}")
        return f"batch-{batch_index:08d}"

    def _batch_dir(self, batch_index: int) -> Path:
        return self.batches_dir / self._batch_name(batch_index)

    def _valid_batch_manifest(self, batch_index: int) -> dict[str, int] | None:
        batch_dir = self._batch_dir(batch_index)
        if not batch_dir.is_dir() or not (batch_dir / "_SUCCESS").is_file():
            return None
        try:
            with open(batch_dir / "manifest.json", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(manifest, dict) or manifest.get("batch_index") != batch_index:
            return None
        if manifest.get("schema_version") != self.MANIFEST_SCHEMA_VERSION:
            return None
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            return None
        validated: dict[str, int] = {}
        for logical_name, count in files.items():
            if (
                not isinstance(logical_name, str)
                or not logical_name
                or "/" in logical_name
                or "\\" in logical_name
                or isinstance(count, bool)
            ):
                return None
            try:
                expected_count = int(count)
            except (TypeError, ValueError):
                return None
            if not _validate_jsonl(batch_dir / f"{logical_name}.jsonl", expected_count):
                return None
            validated[logical_name] = expected_count
        return validated

    def is_completed(self, batch_index: int) -> bool:
        """Return true only for a fully valid, durably committed batch."""
        return self._valid_batch_manifest(batch_index) is not None

    def completed_batches(self) -> set[int]:
        completed: set[int] = set()
        if not self.batches_dir.exists():
            return completed
        for child in self.batches_dir.iterdir():
            if not child.is_dir():
                continue
            match = self._BATCH_RE.fullmatch(child.name)
            if not match:
                continue
            batch_index = int(match.group(1))
            if self.is_completed(batch_index):
                completed.add(batch_index)
        return completed

    def validate_complete(self, expected_batches: int) -> None:
        """Require exactly ``range(expected_batches)`` to be durably committed."""
        if expected_batches < 0:
            raise ValueError(f"expected_batches must be non-negative, got {expected_batches}")
        completed = self.completed_batches()
        expected = set(range(expected_batches))
        missing = sorted(expected - completed)
        unexpected = sorted(completed - expected)
        if missing or unexpected:
            raise RuntimeError(
                f"Incomplete checkpoint for {self.stage} shard "
                f"{self.shard_idx}/{self.num_shards}: missing={missing[:20]}, "
                f"unexpected={unexpected[:20]}, expected_batches={expected_batches}"
            )

    def is_complete(self, expected_batches: int) -> bool:
        try:
            self.validate_complete(expected_batches)
        except RuntimeError:
            return False
        return True

    def commit(self, batch_index: int, files: Mapping[str, Iterable[Mapping[str, Any]]]) -> None:
        """Atomically persist all logical output files for one input batch."""
        if self.is_completed(batch_index):
            return
        if not files:
            raise ValueError("A checkpoint batch must contain at least one logical file")
        target = self._batch_dir(batch_index)
        if target.exists():
            # A stale/corrupt directory is never considered resumable.
            shutil.rmtree(target)
        tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=self.batches_dir))
        try:
            counts: dict[str, int] = {}
            for logical_name, rows in files.items():
                if not logical_name or "/" in logical_name or "\\" in logical_name:
                    raise ValueError(f"Unsafe logical checkpoint name: {logical_name!r}")
                counts[logical_name] = _write_jsonl(tmp / f"{logical_name}.jsonl", rows)
            _atomic_json(
                tmp / "manifest.json",
                {
                    "schema_version": self.MANIFEST_SCHEMA_VERSION,
                    "batch_index": batch_index,
                    "files": counts,
                },
            )
            (tmp / "_SUCCESS").write_text("ok\n", encoding="utf-8")
            os.replace(tmp, target)
            self._update_state()
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def _update_state(self) -> None:
        state = {
            "schema_version": self.SCHEMA_VERSION,
            "stage": self.stage,
            "shard_idx": self.shard_idx,
            "num_shards": self.num_shards,
            "fingerprint": self.fingerprint,
            "completed_batches": sorted(self.completed_batches()),
        }
        _atomic_json(self.state_path, state)

    def materialize(
        self,
        logical_name: str,
        output_path: str | Path,
        *,
        expected_batches: int,
    ) -> int:
        """Concatenate one logical file after validating the full batch range."""
        self.validate_complete(expected_batches)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
        count = 0
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                for batch_index in range(expected_batches):
                    manifest = self._valid_batch_manifest(batch_index)
                    if manifest is None or logical_name not in manifest:
                        raise RuntimeError(
                            f"Checkpoint batch {batch_index} is missing logical output "
                            f"{logical_name!r}"
                        )
                    source = self._batch_dir(batch_index) / f"{logical_name}.jsonl"
                    with open(source, encoding="utf-8") as handle:
                        for line in handle:
                            target.write(line)
                            if line.strip():
                                count += 1
            os.replace(tmp_name, output)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return count
