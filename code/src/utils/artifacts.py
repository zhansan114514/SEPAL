"""Atomic identities and completion markers for training artifacts."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


class ArtifactMismatchError(RuntimeError):
    """Raised when resumable output belongs to different training inputs."""


def stable_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, target)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_identity(
    path: str | Path,
    *,
    hash_weights: bool = False,
) -> dict[str, Any]:
    """Return a compact identity for a registry, adapter, or ordinary path."""
    target = Path(path)
    if target.is_file():
        return {
            "path": str(target),
            "kind": "file",
            "sha256": file_sha256(target),
            "size": target.stat().st_size,
        }
    if target.is_dir():
        selected: dict[str, Any] = {}
        for name in ("_SUCCESS", "adapter_config.json"):
            item = target / name
            if item.is_file():
                selected[name] = {
                    "sha256": file_sha256(item),
                    "size": item.stat().st_size,
                }
        weight = next(
            (
                target / name
                for name in ("adapter_model.safetensors", "adapter_model.bin")
                if (target / name).is_file()
            ),
            None,
        )
        if weight is not None:
            selected[weight.name] = {"size": weight.stat().st_size}
            if hash_weights:
                selected[weight.name]["sha256"] = file_sha256(weight)
        return {"path": str(target), "kind": "directory", "selected": selected}
    return {"path": str(target), "kind": "missing"}


def _adapter_candidates(path: str | Path) -> list[Path]:
    target = Path(path)
    suffixed = Path(str(target) + "_adapter")
    return [suffixed, target] if suffixed != target else [target]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def completed_adapter_path(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> str:
    """Return a completed adapter path, rejecting mismatched success markers."""
    for adapter in _adapter_candidates(path):
        config = adapter / "adapter_config.json"
        success = adapter / "_SUCCESS"
        weight_exists = any(
            (adapter / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        )
        if not config.is_file() or not success.is_file() or not weight_exists:
            continue
        payload = _read_json(success)
        if not payload or payload.get("status") != "complete":
            continue
        actual = payload.get("training_fingerprint")
        if expected_fingerprint is not None and actual != expected_fingerprint:
            raise ArtifactMismatchError(
                f"Completed adapter {adapter} belongs to training fingerprint {actual!r}, "
                f"but current run expects {expected_fingerprint!r}. Use a new output_dir or "
                "remove the old training output intentionally."
            )
        return str(adapter)
    return ""


def write_adapter_success(
    adapter_dir: str | Path,
    *,
    training_fingerprint: str,
    base_model: str,
    output_dir: str,
    training_kind: str,
) -> None:
    adapter = Path(adapter_dir)
    if not (adapter / "adapter_config.json").is_file() or not any(
        (adapter / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise RuntimeError(f"Cannot mark incomplete adapter as successful: {adapter}")
    atomic_write_json(
        adapter / "_SUCCESS",
        {
            "schema_version": 1,
            "status": "complete",
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "training_kind": training_kind,
            "training_fingerprint": training_fingerprint,
            "base_model": base_model,
            "output_dir": output_dir,
        },
    )
