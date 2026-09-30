"""Runtime library path helpers for subprocess-heavy experiments."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from collections.abc import MutableMapping
from pathlib import Path

logger = logging.getLogger(__name__)

_PRELOADED = False


def _candidate_lib_dirs() -> list[str]:
    """Return local runtime library directories, newest/preferred first."""
    candidates: list[Path] = []

    for env_name in ("ACC_CONDA_LIB_PATH", "CONDA_PREFIX"):
        value = os.environ.get(env_name)
        if value:
            path = Path(value)
            candidates.append(path if path.name == "lib" else path / "lib")

    for prefix in (sys.prefix, sys.base_prefix):
        if prefix:
            candidates.append(Path(prefix) / "lib")

    executable = Path(sys.executable).resolve()
    if executable.parent.name == "bin":
        candidates.append(executable.parent.parent / "lib")

    seen: set[str] = set()
    result: list[str] = []
    for path in candidates:
        if path.is_dir():
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                result.append(resolved)
    return result


def _prepend_path(value: str, prefixes: list[str]) -> str:
    existing = [part for part in value.split(":") if part]
    merged: list[str] = []
    seen: set[str] = set()
    for part in [*prefixes, *existing]:
        if part not in seen:
            seen.add(part)
            merged.append(part)
    return ":".join(merged)


def ensure_runtime_library_path(
    env: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Prepend conda lib dirs to LD_LIBRARY_PATH for child processes."""
    target = os.environ if env is None else env
    lib_dirs = _candidate_lib_dirs()
    if lib_dirs:
        target["LD_LIBRARY_PATH"] = _prepend_path(
            target.get("LD_LIBRARY_PATH", ""),
            lib_dirs,
        )
    return target


def preload_runtime_libraries() -> None:
    """Load conda libstdc++ early so direct dlopen users avoid system libstdc++."""
    global _PRELOADED
    if _PRELOADED:
        return

    for lib_dir in _candidate_lib_dirs():
        libstdcpp = Path(lib_dir) / "libstdc++.so.6"
        if not libstdcpp.exists():
            continue
        try:
            ctypes.CDLL(str(libstdcpp), mode=ctypes.RTLD_GLOBAL)
            _PRELOADED = True
            logger.debug("Preloaded runtime library: %s", libstdcpp)
            return
        except OSError as exc:
            logger.debug("Could not preload %s: %s", libstdcpp, exc)


def configure_runtime_libraries(
    env: MutableMapping[str, str] | None = None,
    *,
    preload: bool = True,
) -> MutableMapping[str, str]:
    """Configure library lookup for current and child Python processes."""
    target = ensure_runtime_library_path(env)
    if preload and env is None:
        preload_runtime_libraries()
    return target
