"""Wait for two selected H100s, then resume the Qwen2.5 ablation matrix."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.runtime import offline_ablation_env

MODEL_RUNNER = PROJECT_ROOT / "scripts/ablations/07_run_model.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--devices", default="4,5")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--free-observations", type=int, default=3)
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument(
        "--lock-file",
        default="logs/.qwen25_ablation_h100.lock",
    )
    args = parser.parse_args()
    if args.poll_seconds < 5 or args.free_observations < 1:
        raise ValueError("Invalid watcher polling configuration")
    devices = _parse_devices(args.devices)
    lock_path = _resolve_path(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as handle:
        _acquire_lock(handle)
        _write_event(
            "watcher_started",
            devices=list(devices),
            manifest=args.manifest,
        )
        consecutive = 0
        while consecutive < args.free_observations:
            snapshot = _gpu_snapshot()
            selected = {index: snapshot[index] for index in devices}
            free = all(
                not info["compute_pids"] and int(info["memory_used_mib"]) < 1024
                for info in selected.values()
            )
            consecutive = consecutive + 1 if free else 0
            _write_event(
                "gpu_probe",
                free=free,
                consecutive_free_observations=consecutive,
                selected=selected,
            )
            if consecutive < args.free_observations:
                time.sleep(args.poll_seconds)

        _write_event("launching_matrix", devices=list(devices))
        command = [
            sys.executable,
            str(MODEL_RUNNER),
            "--manifest",
            args.manifest,
            "--devices",
            ",".join(str(device) for device in devices),
        ]
        if args.skip_offline:
            command.append("--skip-offline")
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=_offline_env(),
            check=False,
        )
        if completed.returncode != 0:
            _write_event("matrix_failed", exit_code=completed.returncode)
            raise SystemExit(completed.returncode)
        _write_event("matrix_complete")


def _gpu_snapshot() -> dict[int, dict[str, Any]]:
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    snapshot: dict[int, dict[str, Any]] = {}
    uuid_to_index: dict[str, int] = {}
    for line in gpu_result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        index = int(fields[0])
        snapshot[index] = {
            "uuid": fields[1],
            "memory_used_mib": int(fields[2]),
            "utilization_percent": int(fields[3]),
            "compute_pids": [],
        }
        uuid_to_index[fields[1]] = index
    app_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    for line in app_result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[1].isdigit():
            continue
        index = uuid_to_index.get(fields[0])
        if index is not None:
            snapshot[index]["compute_pids"].append(int(fields[1]))
    return snapshot


def _parse_devices(raw: str) -> tuple[int, ...]:
    devices = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if len(devices) != 2 or len(set(devices)) != 2 or any(device < 0 for device in devices):
        raise ValueError("H100 watcher requires exactly two distinct GPU ids")
    return devices


def _acquire_lock(handle: Any) -> None:
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another H100 ablation watcher already owns the lock") from None


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_event(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "event": event,
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _offline_env() -> dict[str, str]:
    return offline_ablation_env(PROJECT_ROOT, base=os.environ)


if __name__ == "__main__":
    main()
