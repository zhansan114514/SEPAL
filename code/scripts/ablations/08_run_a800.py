"""Run all three model ablations efficiently on four A800 GPUs."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.config import load_ablation_manifest
from src.ablations.runtime import offline_ablation_env

ANALYZE = PROJECT_ROOT / "scripts/ablations/04_analyze_existing.py"
POLICY_MATRIX = PROJECT_ROOT / "scripts/ablations/03_run_policy_matrix.py"
NO_SFT = PROJECT_ROOT / "scripts/ablations/05_run_no_sft_full.py"
_ACTIVE: list[subprocess.Popen[Any]] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-manifest", required=True)
    parser.add_argument("--qwen-manifest", required=True)
    parser.add_argument("--gemma-manifest", required=True)
    parser.add_argument("--llama-devices", default="0,1")
    parser.add_argument("--qwen-devices", default="2")
    parser.add_argument("--gemma-devices", default="3")
    parser.add_argument("--all-devices", default="0,1,2,3")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all manifests and print the A800 phase plan without running work.",
    )
    args = parser.parse_args()
    pools = {
        "llama3": _parse_devices(args.llama_devices),
        "qwen25": _parse_devices(args.qwen_devices),
        "gemma2": _parse_devices(args.gemma_devices),
    }
    all_devices = _parse_devices(args.all_devices)
    _assert_disjoint_complete(pools, all_devices)
    manifests = {
        "llama3": args.llama_manifest,
        "qwen25": args.qwen_manifest,
        "gemma2": args.gemma_manifest,
    }
    for model, manifest_path in manifests.items():
        manifest = load_ablation_manifest(PROJECT_ROOT / manifest_path)
        actual_model = str(manifest["model"]["key"])
        if actual_model != model:
            raise ValueError(
                f"Manifest/model mismatch for {model}: {actual_model} / {manifest_path}"
            )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "phase_0": {
                        "kind": "offline_existing_result_analysis",
                        "models": list(manifests),
                    },
                    "phase_1": {
                        "kind": "policy_lattice",
                        "concurrent_device_pools": {
                            model: list(devices) for model, devices in pools.items()
                        },
                    },
                    "phase_2": {
                        "kind": "no_sft_full_retraining",
                        "sequential_models": list(manifests),
                        "devices_per_model": list(all_devices),
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    _install_signal_handlers()

    # CPU-only analysis is done first. Completed outputs are hash-authenticated
    # and return immediately on a resumed launcher.
    for model in ("llama3", "qwen25", "gemma2"):
        _run(
            [
                sys.executable,
                str(ANALYZE),
                "--manifest",
                manifests[model],
            ]
        )

    # Phase 1 keeps all four cards occupied: the largest model gets two cards.
    processes: dict[str, subprocess.Popen[Any]] = {}
    for model in ("llama3", "qwen25", "gemma2"):
        process = subprocess.Popen(
            [
                sys.executable,
                str(POLICY_MATRIX),
                "--manifest",
                manifests[model],
                "--devices",
                ",".join(str(device) for device in pools[model]),
            ],
            cwd=PROJECT_ROOT,
            env=_offline_env(),
        )
        processes[model] = process
        _ACTIVE.append(process)
    _wait_fail_fast(processes, stage="policy_lattice")

    # Phase 2 gives each full retraining run all four cards. This maximizes
    # generation/evaluation throughput while preserving four logical shards.
    for model in ("llama3", "qwen25", "gemma2"):
        _run(
            [
                sys.executable,
                str(NO_SFT),
                "--manifest",
                manifests[model],
                "--devices",
                ",".join(str(device) for device in all_devices),
            ]
        )

    print(
        f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] "
        "All A800 ablation stages completed.",
        flush=True,
    )


def _run(command: list[str]) -> None:
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=_offline_env())
    _ACTIVE.append(process)
    code = process.wait()
    if process in _ACTIVE:
        _ACTIVE.remove(process)
    if code != 0:
        raise RuntimeError(f"A800 ablation stage failed with exit code {code}: {command}")


def _wait_fail_fast(
    processes: dict[str, subprocess.Popen[Any]],
    *,
    stage: str,
) -> None:
    remaining = dict(processes)
    while remaining:
        for model, process in list(remaining.items()):
            code = process.poll()
            if code is None:
                continue
            remaining.pop(model)
            if process in _ACTIVE:
                _ACTIVE.remove(process)
            if code != 0:
                _terminate_active()
                raise RuntimeError(f"{stage}/{model} failed with exit code {code}")
        if remaining:
            time.sleep(2)


def _parse_devices(raw: str) -> tuple[int, ...]:
    devices = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not devices or len(set(devices)) != len(devices) or any(device < 0 for device in devices):
        raise ValueError(f"Invalid device pool: {raw!r}")
    return devices


def _assert_disjoint_complete(
    pools: dict[str, tuple[int, ...]],
    all_devices: tuple[int, ...],
) -> None:
    assigned: list[int] = [
        device
        for devices in pools.values()
        for device in devices
    ]
    if len(assigned) != len(set(assigned)):
        raise ValueError(f"Phase-1 A800 pools overlap: {pools}")
    if set(assigned) != set(all_devices):
        raise ValueError(
            f"Phase-1 pools must cover --all-devices exactly: {pools} / {all_devices}"
        )


def _offline_env() -> dict[str, str]:
    return offline_ablation_env(PROJECT_ROOT, base=os.environ)


def _install_signal_handlers() -> None:
    def stop(_signum: int, _frame: Any) -> None:
        _terminate_active()
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, stop)


def _terminate_active() -> None:
    for process in list(_ACTIVE):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 20
    while any(process.poll() is None for process in _ACTIVE) and time.monotonic() < deadline:
        time.sleep(0.5)
    for process in list(_ACTIVE):
        if process.poll() is None:
            process.kill()
    _ACTIVE.clear()


if __name__ == "__main__":
    main()
