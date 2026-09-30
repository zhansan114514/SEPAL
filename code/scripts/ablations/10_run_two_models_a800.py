"""Run two authenticated model ablations efficiently on four A800 GPUs."""

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
    parser.add_argument("--first-manifest", required=True)
    parser.add_argument("--second-manifest", required=True)
    parser.add_argument("--first-devices", default="0,1")
    parser.add_argument("--second-devices", default="2,3")
    parser.add_argument("--all-devices", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_paths = (args.first_manifest, args.second_manifest)
    manifests = [load_ablation_manifest(PROJECT_ROOT / path) for path in manifest_paths]
    model_keys = [str(manifest["model"]["key"]) for manifest in manifests]
    if len(set(model_keys)) != 2:
        raise ValueError(f"Two different model manifests are required: {model_keys}")
    pools = (_parse_devices(args.first_devices), _parse_devices(args.second_devices))
    all_devices = _parse_devices(args.all_devices)
    _assert_disjoint_complete(pools, all_devices)

    plan = {
        "phase_0": {"kind": "offline_existing_result_analysis", "models": model_keys},
        "phase_1": {
            "kind": "policy_lattice",
            "concurrent_device_pools": {
                model: list(devices) for model, devices in zip(model_keys, pools, strict=True)
            },
        },
        "phase_2": {
            "kind": "no_sft_full_retraining",
            "sequential_models": model_keys,
            "devices_per_model": list(all_devices),
        },
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    _install_signal_handlers()
    for manifest_path in manifest_paths:
        _run([sys.executable, str(ANALYZE), "--manifest", manifest_path])

    processes: dict[str, subprocess.Popen[Any]] = {}
    for model, manifest_path, devices in zip(
        model_keys, manifest_paths, pools, strict=True
    ):
        process = subprocess.Popen(
            [
                sys.executable,
                str(POLICY_MATRIX),
                "--manifest",
                manifest_path,
                "--devices",
                ",".join(str(device) for device in devices),
            ],
            cwd=PROJECT_ROOT,
            env=_offline_env(),
        )
        processes[model] = process
        _ACTIVE.append(process)
    _wait_fail_fast(processes, stage="policy_lattice")

    for manifest_path in manifest_paths:
        _run(
            [
                sys.executable,
                str(NO_SFT),
                "--manifest",
                manifest_path,
                "--devices",
                ",".join(str(device) for device in all_devices),
            ]
        )
    print(
        f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] "
        "All two-model A800 ablation stages completed.",
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
    pools: tuple[tuple[int, ...], tuple[int, ...]],
    all_devices: tuple[int, ...],
) -> None:
    assigned = [device for devices in pools for device in devices]
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
