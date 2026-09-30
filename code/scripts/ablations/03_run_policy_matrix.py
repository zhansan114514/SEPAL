"""Run all pre-registered intermediate-policy ablations on a physical GPU pool."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acccollab.io import write_json
from src.ablations.config import (
    DATASET_LOGICAL_SHARDS,
    POLICY_VARIANT_DATASETS,
    POLICY_VARIANTS,
    ROLE_NAMES,
    load_ablation_manifest,
)
from src.ablations.runtime import offline_ablation_env
from src.utils.artifacts import stable_fingerprint

EVALUATOR = PROJECT_ROOT / "scripts/ablations/02_evaluate_policy.py"
_ACTIVE: list[subprocess.Popen] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--variants", nargs="*", choices=POLICY_VARIANTS, default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    devices = _parse_devices(args.devices)
    manifest = load_ablation_manifest(args.manifest)
    variants = tuple(args.variants or POLICY_VARIANTS)
    output_root = _resolve_path(manifest["output_root"])
    plan = _plan(manifest, variants=variants, devices=devices)
    plan_dir = output_root / "policy_lattice"
    if args.dry_run:
        print(plan)
        return

    plan_dir.mkdir(parents=True, exist_ok=True)
    write_json(plan_dir / "plan.json", plan)
    _install_signal_handlers()
    for variant in variants:
        for dataset_name in POLICY_VARIANT_DATASETS[variant]:
            _run_dataset_variant(
                manifest_path=Path(args.manifest),
                manifest=manifest,
                output_root=plan_dir,
                dataset_name=dataset_name,
                variant=variant,
                devices=devices,
            )

    write_json(
        plan_dir / "_SUCCESS",
        {
            "schema_version": 1,
            "pipeline": "multi_acccollab_ablation",
            "stage": "policy_lattice_matrix",
            "status": "complete",
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "plan_fingerprint": plan["fingerprint"],
        },
    )


def _run_dataset_variant(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    output_root: Path,
    dataset_name: str,
    variant: str,
    devices: tuple[int, ...],
) -> None:
    dataset = dict(dict(manifest["datasets"])[dataset_name])
    logical_shards = int(dataset["logical_shards"])
    output_dir = output_root / dataset_name / variant
    jobs = deque(
        (role, shard_idx)
        for role in ROLE_NAMES
        for shard_idx in range(logical_shards)
    )
    running: dict[int, tuple[str, int, subprocess.Popen]] = {}
    while jobs or running:
        for device in devices:
            if device in running or not jobs:
                continue
            role, shard_idx = jobs.popleft()
            command = [
                sys.executable,
                str(EVALUATOR),
                "--manifest",
                str(manifest_path),
                "--dataset",
                dataset_name,
                "--variant",
                variant,
                "--role",
                role,
                "--output-dir",
                str(output_dir),
                "--device",
                str(device),
                "--shard-idx",
                str(shard_idx),
                "--num-shards",
                str(logical_shards),
            ]
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=_offline_env())
            running[device] = (role, shard_idx, process)
            _ACTIVE.append(process)
        for device, (role, shard_idx, process) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            running.pop(device)
            if process in _ACTIVE:
                _ACTIVE.remove(process)
            if code != 0:
                _terminate_active()
                raise RuntimeError(
                    f"{variant}/{dataset_name}/{role} shard {shard_idx} "
                    f"failed on GPU {device}: exit={code}"
                )
        if running:
            time.sleep(1.0)

    for role in ROLE_NAMES:
        _run(
            [
                sys.executable,
                str(EVALUATOR),
                "--manifest",
                str(manifest_path),
                "--dataset",
                dataset_name,
                "--variant",
                variant,
                "--role",
                role,
                "--output-dir",
                str(output_dir),
                "--num-shards",
                str(logical_shards),
                "--merge",
            ]
        )
    _run(
        [
            sys.executable,
            str(EVALUATOR),
            "--manifest",
            str(manifest_path),
            "--dataset",
            dataset_name,
            "--variant",
            variant,
            "--output-dir",
            str(output_dir),
            "--aggregate",
        ]
    )


def _plan(
    manifest: dict[str, Any],
    *,
    variants: tuple[str, ...],
    devices: tuple[int, ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": "multi_acccollab_ablation",
        "stage": "policy_lattice_matrix",
        "scope_version": "multi_acccollab_policy_matrix_all_datasets_v2",
        "model": dict(manifest["model"]),
        "manifest_fingerprint": manifest["fingerprint"],
        "variants": {
            variant: list(POLICY_VARIANT_DATASETS[variant])
            for variant in variants
        },
        "devices": list(devices),
        "logical_shards": dict(DATASET_LOGICAL_SHARDS),
        "preserves_main_experiment_batch_and_seed_layout": True,
    }
    payload["fingerprint"] = stable_fingerprint(payload)
    return payload


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, env=_offline_env(), check=True)


def _parse_devices(raw: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --devices value: {raw!r}") from exc
    if not devices or any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative GPU ids")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")
    return devices


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


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
    deadline = time.monotonic() + 15
    while any(process.poll() is None for process in _ACTIVE) and time.monotonic() < deadline:
        time.sleep(0.5)
    for process in list(_ACTIVE):
        if process.poll() is None:
            process.kill()
    _ACTIVE.clear()


if __name__ == "__main__":
    main()
