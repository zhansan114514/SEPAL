"""Run all registered ablations for one authenticated model manifest."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acccollab.io import read_json, write_json
from src.ablations.config import load_ablation_manifest
from src.ablations.runtime import offline_ablation_env
from src.utils.artifacts import file_sha256, stable_fingerprint

ANALYZE = PROJECT_ROOT / "scripts/ablations/04_analyze_existing.py"
POLICY_MATRIX = PROJECT_ROOT / "scripts/ablations/03_run_policy_matrix.py"
NO_SFT = PROJECT_ROOT / "scripts/ablations/05_run_no_sft_full.py"
MODEL_MATRIX_VERSION = "multi_acccollab_model_ablation_matrix_v1"
_CURRENT: subprocess.Popen[Any] | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_ablation_manifest(manifest_path)
    output_root = _resolve_path(manifest["output_root"])
    plan = {
        "schema_version": 1,
        "pipeline": "multi_acccollab_ablation",
        "stage": "model_matrix",
        "implementation_version": MODEL_MATRIX_VERSION,
        "manifest_fingerprint": manifest["fingerprint"],
        "model": dict(manifest["model"]),
        "devices": args.devices,
        "steps": (
            ["intermediate_policy_lattice", "no_sft_full"]
            if args.skip_offline
            else [
                "offline_full_diagnostics",
                "intermediate_policy_lattice",
                "no_sft_full",
            ]
        ),
    }
    plan["fingerprint"] = stable_fingerprint(plan)
    if args.dry_run:
        print(plan)
        return

    success_path = output_root / "_MATRIX_SUCCESS"
    if _completed(success_path, plan["fingerprint"]):
        return
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "model_matrix_plan.json"
    write_json(plan_path, plan)
    _install_signal_handlers()

    if not args.skip_offline:
        _run(
            [
                sys.executable,
                str(ANALYZE),
                "--manifest",
                str(manifest_path),
            ]
        )
    _run(
        [
            sys.executable,
            str(POLICY_MATRIX),
            "--manifest",
            str(manifest_path),
            "--devices",
            args.devices,
        ]
    )
    _run(
        [
            sys.executable,
            str(NO_SFT),
            "--manifest",
            str(manifest_path),
            "--devices",
            args.devices,
        ]
    )
    write_json(
        success_path,
        {
            "schema_version": 1,
            "pipeline": "multi_acccollab_ablation",
            "stage": "model_matrix",
            "status": "complete",
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "implementation_version": MODEL_MATRIX_VERSION,
            "fingerprint": plan["fingerprint"],
            "artifacts": {
                "plan": {
                    "path": str(plan_path),
                    "sha256": file_sha256(plan_path),
                }
            },
        },
    )


def _run(command: list[str]) -> None:
    global _CURRENT
    _CURRENT = subprocess.Popen(command, cwd=PROJECT_ROOT, env=_offline_env())
    return_code = _CURRENT.wait()
    _CURRENT = None
    if return_code != 0:
        raise RuntimeError(f"Ablation step failed with exit code {return_code}: {command}")


def _completed(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    payload = read_json(path)
    return (
        payload.get("status") == "complete"
        and payload.get("stage") == "model_matrix"
        and payload.get("fingerprint") == fingerprint
    )


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _offline_env() -> dict[str, str]:
    return offline_ablation_env(PROJECT_ROOT, base=os.environ)


def _install_signal_handlers() -> None:
    def stop(_signum: int, _frame: Any) -> None:
        if _CURRENT is not None and _CURRENT.poll() is None:
            _CURRENT.terminate()
            try:
                _CURRENT.wait(timeout=20)
            except subprocess.TimeoutExpired:
                _CURRENT.kill()
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, stop)


if __name__ == "__main__":
    main()
