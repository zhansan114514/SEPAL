"""Rescore completed full Multi-A/C records and expose component diagnostics."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acccollab.config import load_acccollab_config
from src.acccollab.io import read_json
from src.ablations.config import (
    DATASET_ORDER,
    ROLE_NAMES,
    load_ablation_manifest,
    resolve_manifest_path,
)
from src.ablations.policy_lattice import aggregate_role_records
from src.utils.artifacts import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=DATASET_ORDER,
        default=[],
    )
    args = parser.parse_args()

    manifest = load_ablation_manifest(args.manifest)
    datasets = tuple(args.datasets or DATASET_ORDER)
    output_root = _resolve_path(manifest["output_root"]) / "offline_full"
    for dataset_name in datasets:
        role_paths = _existing_role_records(
            manifest,
            dataset_name=dataset_name,
        )
        aggregate_role_records(
            role_paths,
            output_dir=output_root / dataset_name,
            dataset_name=dataset_name,
            variant="full_trained_rounds",
            expected_samples=int(manifest["datasets"][dataset_name]["expected_samples"]),
            reparse_completions=True,
        )


def _existing_role_records(
    manifest: dict[str, Any],
    *,
    dataset_name: str,
) -> dict[str, Path]:
    roles = dict(dict(manifest["datasets"])[dataset_name]["roles"])
    records: dict[str, Path] = {}
    for role in ROLE_NAMES:
        source_record = dict(dict(roles[role])["source_config"])
        config_path = resolve_manifest_path(
            source_record["path"],
            project_root=PROJECT_ROOT,
        )
        if file_sha256(config_path) != source_record["sha256"]:
            raise RuntimeError(f"Source role config changed: {config_path}")
        config = load_acccollab_config(str(config_path))
        records_path = config.paths.eval_dir / "records.jsonl"
        success_path = config.paths.eval_dir / "_SUCCESS"
        if not records_path.is_file() or not success_path.is_file():
            raise FileNotFoundError(
                f"Completed full evaluation is missing for {dataset_name}/{role}: "
                f"{config.paths.eval_dir}"
            )
        marker = read_json(success_path)
        record_artifact = dict(dict(marker.get("artifacts") or {}).get("records") or {})
        if (
            marker.get("pipeline") != "acccollab_original"
            or marker.get("stage") != "evaluate"
            or marker.get("status") != "complete"
            or record_artifact.get("sha256") != file_sha256(records_path)
        ):
            raise RuntimeError(
                f"Full evaluation marker is stale for {dataset_name}/{role}: "
                f"{success_path}"
            )
        expected = int(manifest["datasets"][dataset_name]["expected_samples"])
        if int(dict(marker.get("metadata") or {}).get("samples", -1)) != expected:
            raise RuntimeError(
                f"Full evaluation coverage is wrong for {dataset_name}/{role}"
            )
        records[role] = records_path
    return records


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
