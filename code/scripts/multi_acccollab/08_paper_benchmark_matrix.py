"""Run the single-trial model benchmark matrix in fixed method order."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from _utils import PROJECT_ROOT, offline_subprocess_env, setup_logging
from src.acccollab.io import write_json
from src.benchmark_matrix.configs import MODEL_SPECS, ResolvedModelConfigs, materialize_model_configs
from src.utils.artifacts import file_sha256, stable_fingerprint


logger = logging.getLogger(__name__)
ACCCOLLAB_PIPELINE = PROJECT_ROOT / "scripts/acccollab/06_pipeline.py"
MULTI_FORMAL = PROJECT_ROOT / "scripts/multi_acccollab/07_formal_experiment.py"
MULTI_EVALUATE = PROJECT_ROOT / "scripts/multi_acccollab/06_evaluate_dataset.py"
METHODS = ("original", "current")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=["llama3", "mistral", "gemma2"],
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--benchmark-data-dir",
        default="benchmark_data/acccollab_paper",
    )
    parser.add_argument(
        "--output-root",
        default="output/benchmark_matrix",
    )
    parser.add_argument(
        "--devices",
        default="0,1,2,3",
        help="Comma-separated physical GPU ids reserved exclusively for this run.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(42)
    devices = _parse_devices(args.devices)
    output_root = Path(args.output_root)
    paper_data = Path(args.benchmark_data_dir)
    _validate_paper_data(paper_data)
    plans = [
        materialize_model_configs(
            model_key,
            output_root=output_root,
            benchmark_data_dir=paper_data,
            devices=devices,
        )
        for model_key in args.models
    ]
    plan_payload = _plan_payload(
        plans,
        methods=tuple(args.methods),
        paper_data=paper_data,
        devices=devices,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "plan.json", plan_payload)
    logger.info("Prepared benchmark matrix: %s", plan_payload["schedule"])
    if args.prepare_only:
        return

    for configs in plans:
        _require_model(configs)
        datasets = set(configs.spec.datasets)
        if "original" in args.methods:
            _train_original_if_needed(configs, devices)
            _evaluate_original(configs, datasets, devices)
        if "current" in args.methods:
            _train_current_if_needed(configs, devices)
            _evaluate_current(configs, datasets, devices)
        _write_model_success(output_root, configs, methods=tuple(args.methods))
    write_json(
        output_root / "_SUCCESS",
        {
            "schema_version": 1,
            "pipeline": "paper_benchmark_matrix",
            "stage": "all_models",
            "status": "complete",
            "models": list(args.models),
            "methods": list(args.methods),
            "plan_fingerprint": plan_payload["fingerprint"],
        },
    )


def _train_original_if_needed(
    configs: ResolvedModelConfigs,
    devices: tuple[int, ...],
) -> None:
    registry = configs.original_output_dir / "registry/final.json"
    if registry.is_file():
        logger.info("Reusing original policy for %s: %s", configs.spec.key, registry)
        return
    _run(
        [
            sys.executable,
            str(ACCCOLLAB_PIPELINE),
            "--config",
            str(configs.original_primary),
            "--devices",
            _device_arg(devices),
            "--training-devices",
            str(devices[0]),
            "--only",
            "critic-data",
            "critic-train",
            "actor-data",
            "actor-train",
        ]
    )
    _require_file(registry)


def _train_current_if_needed(
    configs: ResolvedModelConfigs,
    devices: tuple[int, ...],
) -> None:
    registries = [
        configs.multi_output_dir / "roles" / role / "registry/final.json"
        for role in ("direct", "evidence", "verification")
    ]
    if all(path.is_file() for path in registries):
        logger.info("Reusing current-method policies for %s", configs.spec.key)
        return
    _run(
        [
            sys.executable,
            str(MULTI_FORMAL),
            "--config",
            str(configs.multi_primary),
            "--boolq-config",
            str(configs.multi_evaluations["boolq"]),
            "--sciq-config",
            str(configs.multi_evaluations["sciq"]),
            "--devices",
            _device_arg(devices),
            "--only",
            "sft",
            "role-training",
        ]
    )
    for registry in registries:
        _require_file(registry)


def _evaluate_original(
    configs: ResolvedModelConfigs,
    datasets: set[str],
    devices: tuple[int, ...],
) -> None:
    phases = _evaluation_phases(devices)
    for phase in phases:
        jobs = []
        for dataset_name, device_spec in phase:
            if dataset_name not in datasets:
                continue
            jobs.append(
                (
                    dataset_name,
                    [
                        sys.executable,
                        str(ACCCOLLAB_PIPELINE),
                        "--config",
                        str(configs.original_evaluations[dataset_name]),
                        "--devices",
                        device_spec,
                        "--training-devices",
                        str(devices[0]),
                        "--only",
                        "evaluate",
                    ],
                )
            )
        _run_parallel(jobs)


def _evaluate_current(
    configs: ResolvedModelConfigs,
    datasets: set[str],
    devices: tuple[int, ...],
) -> None:
    phases = _evaluation_phases(devices)
    for phase in phases:
        jobs = []
        for dataset_name, device_spec in phase:
            if dataset_name not in datasets:
                continue
            jobs.append(
                (
                    dataset_name,
                    [
                        sys.executable,
                        str(MULTI_EVALUATE),
                        "--config",
                        str(configs.multi_evaluations[dataset_name]),
                        "--devices",
                        device_spec,
                    ],
                )
            )
        _run_parallel(jobs)


def _run_parallel(jobs: list[tuple[str, list[str]]]) -> None:
    if not jobs:
        return
    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        for name, command in jobs:
            logger.info("Starting evaluation job %s", name)
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=offline_subprocess_env(),
            )
            processes.append((name, process))
        pending = dict(processes)
        while pending:
            for name, process in list(pending.items()):
                code = process.poll()
                if code is None:
                    continue
                del pending[name]
                if code != 0:
                    _terminate_all(list(pending.items()))
                    raise RuntimeError(f"Evaluation job failed: {name}={code}")
            if pending:
                time.sleep(1.0)
    except Exception:
        _terminate_all(processes)
        raise


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


def _device_arg(devices: tuple[int, ...]) -> str:
    return ",".join(str(device) for device in devices)


def _evaluation_phases(
    devices: tuple[int, ...],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Allocate concurrent datasets by coverage while keeping GPU sets disjoint."""
    if len(devices) == 1:
        only = str(devices[0])
        return tuple(
            ((dataset, only),)
            for dataset in ("boolq", "sciq", "bbh", "arc", "mmlu")
        )

    def split(left_weight: int, right_weight: int) -> tuple[str, str]:
        left_count = round(len(devices) * left_weight / (left_weight + right_weight))
        left_count = max(1, min(len(devices) - 1, left_count))
        return _device_arg(devices[:left_count]), _device_arg(devices[left_count:])

    boolq_devices, sciq_devices = split(3270, 1000)
    bbh_devices, arc_devices = split(1260, 3548)
    return (
        (("boolq", boolq_devices), ("sciq", sciq_devices)),
        (("bbh", bbh_devices), ("arc", arc_devices)),
        (("mmlu", _device_arg(devices)),),
    )


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, env=offline_subprocess_env(), check=True)


def _terminate_all(processes: list[tuple[str, subprocess.Popen]]) -> None:
    for _name, process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(
        process.poll() is None for _name, process in processes
    ):
        time.sleep(0.1)
    for _name, process in processes:
        if process.poll() is None:
            process.kill()


def _validate_paper_data(root: Path) -> None:
    expected = [
        root / "BBH/boolean_expressions.json",
        root / "BBH/word_sorting.json",
        root / "ARC/test-00000-of-00001-2.parquet",
        root / "ARC/test-00000-of-00001-3.parquet",
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Paper benchmark data is incomplete: {missing}")


def _require_model(configs: ResolvedModelConfigs) -> None:
    root = Path(configs.spec.model_path)
    required = [root / "config.json", root / "tokenizer_config.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Model {configs.spec.key} is not ready: {missing}")

    index_candidates = [
        root / "model.safetensors.index.json",
        root / "pytorch_model.bin.index.json",
    ]
    index_path = next((path for path in index_candidates if path.is_file()), None)
    if index_path is not None:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_names = sorted(set(payload.get("weight_map", {}).values()))
        missing_weights = [
            str(root / name)
            for name in weight_names
            if not (root / name).is_file() or (root / name).stat().st_size == 0
        ]
        if missing_weights:
            raise FileNotFoundError(
                f"Model {configs.spec.key} has an incomplete sharded snapshot: "
                f"{missing_weights}"
            )
        return

    single_weights = [root / "model.safetensors", root / "pytorch_model.bin"]
    if not any(path.is_file() and path.stat().st_size > 0 for path in single_weights):
        raise FileNotFoundError(
            f"Model {configs.spec.key} has no complete supported weight snapshot under {root}"
        )


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required experiment artifact is missing: {path}")


def _write_model_success(
    output_root: Path,
    configs: ResolvedModelConfigs,
    *,
    methods: tuple[str, ...],
) -> None:
    write_json(
        output_root / f"{configs.spec.key}_SUCCESS.json",
        {
            "schema_version": 1,
            "pipeline": "paper_benchmark_matrix",
            "stage": "model",
            "status": "complete",
            "model": configs.spec.key,
            "methods": list(methods),
            "datasets": list(configs.spec.datasets),
        },
    )


def _plan_payload(
    plans: list[ResolvedModelConfigs],
    *,
    methods: tuple[str, ...],
    paper_data: Path,
    devices: tuple[int, ...],
) -> dict[str, object]:
    phases = _evaluation_phases(devices)
    payload: dict[str, object] = {
        "schema_version": 1,
        "pipeline": "paper_benchmark_matrix",
        "models": [plan.spec.key for plan in plans],
        "methods": list(methods),
        "trials": 1,
        "devices": list(devices),
        "paper_data": str(paper_data.resolve(strict=False)),
        "coverage": {
            "boolq": 3270,
            "mmlu": 14042,
            "bbh": 1260,
            "sciq": 1000,
            "arc": 3548,
        },
        "schedule": [
            "per model: original training/evaluation before current-method training/evaluation",
            *[
                "phase "
                f"{index}: "
                + " concurrently with ".join(
                    f"{dataset} GPU(s) {assigned}" for dataset, assigned in phase
                )
                for index, phase in enumerate(phases, start=1)
            ],
        ],
        "configs": {
            plan.spec.key: {
                "original_primary": str(plan.original_primary),
                "original_evaluations": {
                    name: str(path) for name, path in plan.original_evaluations.items()
                },
                "multi_primary": str(plan.multi_primary),
                "multi_evaluations": {
                    name: str(path) for name, path in plan.multi_evaluations.items()
                },
            }
            for plan in plans
        },
    }
    config_paths = []
    for plan in plans:
        config_paths.extend(plan.original_evaluations.values())
        config_paths.extend(plan.multi_evaluations.values())
    payload["config_sha256"] = {
        str(path): file_sha256(path) for path in sorted(set(config_paths))
    }
    payload["fingerprint"] = stable_fingerprint(payload)
    return payload


if __name__ == "__main__":
    main()
