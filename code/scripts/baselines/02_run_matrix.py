"""Run Direct, Debate, SoM-2x, and SoM-4x for selected benchmark models."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EVALUATOR = SCRIPT_DIR / "01_evaluate.py"
PYTHON = sys.executable
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

MODEL_TAGS = (
    "llama3_8b_instruct",
    "gemma2_2b_instruct",
    "qwen2_5_3b_instruct",
    "phi4_mini_instruct",
    "mistral_7b_instruct_v03",
)
DEFAULT_MODEL_TAGS = ("llama3_8b_instruct", "gemma2_2b_instruct")
DATASETS = ("boolq", "sciq", "bbh", "arc", "mmlu")
METHODS = ("debate", "som_2x", "som_4x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolved-config-root",
        default="output/benchmark_matrix/resolved_configs",
    )
    parser.add_argument("--output-root", default="output/baselines")
    parser.add_argument("--models", default=",".join(DEFAULT_MODEL_TAGS))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)

    models = _selection(args.models, MODEL_TAGS, "model")
    datasets = _selection(args.datasets, DATASETS, "dataset")
    methods = _selection(args.methods, METHODS, "method")
    devices = [int(item) for item in args.devices.split(",") if item.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("devices must be a non-empty unique list")

    config_root = Path(args.resolved_config_root)
    output_root = Path(args.output_root)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        with _model_run_lock(output_root=output_root, model=model):
            for method in methods:
                for dataset in datasets:
                    config = _config_path(config_root, model=model, dataset=dataset)
                    output_dir = output_root / model / dataset / method
                    _run_task(
                        config=config,
                        output_dir=output_dir,
                        method=method,
                        devices=devices,
                        logs_dir=logs_dir,
                        model=model,
                        dataset=dataset,
                        batch_size=args.batch_size,
                        max_samples=args.max_samples,
                    )


@contextmanager
def _model_run_lock(*, output_root: Path, model: str) -> Iterator[None]:
    """Serialize writers for one model while allowing checkpoint reuse."""
    lock_dir = output_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{model}.lock"
    if os.name != "posix":
        logging.warning("Model-level baseline locking is unavailable on %s", os.name)
        yield
        return

    import fcntl

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        logging.info("Waiting for baseline model lock: %s", lock_path)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        logging.info("Acquired baseline model lock: %s", lock_path)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _run_task(
    *,
    config: Path,
    output_dir: Path,
    method: str,
    devices: list[int],
    logs_dir: Path,
    model: str,
    dataset: str,
    batch_size: int | None,
    max_samples: int | None,
) -> None:
    method_name, agents = _method_args(method)
    common = [
        PYTHON,
        str(EVALUATOR),
        "--config",
        str(config),
        "--method",
        method_name,
        "--agents",
        str(agents),
        "--output-dir",
        str(output_dir),
        "--num-shards",
        str(len(devices)),
    ]
    if batch_size is not None:
        common.extend(["--batch-size", str(batch_size)])
    if max_samples is not None:
        common.extend(["--max-samples", str(max_samples)])
    processes: list[tuple[subprocess.Popen, object, Path]] = []
    logging.info("Starting %s/%s/%s on GPUs %s", model, dataset, method, devices)
    for shard_idx, device in enumerate(devices):
        log_path = logs_dir / f"{model}_{dataset}_{method}_shard{shard_idx}.log"
        handle = open(log_path, "a", encoding="utf-8")
        command = [
            *common,
            "--device",
            str(device),
            "--shard-idx",
            str(shard_idx),
        ]
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=_offline_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, handle, log_path))
    failures = []
    for process, handle, log_path in processes:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append((return_code, log_path))
    if failures:
        raise RuntimeError(f"Baseline shards failed: {failures}")
    merge_command = [*common, "--merge-only", "--shard-idx", "0", "--device", "0"]
    subprocess.run(
        merge_command,
        cwd=PROJECT_ROOT,
        env=_offline_env(),
        check=True,
    )
    logging.info("Completed %s/%s/%s", model, dataset, method)


def _config_path(root: Path, *, model: str, dataset: str) -> Path:
    filename = "original_mmlu_train.yaml" if dataset == "mmlu" else f"original_eval_{dataset}.yaml"
    path = root / model / filename
    if not path.is_file():
        raise FileNotFoundError(f"Resolved benchmark config not found: {path}")
    return path


def _method_args(method: str) -> tuple[str, int]:
    if method == "debate":
        return "debate", 2
    if method == "som_2x":
        return "som", 2
    if method == "som_4x":
        return "som", 4
    raise ValueError(f"Unsupported method: {method}")


def _selection(raw: str, allowed: tuple[str, ...], label: str) -> list[str]:
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid {label} selection; unknown={unknown}")
    return selected


def _offline_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


if __name__ == "__main__":
    main()
