"""Shared training subprocess and save helpers."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import torch

from src.utils.runtime_env import configure_runtime_libraries


def maybe_merge_lora(
    *,
    merge_lora: bool,
    model_name_or_path: str,
    adapter_dir: str,
    output_dir: str,
    tokenizer,
    torch_dtype,
    fallback_trainer,
    logger,
) -> None:
    if merge_lora and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        logger.info("Merging LoRA weights into base model...")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )
        merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Merged model saved to: %s", output_dir)
        del base_model, merged_model
    elif merge_lora:
        logger.warning("No adapter_config.json found, saving model as-is.")
        fallback_trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        logger.info("Skipping LoRA merge; downstream vLLM will load adapter directly.")


def run_training_subprocess(
    *,
    runner_script: str,
    config: dict,
    dataset,
    device: int,
    timeout_per_1k: int,
    temp_prefix: str,
    dataset_subdir: str,
    logger,
) -> str:
    runner_name = os.path.basename(runner_script)
    if runner_name == "_sft_runner.py":
        training_label = "SFT"
        count_label = "examples"
        saved_dataset_message = "Saved SFT dataset to %s for subprocess training"
    elif runner_name == "_dpo_runner.py":
        training_label = "DPO"
        count_label = "pairs"
        saved_dataset_message = "Saved preference dataset to %s for subprocess training"
    else:
        training_label = "Training"
        count_label = "examples"
        saved_dataset_message = "Saved dataset to %s for subprocess training"

    _prev_cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    _all_gpus = _prev_cuda_vis.split(",") if _prev_cuda_vis else [
        str(i) for i in range(torch.cuda.device_count())
    ]
    if device < len(_all_gpus):
        target_physical = _all_gpus[device].strip()
    else:
        target_physical = str(device)

    temp_dir = tempfile.mkdtemp(prefix=temp_prefix)
    try:
        if dataset is not None:
            dataset_path = os.path.join(temp_dir, dataset_subdir)
            dataset.save_to_disk(dataset_path)
            logger.info(saved_dataset_message, dataset_path)
        else:
            dataset_path = None
        config["dataset_path"] = dataset_path

        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        env = os.environ.copy()
        configure_runtime_libraries(env, preload=False)
        env["CUDA_VISIBLE_DEVICES"] = target_physical
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = project_root + ((":" + existing_pp) if existing_pp else "")
        logger.info(
            "%s training on physical GPU %s (isolated subprocess)",
            training_label,
            target_physical,
        )

        n = len(dataset) if dataset is not None else 1
        timeout = max(1800, timeout_per_1k * math.ceil(n / 1000))
        logger.info(
            "%s timeout: %ss (%s %s, %ss/1k %s)",
            training_label,
            timeout,
            n,
            count_label,
            timeout_per_1k,
            count_label,
        )

        try:
            result = subprocess.run(
                [sys.executable, runner_script, config_path],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            logger.error("%s subprocess timed out after %ss", training_label, timeout)
            raise RuntimeError(f"{training_label} training timed out") from exc

        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    logger.info("[worker] %s", line)
        if result.returncode != 0:
            logger.error("%s subprocess failed (exit %s)", training_label, result.returncode)
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-20:]:
                    logger.error("  %s", line)
            raise RuntimeError(
                f"{training_label} training failed with exit code {result.returncode}"
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return config["output_dir"] if config.get("merge_lora") else config["output_dir"] + "_adapter"
