"""SFT training runner for isolated subprocess execution."""

from __future__ import annotations

import gc
import json
import logging
import os
import sys

from src.utils.nvml_patch import apply_nvml_cdll_patch

apply_nvml_cdll_patch()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _tokenize_response_only(
    example,
    tokenizer,
    max_length: int,
    model_type: str | None = None,
):
    """Tokenize prompt+response and mask prompt tokens with -100 labels."""
    from src.training.chat_format import (
        format_training_prompt,
        uses_gemma2_chat_template,
    )

    prompt = format_training_prompt(
        str(example.get("prompt") or ""),
        model_type=model_type,
        tokenizer=tokenizer,
    )
    response = str(example.get("response") or "")
    if not response:
        raise ValueError("SFT example has empty response")

    prompt_ids = tokenizer(
        prompt,
        # Gemma's apply_chat_template output already starts with <bos>.
        add_special_tokens=not uses_gemma2_chat_template(model_type),
        truncation=False,
    )["input_ids"]
    response_ids = tokenizer(
        response,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    if (
        prompt_ids
        and response_ids
        and prompt_ids[0] == response_ids[0]
        and getattr(tokenizer, "bos_token_id", None) == response_ids[0]
    ):
        response_ids = response_ids[1:]
    if not response_ids:
        raise ValueError("SFT example response produced no tokens")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and (
        not response_ids or response_ids[-1] != eos_token_id
    ):
        response_ids.append(eos_token_id)

    if len(response_ids) >= max_length:
        input_ids = response_ids[:max_length]
        labels = list(input_ids)
    else:
        prompt_budget = max_length - len(response_ids)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class ResponseOnlyDataCollator:
    def __init__(self, tokenizer, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        pad_id = self.tokenizer.pad_token_id
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _run():
    if len(sys.argv) < 2:
        print("Usage: python _sft_runner.py <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = json.load(f)

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from src.training._common import maybe_merge_lora
    from src.training.chat_format import training_prompt_format_version
    from src.training.lora_config import get_lora_config
    from src.utils.torch_dtype import resolve_training_dtype

    dataset_path = cfg["dataset_path"]
    if dataset_path and os.path.exists(dataset_path):
        raw_dataset = load_from_disk(dataset_path)
        logger.info("Loaded %s SFT examples from %s", len(raw_dataset), dataset_path)
    else:
        logger.error("Dataset not found at %s", dataset_path)
        sys.exit(1)

    torch_dtype, use_bf16 = resolve_training_dtype(cfg["model_name_or_path"])

    lora_config = get_lora_config(
        cfg["model_type"],
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
    )

    logger.info("Loading model: %s (dtype=%s)", cfg["model_name_or_path"], torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    from peft import get_peft_model

    model = get_peft_model(model, lora_config)
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    tokenized = raw_dataset.map(
        lambda example: _tokenize_response_only(
            example,
            tokenizer=tokenizer,
            max_length=cfg["max_length"],
            model_type=cfg.get("model_type"),
        ),
        remove_columns=list(raw_dataset.column_names),
        desc="Tokenizing response-only SFT data",
    )

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"],
        weight_decay=cfg["weight_decay"],
        seed=cfg["seed"],
        logging_steps=10,
        save_strategy=("steps" if int(cfg.get("save_steps") or 0) > 0 else "epoch"),
        save_steps=(int(cfg["save_steps"]) if int(cfg.get("save_steps") or 0) > 0 else None),
        save_total_limit=(int(cfg["save_total_limit"]) if cfg.get("save_total_limit") else None),
        bf16=use_bf16,
        fp16=(not use_bf16 and torch_dtype != torch.float32),
        gradient_checkpointing=cfg["gradient_checkpointing"],
        remove_unused_columns=False,
        report_to="wandb" if cfg.get("use_wandb") else "none",
        run_name=cfg.get("wandb_project") if cfg.get("use_wandb") else None,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized,
        "data_collator": ResponseOnlyDataCollator(tokenizer),
    }
    import inspect

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    resume_checkpoint = None
    if cfg.get("resume_from_checkpoint", True):
        from transformers.trainer_utils import get_last_checkpoint

        if os.path.isdir(cfg["output_dir"]):
            resume_checkpoint = get_last_checkpoint(cfg["output_dir"])
        if resume_checkpoint:
            logger.info("Resuming SFT training from checkpoint: %s", resume_checkpoint)
        else:
            logger.info("No SFT checkpoint found under %s; starting fresh.", cfg["output_dir"])

    logger.info("Starting SFT training...")
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    logger.info("Training complete.")

    output_dir = cfg["output_dir"]
    adapter_dir = output_dir + "_adapter"
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("LoRA adapter saved to: %s", adapter_dir)

    # Mark the adapter durably complete so the pipeline can finalize its registry
    # and skip already-trained agents on re-entry. Without this marker the
    # orchestrator's ``--finalize-only`` step cannot detect a finished adapter.
    from src.utils.artifacts import stable_fingerprint, write_adapter_success

    write_adapter_success(
        adapter_dir,
        training_fingerprint=str(
            cfg.get("training_fingerprint")
            or stable_fingerprint({
                "training_kind": "sft",
                "base_model": cfg["model_name_or_path"],
                "lora_r": cfg["lora_r"],
                "lora_alpha": cfg["lora_alpha"],
                "num_train_examples": len(trainer.train_dataset),
                "num_epochs": cfg["num_epochs"],
                "max_length": cfg["max_length"],
                "model_type": cfg.get("model_type"),
                "prompt_format_version": training_prompt_format_version(
                    cfg.get("model_type")
                ),
            })
        ),
        base_model=cfg["model_name_or_path"],
        output_dir=output_dir,
        training_kind="sft",
    )
    logger.info("Marked SFT adapter complete: %s", adapter_dir)

    maybe_merge_lora(
        merge_lora=cfg.get("merge_lora", False),
        model_name_or_path=cfg["model_name_or_path"],
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        fallback_trainer=trainer,
        logger=logger,
    )

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass

    logger.info("SFT runner finished successfully.")


if __name__ == "__main__":
    _run()
