"""
DPO training runner for isolated subprocess execution.

This module is called by dpo_trainer.py via subprocess to run DPO training
in a fresh CUDA context where CUDA_VISIBLE_DEVICES works correctly.
It reads config from a JSON file and loads the dataset from disk.
"""

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


def _format_preference_dataset(
    preference_dataset,
    *,
    model_type: str | None,
    tokenizer=None,
):
    """Apply the model's training prompt template without touching completions."""
    from src.training.chat_format import (
        format_preference_example,
        training_prompt_format_version,
    )

    version = training_prompt_format_version(model_type)
    if version == "raw_prompt_v1":
        return preference_dataset
    logger.info("Formatting DPO prompts with template: %s", version)
    return preference_dataset.map(
        lambda example: format_preference_example(
            example,
            model_type=model_type,
            tokenizer=tokenizer,
        ),
        desc=f"Formatting DPO prompts ({version})",
    )


def _prepare_policy_model(
    model,
    *,
    initial_lora_path: str | None,
    model_type: str,
    lora_r: int,
    lora_alpha: int,
):
    """Prepare the policy/reference setup for a fresh or continuing DPO job.

    For the first ACC-Collab alternation there is no SFT adapter: TRL receives
    the base model plus ``peft_config`` and computes reference log-probabilities
    with the new adapter disabled.  For later alternations, the prior role
    adapter is loaded as trainable ``default``; TRL then snapshots it into a
    frozen ``ref`` adapter.  Keeping this branch explicit prevents the first
    original-protocol DPO stage from depending on an invented SFT phase.
    """
    from src.training.adapter_utils import resolve_lora_adapter_path
    from src.training.lora_config import get_lora_config

    requested = str(initial_lora_path or "").strip()
    if not requested:
        peft_config = get_lora_config(
            model_type=model_type,
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=0.0,
        )
        logger.info("Starting DPO from the base model with a new trainable LoRA.")
        return model, peft_config, None

    from peft import PeftModel

    resolved = resolve_lora_adapter_path(requested)
    logger.info("Continuing DPO from initial LoRA adapter: %s", resolved)
    model = PeftModel.from_pretrained(
        model,
        resolved,
        adapter_name="default",
        is_trainable=True,
    )
    model.set_adapter("default")
    return model, None, resolved


def _precompute_train_ref_log_probs_with_cache(trainer, cfg: dict) -> None:
    """Persist DPO reference log-prob precompute batches for crash recovery.

    TRL >=1.0 precomputes reference log-probs during ``DPOTrainer`` construction
    and writes them onto ``train_dataset`` as ``ref_chosen_logps`` /
    ``ref_rejected_logps`` columns (computed with the frozen ``ref`` adapter).
    On those builds the reference log-probs are already materialized by the time
    we get here, so this disk cache -- used by older TRL where precompute was
    lazy -- simply short-circuits via the column check below.
    """
    precompute_on = bool(
        getattr(trainer, "precompute_ref_logps", None)
        or getattr(trainer, "precompute_ref_log_probs", None)
    )
    if not precompute_on or bool(getattr(trainer, "_precomputed_train_ref_log_probs", False)):
        return

    existing_columns = set(getattr(trainer.train_dataset, "column_names", []) or [])
    if {"ref_chosen_logps", "ref_rejected_logps"}.issubset(existing_columns):
        trainer._precomputed_train_ref_log_probs = True
        return

    import shutil
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    output_dir = Path(cfg["output_dir"])
    cache_dir = output_dir / "ref_log_probs_cache"
    batch_dir = cache_dir / "batches"
    metadata_path = cache_dir / "metadata.json"
    batch_size = trainer.args.precompute_ref_batch_size or trainer.args.per_device_train_batch_size
    expected_metadata = {
        "num_samples": len(trainer.train_dataset),
        "batch_size": int(batch_size),
        "model_name_or_path": cfg["model_name_or_path"],
        "initial_lora_path": cfg["initial_lora_path"],
        "dpo_max_length": cfg["dpo_max_length"],
        "dpo_max_prompt_length": cfg["dpo_max_prompt_length"],
        "dpo_max_completion_length": cfg["dpo_max_completion_length"],
        "truncation_mode": trainer.args.truncation_mode,
        "model_type": cfg.get("model_type"),
        "prompt_format_version": cfg.get("prompt_format_version"),
    }

    if metadata_path.exists():
        try:
            with open(metadata_path, encoding="utf-8") as f:
                existing_metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing_metadata = {}
        cache_identity_keys = (
            "num_samples",
            "model_name_or_path",
            "initial_lora_path",
            "dpo_max_length",
            "dpo_max_prompt_length",
            "dpo_max_completion_length",
            "truncation_mode",
            "model_type",
            "prompt_format_version",
        )
        identity_matches = all(
            existing_metadata.get(k) == expected_metadata[k]
            for k in cache_identity_keys
        )
        if identity_matches and existing_metadata.get("complete") is True:
            try:
                batch_paths = sorted(batch_dir.glob("batch_*.pt"))
                ref_chosen_logps = []
                ref_rejected_logps = []
                for batch_path in batch_paths:
                    cached = torch.load(batch_path, map_location="cpu")
                    ref_chosen_logps.append(cached["chosen"])
                    ref_rejected_logps.append(cached["rejected"])

                expected_samples = expected_metadata["num_samples"]
                all_ref_chosen_logps = (
                    torch.cat(ref_chosen_logps).float().numpy()[:expected_samples]
                )
                all_ref_rejected_logps = (
                    torch.cat(ref_rejected_logps).float().numpy()[:expected_samples]
                )
                if (
                    len(all_ref_chosen_logps) == expected_samples
                    and len(all_ref_rejected_logps) == expected_samples
                ):
                    trainer.train_dataset = trainer.train_dataset.add_column(
                        name="ref_chosen_logps",
                        column=all_ref_chosen_logps,
                    )
                    trainer.train_dataset = trainer.train_dataset.add_column(
                        name="ref_rejected_logps",
                        column=all_ref_rejected_logps,
                    )
                    trainer._precomputed_train_ref_log_probs = True
                    logger.info("Loaded complete reference log-prob cache: %s", cache_dir)
                    return
            except (KeyError, OSError, RuntimeError, EOFError) as e:
                logger.warning("Ignoring invalid reference log-prob cache: %s", e)

        if any(existing_metadata.get(k) != v for k, v in expected_metadata.items()):
            logger.info("Discarding stale reference log-prob cache: %s", cache_dir)
            shutil.rmtree(cache_dir, ignore_errors=True)

    batch_dir.mkdir(parents=True, exist_ok=True)
    metadata_payload = {
        **expected_metadata,
        "complete": False,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, ensure_ascii=False, indent=2)

    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": trainer.data_collator,
        "num_workers": trainer.args.dataloader_num_workers,
        "pin_memory": trainer.args.dataloader_pin_memory,
        "shuffle": False,
    }
    data_loader = trainer.accelerator.prepare(
        DataLoader(trainer.train_dataset, **dataloader_params)
    )

    ref_chosen_logps = []
    ref_rejected_logps = []
    logger.info("Precomputing train reference log probs with cache: %s", cache_dir)
    for batch_index, padded_batch in enumerate(
        tqdm(iterable=data_loader, desc="Train dataset reference log probs")
    ):
        batch_path = batch_dir / f"batch_{batch_index:05d}.pt"
        cached = None
        if batch_path.exists():
            try:
                cached = torch.load(batch_path, map_location="cpu")
            except (OSError, RuntimeError, EOFError):
                logger.warning("Ignoring unreadable ref log-prob batch: %s", batch_path)
                batch_path.unlink(missing_ok=True)

        if cached is not None:
            ref_chosen_logp = cached["chosen"]
            ref_rejected_logp = cached["rejected"]
        else:
            ref_chosen_logp, ref_rejected_logp = trainer.compute_ref_log_probs(
                padded_batch
            )
            ref_chosen_logp, ref_rejected_logp = trainer.accelerator.gather_for_metrics(
                (ref_chosen_logp, ref_rejected_logp)
            )
            ref_chosen_logp = ref_chosen_logp.cpu()
            ref_rejected_logp = ref_rejected_logp.cpu()
            tmp_path = batch_path.with_suffix(".tmp")
            torch.save(
                {"chosen": ref_chosen_logp, "rejected": ref_rejected_logp},
                tmp_path,
            )
            tmp_path.replace(batch_path)

            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
            trainer.accelerator.free_memory()

        ref_chosen_logps.append(ref_chosen_logp)
        ref_rejected_logps.append(ref_rejected_logp)

    expected_samples = expected_metadata["num_samples"]
    all_ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()[:expected_samples]
    all_ref_rejected_logps = (
        torch.cat(ref_rejected_logps).float().numpy()[:expected_samples]
    )

    trainer.train_dataset = trainer.train_dataset.add_column(
        name="ref_chosen_logps",
        column=all_ref_chosen_logps,
    )
    trainer.train_dataset = trainer.train_dataset.add_column(
        name="ref_rejected_logps",
        column=all_ref_rejected_logps,
    )
    trainer._precomputed_train_ref_log_probs = True

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **expected_metadata,
                "complete": True,
                "num_batches": len(ref_chosen_logps),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Reference log-prob cache ready: %s", cache_dir)


def _prepare_resume_checkpoint_without_optimizer(
    resume_checkpoint: str,
    output_dir: str,
) -> str:
    """Create a checkpoint view that omits optimizer state."""
    import shutil
    from pathlib import Path

    source = Path(resume_checkpoint)
    target = Path(output_dir) / ".resume_no_optimizer" / source.name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    excluded = {"optimizer.pt", "optimizer.bin"}
    for item in source.iterdir():
        if item.name in excluded:
            continue
        destination = target / item.name
        if item.is_dir():
            shutil.copytree(item, destination, symlinks=True)
            continue
        try:
            os.link(item, destination)
        except OSError:
            shutil.copy2(item, destination)

    logger.info("Prepared resume checkpoint without optimizer state: %s", target)
    return str(target)


def _run():
    if len(sys.argv) < 2:
        print("Usage: python _dpo_runner.py <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = json.load(f)

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig

    from src.training._common import maybe_merge_lora
    from src.utils.torch_dtype import resolve_training_dtype

    # Load preference dataset from disk
    dataset_path = cfg["dataset_path"]
    if dataset_path and os.path.exists(dataset_path):
        preference_dataset = load_from_disk(dataset_path)
        logger.info(f"Loaded {len(preference_dataset)} preference samples from {dataset_path}")
    else:
        logger.error(f"Dataset not found at {dataset_path}")
        sys.exit(1)

    torch_dtype, use_bf16 = resolve_training_dtype(cfg["model_name_or_path"])

    # max_length is the only length TRL >=1.0 honors; the per-part budgets are
    # carried in the reference-logprob cache identity only (see metadata below).
    dpo_max_length = int(cfg["dpo_max_length"])

    # Load model on the single visible GPU (CUDA_VISIBLE_DEVICES is set by parent)
    logger.info(f"Loading model: {cfg['model_name_or_path']} (dtype={torch_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    try:
        model, peft_config, initial_lora_path = _prepare_policy_model(
            model,
            initial_lora_path=cfg.get("initial_lora_path"),
            model_type=str(cfg.get("model_type") or "llama3"),
            lora_r=int(cfg["lora_r"]),
            lora_alpha=int(cfg["lora_alpha"]),
        )
    except Exception as exc:
        logger.error("Failed to prepare DPO policy model: %s", exc)
        sys.exit(1)
    cfg["initial_lora_path"] = initial_lora_path

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from src.training.chat_format import training_prompt_format_version

    cfg["prompt_format_version"] = training_prompt_format_version(
        cfg.get("model_type")
    )
    preference_dataset = _format_preference_dataset(
        preference_dataset,
        model_type=cfg.get("model_type"),
        tokenizer=tokenizer,
    )

    # DPO training arguments. TRL names the chosen-completion NLL term "sft",
    # but this is only a DPO regularizer here, not a separate Critic SFT stage.
    loss_type_val = cfg.get("loss_type", "sigmoid")
    nll_weight = cfg.get("nll_weight", 1.0)

    if nll_weight > 0:
        effective_loss_type = [loss_type_val, "sft"]
        effective_loss_weights = [1.0, nll_weight]
        logger.info(
            "Using DPO + chosen-completion NLL regularization: "
            "loss_type=%s, weights=%s",
            effective_loss_type,
            effective_loss_weights,
        )
    else:
        effective_loss_type = loss_type_val
        effective_loss_weights = None
        logger.info(f"Using pure DPO loss: loss_type={effective_loss_type}")

    save_steps = cfg.get("save_steps")
    try:
        save_steps = int(save_steps) if save_steps is not None else None
    except (TypeError, ValueError):
        save_steps = None
    save_total_limit = cfg.get("save_total_limit", 2)
    try:
        save_total_limit = int(save_total_limit) if save_total_limit is not None else None
    except (TypeError, ValueError):
        save_total_limit = 2

    save_kwargs = {}
    if save_steps is not None and save_steps > 0:
        save_kwargs["save_strategy"] = "steps"
        save_kwargs["save_steps"] = save_steps
        logger.info(
            "DPO checkpointing enabled: save_steps=%s, save_total_limit=%s",
            save_steps,
            save_total_limit,
        )
    else:
        save_kwargs["save_strategy"] = "epoch"
        logger.info("DPO checkpointing uses epoch saves only.")
    if save_total_limit is not None and save_total_limit > 0:
        save_kwargs["save_total_limit"] = save_total_limit

    training_args = DPOConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        max_length=dpo_max_length,
        beta=cfg["beta"],
        loss_type=effective_loss_type,
        loss_weights=effective_loss_weights,
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"],
        weight_decay=cfg["weight_decay"],
        seed=cfg["seed"],
        logging_steps=10,
        bf16=use_bf16,
        fp16=(not use_bf16 and torch_dtype != torch.float32),
        gradient_checkpointing=cfg["gradient_checkpointing"],
        remove_unused_columns=False,
        report_to="wandb" if cfg.get("use_wandb") else "none",
        run_name=cfg.get("wandb_project") if cfg.get("use_wandb") else None,
        # Precompute reference log-probs to avoid holding two models in GPU memory.
        # TRL >=1.0 computes them with the frozen "ref" adapter (see model setup).
        precompute_ref_log_probs=True,
        **save_kwargs,
    )

    # Initialize DPO trainer
    import trl
    trl_version = tuple(int(x) for x in trl.__version__.split('.')[:2])
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=preference_dataset,
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    if trl_version >= (0, 12):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    # Compatibility shim: trl >=0.24 tries to set model.warnings_issued
    # but PEFT-wrapped models may not expose it from the underlying PreTrainedModel.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = DPOTrainer(**trainer_kwargs)

    _precompute_train_ref_log_probs_with_cache(trainer, cfg)

    resume_checkpoint = None
    if cfg.get("resume_from_checkpoint", True):
        from transformers.trainer_utils import get_last_checkpoint

        if os.path.isdir(cfg["output_dir"]):
            resume_checkpoint = get_last_checkpoint(cfg["output_dir"])
        if resume_checkpoint:
            logger.info("Resuming DPO training from checkpoint: %s", resume_checkpoint)
            if not cfg.get("resume_optimizer_state", True):
                resume_checkpoint = _prepare_resume_checkpoint_without_optimizer(
                    resume_checkpoint,
                    cfg["output_dir"],
                )
        else:
            logger.info("No DPO checkpoint found under %s; starting fresh.", cfg["output_dir"])

    logger.info("Starting DPO training...")
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    logger.info("Training complete.")

    # Save LoRA adapter. vLLM supports loading LoRA adapters directly, so
    # development runs skip the expensive full-model merge by default.
    output_dir = cfg["output_dir"]
    adapter_dir = output_dir + "_adapter"
    trainer.model.set_adapter("default")
    trainer.model.save_pretrained(adapter_dir, selected_adapters=["default"])
    tokenizer.save_pretrained(adapter_dir)
    logger.info(f"LoRA adapter saved to: {adapter_dir}")

    # Mark the adapter durably complete so the pipeline can finalize its registry
    # and skip already-trained agents on re-entry. Without this marker the
    # orchestrator's ``--finalize-only`` step cannot detect a finished adapter.
    from src.utils.artifacts import stable_fingerprint, write_adapter_success

    derived_fingerprint = stable_fingerprint({
        "training_kind": "dpo",
        "base_model": cfg["model_name_or_path"],
        "initial_lora_path": initial_lora_path,
        "lora_r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "num_train_examples": len(trainer.train_dataset),
        "num_epochs": cfg["num_epochs"],
        "beta": cfg["beta"],
        "actor_nll_weight": cfg.get("nll_weight"),
        "loss_type": cfg.get("loss_type"),
        "model_type": cfg.get("model_type"),
        "prompt_format_version": cfg.get("prompt_format_version"),
    })
    write_adapter_success(
        adapter_dir,
        training_fingerprint=str(cfg.get("training_fingerprint") or derived_fingerprint),
        base_model=cfg["model_name_or_path"],
        output_dir=output_dir,
        training_kind="dpo",
    )
    logger.info("Marked DPO adapter complete: %s", adapter_dir)

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

    logger.info("DPO runner finished successfully.")


if __name__ == "__main__":
    _run()
