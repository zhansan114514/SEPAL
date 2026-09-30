"""Train one role Actor SFT adapter or finalize the three-role registry."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import logging

from _utils import add_config_argument, load_config, setup_logging
from src.multi_acccollab.config import EXPECTED_ROLES
from src.multi_acccollab.sft import (
    load_role_sft_rows,
    sft_training_fingerprint,
    validate_sft_data_stage,
    write_sft_registry,
)
from src.training.sft_trainer import train_sft
from src.utils.artifacts import completed_adapter_path

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--role", choices=list(EXPECTED_ROLES), default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--finalize-only", action="store_true")
    return parser


def train_role(config, role_name: str, *, device: int | None = None) -> str:
    validate_sft_data_stage(config)
    role = config.role(role_name)
    fingerprint = sft_training_fingerprint(config, role)
    output = config.paths.sft_training_output(role.name)
    completed = completed_adapter_path(output, expected_fingerprint=fingerprint)
    if completed:
        logger.info("Reusing completed %s Actor SFT adapter: %s", role.name, completed)
        return completed
    rows = load_role_sft_rows(config, role.name)
    if len(rows) < config.sft.training.min_examples_per_role:
        raise RuntimeError(
            f"Role {role.name} has {len(rows)} SFT rows, below "
            f"{config.sft.training.min_examples_per_role}"
        )
    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "prompt": [row["prompt"] for row in rows],
            "response": [row["response"] for row in rows],
        }
    )
    base = config.base_config()
    training = config.sft.training
    selected_device = role.device if device is None else int(device)
    train_sft(
        model_name_or_path=base.model.name,
        sft_dataset=dataset,
        output_dir=str(output),
        model_type=base.model.type,
        lora_r=base.training.lora.r,
        lora_alpha=base.training.lora.alpha,
        learning_rate=training.learning_rate,
        batch_size=training.batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        num_epochs=training.epochs,
        max_length=training.max_length,
        warmup_ratio=training.warmup_ratio,
        max_grad_norm=training.max_grad_norm,
        optim=training.optim,
        weight_decay=training.weight_decay,
        seed=config.run.seed + role.seed_offset,
        gradient_checkpointing=True,
        merge_lora=False,
        device=selected_device,
        timeout_per_1k=training.timeout_per_1k,
        save_steps=training.checkpoint_steps,
        save_total_limit=training.checkpoint_total_limit,
        resume_from_checkpoint=training.resume_from_checkpoint,
        training_fingerprint=fingerprint,
        wandb_project="multi-acccollab-sft",
    )
    completed = completed_adapter_path(output, expected_fingerprint=fingerprint)
    if not completed:
        raise RuntimeError(f"SFT runner did not produce a completed adapter: {output}")
    return completed


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    setup_logging(config.run.seed)
    if args.finalize_only:
        write_sft_registry(config)
        return
    if args.role is None:
        for role in config.roles:
            train_role(config, role.name, device=args.device)
        write_sft_registry(config)
        return
    train_role(config, args.role, device=args.device)


if __name__ == "__main__":
    main()
