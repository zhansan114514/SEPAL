"""Generate, merge, or aggregate one policy-lattice ablation."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.runtime import configure_current_ablation_process

configure_current_ablation_process(PROJECT_ROOT)

from src.ablations.config import POLICY_VARIANTS, ROLE_NAMES, load_ablation_manifest
from src.ablations.policy_lattice import (
    aggregate_role_records,
    generate_variant_shard,
    merge_variant_shards,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", required=True, choices=POLICY_VARIANTS)
    parser.add_argument("--role", choices=ROLE_NAMES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int)
    parser.add_argument("--shard-idx", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    selected_modes = int(args.merge) + int(args.aggregate)
    if selected_modes > 1:
        parser.error("--merge and --aggregate are mutually exclusive")
    manifest = load_ablation_manifest(args.manifest)
    dataset = dict(dict(manifest["datasets"])[args.dataset])
    expected_samples = int(dataset["expected_samples"])
    logical_shards = int(dataset["logical_shards"])
    output = Path(args.output_dir)

    if args.aggregate:
        aggregate_role_records(
            {
                role: output / f"roles/{role}/records.jsonl"
                for role in ROLE_NAMES
            },
            output_dir=output / "aggregate",
            dataset_name=args.dataset,
            variant=args.variant,
            expected_samples=expected_samples,
            reparse_completions=True,
        )
        return

    if args.role is None:
        parser.error("--role is required for shard generation and merge")
    role_output = output / f"roles/{args.role}"
    num_shards = args.num_shards if args.num_shards is not None else logical_shards
    if args.merge:
        merge_variant_shards(
            args.manifest,
            dataset_name=args.dataset,
            variant=args.variant,
            role_name=args.role,
            output_dir=role_output,
            num_shards=num_shards,
            project_root=PROJECT_ROOT,
        )
        return
    if args.device is None or args.shard_idx is None:
        parser.error("--device and --shard-idx are required for shard generation")
    generate_variant_shard(
        args.manifest,
        dataset_name=args.dataset,
        variant=args.variant,
        role_name=args.role,
        output_dir=role_output,
        device=args.device,
        shard_idx=args.shard_idx,
        num_shards=num_shards,
        project_root=PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()
