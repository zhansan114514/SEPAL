"""Validate a completed benchmark matrix and write comparable CSV/JSON results."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_matrix.results import collect_matrix_results, write_matrix_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Path to the matrix plan.json")
    parser.add_argument(
        "--output-dir",
        help="Summary directory (defaults to a results directory next to plan.json)",
    )
    args = parser.parse_args()
    plan = Path(args.plan)
    output_dir = Path(args.output_dir) if args.output_dir else plan.parent / "results"
    results = collect_matrix_results(plan, project_root=PROJECT_ROOT)
    csv_path, json_path = write_matrix_summary(results, output_dir=output_dir)
    print(f"Validated {len(results)} benchmark rows")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
