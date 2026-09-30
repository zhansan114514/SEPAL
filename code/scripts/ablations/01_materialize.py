"""Authenticate completed policies and materialize one model's ablation plan."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.config import MODEL_LAYOUTS, materialize_ablation_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_LAYOUTS))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    path = materialize_ablation_manifest(
        args.model,
        output_root=args.output_root,
        model_path=args.model_path,
        project_root=PROJECT_ROOT,
    )
    print(path)


if __name__ == "__main__":
    main()
