# SEPAL reproducibility supplement

This anonymous package accompanies *SEPAL: Separated Expert Pairs with Answer-Level Fusion for Reliable LLM Collaboration*.

## Audit the reported results

From this directory, use Python 3.10 or newer:

```bash
python reproduce/verify_results.py
python reproduce/verify_decisions.py
```

The first command checks the paper CSVs against aggregate experiment artifacts and the file inventory. The second reconstructs votes and correctness from 125 compressed decision files: 578,000 example records, 2,427,600 round decisions, and 4,950 aggregate checks. Neither command performs model inference or modifies stored evidence. Use `python reproduce/verify_decisions.py --output audit/decision_check.json` to save a new report outside `results/`. Full-R0 and Full-R4 come from different rounds of the same stored trajectories.

All six completed component variants are included: SFT-only, SFT+Base-C, SFT+Trained-C, Full-R0, No-SFT, and Full-R4. Simpler configurations lead on several backbones. These results are reported in the manuscript.

## Contents

- `code/`: source snapshot, parsers, training and inference entry points, and 290 tests.
- `configs/`: 50 recorded training/evaluation configurations for five backbones.
- `results/ablations/`: full-system and component metrics, manifests, and resolved role configurations.
- `results/extended_ablations/`: SFT+Trained-C and No-SFT metrics for all 25 cells.
- `results/decisions/`: sample IDs, labels, role answers, and round decisions, with source SHA-256 hashes.
- `results/main/` and `results/locked/`: baseline artifacts and the paper result matrix.
- `results/decision_audit.json`: per-example verification and correction/regression counts.
- `PROVENANCE_MANIFEST.json`: packaged file sizes and SHA-256 hashes.

Question text, full reasoning traces, model weights, and adapters are outside this compact package. Public checkpoint names, configurations, and recorded adapter identities identify the original runs. The accompanying review checked 1,734,000 stored raw Actor responses against the full-system decision files, with zero mismatches. This verifies the stored predictions; it is not a fresh training replication.

## Tests

Install the source dependencies in a suitable Linux environment, then run:

```bash
python -m pip install -e "code[dev]"
python reproduce/run_tests.py
```

The helper sets the correct working directory. The reviewed snapshot passed 290 tests; two third-party SWIG deprecation warnings were emitted. Tests exercise parsing, voting, prompt formatting, configuration, orchestration, and training setup. They do not launch GPU training.

## Prepare a full rerun

Use an empty destination and provide local checkpoint and benchmark directories:

```bash
python -m pip install pyyaml
python reproduce/prepare_workspace.py --workspace /work/sepal-rerun \
  --model-root /models --data-root /data/acccollab_paper
```

This copies the code, resolves 50 configurations, and changes only filesystem paths. All 50 prepared configurations were validated using the released configuration loaders. Checkpoints should appear under `--model-root` with the basename shown in each configuration. Obtain the official datasets through the providers cited in the paper.

Run training from the prepared workspace:

```bash
cd /work/sepal-rerun
python -m pip install -e ".[dev]"
python scripts/multi_acccollab/05_pipeline.py \
  --config configs/reproduction/qwen25_3b/multi_mmlu_train.yaml
```

Use `llama3_8b`, `gemma2_2b`, `phi4_mini`, or `mistral7b_v03` for the other backbones. The `original_mmlu_train.yaml` configurations use `scripts/acccollab/06_pipeline.py` for the matched single pair. The transfer-evaluation configurations and recorded per-role files retain their adapter source paths. Finish training before invoking transfer evaluation or the component runners. For example, `python scripts/multi_acccollab/06_evaluate_dataset.py --config configs/reproduction/qwen25_3b/multi_eval_boolq.yaml --devices 0,1,2,3` evaluates BoolQ after training. Add `--dry-run` to validate the evaluation schedule without inference.

The recorded configurations allocate up to four 80 GB A800 GPUs. Map their device lists to available GPUs before a new run and record that change. Phi uses a 4,096-token model context; the others use 8,192. Mistral samples five preference trajectories per training question; the others sample one. Every headline evaluation cell uses one trial. Tests were checked on Python 3.13.13, PyTorch 2.12.1, TRL 1.7.0, Transformers 5.12.1, and PEFT 0.19.1. These describe the audit environment, not a reconstructed historical inference lockfile.

New stochastic runs should record their seeds, dependency versions, configuration hashes, and output artifacts. The locked records provide measured reference values; an independent full rerun tests their robustness.
