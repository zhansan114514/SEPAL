# Multi-role ACC-Collab (SEPAL)

This directory contains the main three-role pipeline used by the paper. It trains
independent Direct, Evidence, and Verification Actor--Critic pairs and
aggregates only their final-round answers. The final decision is a two-of-three
majority with a deterministic Direct fallback; no Judge is loaded or called.

## Pipeline

For each backbone, the pipeline samples 10,000 MMLU `auxiliary_train` questions,
generates role-specific candidates, and keeps the common questions with an
eligible target for all three roles in balanced SFT. The Critic starts from the
base model. On the full
1,531-question MMLU validation split, the pipeline runs:

1. Critic guided-trajectory preference generation (`K=10`, `epsilon=0.6`);
2. Critic DPO;
3. Actor preference generation using the trained Critic;
4. Actor DPO continued from the role SFT adapter;
5. five-round, single-trial evaluation and final-answer aggregation.

R0 is the initial Actor answer and R1--R4 are four private Critic-conditioned
revisions. The three role histories remain separate throughout evaluation.

## Run from the public release

The public repository keeps code and path-marked configs in separate directories.
First create an empty prepared workspace from the repository root:

```bash
python reproduce/prepare_workspace.py \
  --workspace /work/sepal-rerun \
  --model-root /models \
  --data-root /data/acccollab_paper
cd /work/sepal-rerun
python -m pip install -e ".[dev]"
```

Then run a resolved multi-role config:

```bash
python scripts/multi_acccollab/05_pipeline.py \
  --config configs/reproduction/llama3_8b/multi_mmlu_train.yaml
```

The same directory contains `multi_eval_{boolq,mmlu,bbh,sciq,arc}.yaml` for
evaluation after role adapters are available. Use
`06_evaluate_dataset.py --dry-run` to validate the schedule before inference.
The prepared configs for `qwen25_3b`, `gemma2_2b`, `phi4_mini`, and
`mistral7b_v03` follow the same layout.

Large outputs are written below the configured `output_dir`. Resolved role YAMLs
and manifests record adapter initialization, role prompts, device and seed
offsets, and the original ACC-Collab hyperparameters. For the complete paper
audit and the distinction between a locked-result audit and a fresh rerun, see
the repository-level [README.md](../../../README.md).
