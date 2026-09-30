# SEPAL

This repository is the public reproducibility release for *SEPAL:
Separated Expert Pairs with Answer-Level Fusion for Reliable LLM Collaboration*.
It contains the experiment source, sanitized configurations, locked result
artifacts, compact per-example decisions, and the scripts used to audit or rerun
the experiments. The paper source and model weights are outside this repository.

## Method in the paper

SEPAL addresses a tension in multi-agent question answering: feedback can repair
an answer, but shared feedback can also make several candidates inherit the same
mistake. Three independent Actor--Critic teams therefore keep feedback local and
combine only their final answers.

The teams use fixed role objectives:

- **Direct** focuses on the decisive fact or calculation.
- **Evidence** grounds the answer in a relevant definition, fact, passage, or
  domain principle.
- **Verification** solves independently and checks likely alternatives or failure
  modes.

Each role has its own Actor and Critic adapters and conversation history. The
Actor produces R0, then four private Critic-conditioned revisions produce R1--R4.
Only the three parsed R4 answers cross team boundaries. A valid answer appearing
at least twice is returned; otherwise the parsed Direct answer is used. There is
no learned Judge or cross-team message.

Training follows the paper contract. For each backbone, the pipeline samples
10,000 MMLU `auxiliary_train` questions and generates role-specific candidates;
balanced SFT keeps the common questions with an eligible target for all three
roles. The Critic and then the Actor are trained with continuation-valued
preferences from the full 1,531-question MMLU validation split. Preference
construction uses `K=10`
one-step continuations and margin `epsilon=0.6`. Released configs use LoRA rank
256 and scaling factor 512.

## Paper--repository map

| Paper component | Implementation or released evidence |
| --- | --- |
| Three private role pairs | `code/scripts/multi_acccollab/05_pipeline.py`, `code/src/multi_acccollab/`, and `configs/*/multi_*.yaml` |
| Role prompts | `code/src/acccollab/prompts.py` and the `roles` blocks in multi-role configs |
| Role SFT on 10,000 questions | `code/scripts/multi_acccollab/01_build_actor_sft_data.py` |
| Critic-then-Actor preference learning | `code/scripts/acccollab/` and `code/src/acccollab/` |
| Private revision and final fusion | `code/scripts/multi_acccollab/04_aggregate_majority.py` and `results/decisions/` |
| Direct, Debate, SoM-2, SoM-4, and matched ACC baselines | `code/src/baselines/`, `code/scripts/acccollab/`, and `results/main/` |
| Component and round ablations | `code/scripts/ablations/`, `results/ablations/`, and `results/extended_ablations/` |
| Paper tables and diagnostics | `results/locked/`, `results/main/`, and `results/decision_audit.json` |

The reported paper matrix uses `multi_acccollab`. Other modules provide shared
training, parsing, evaluation, and historical protocol support; they are not
alternate names for the reported SEPAL method.

## Reported evaluation

The paper evaluates these five open-weight instruction backbones:

| Backbone | Public checkpoint identifier |
| --- | --- |
| Llama-3-8B | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B-Instruct` |
| Gemma-2-2B | `google/gemma-2-2b-it` |
| Phi-4-mini | `microsoft/Phi-4-mini-instruct` |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` |

| Dataset | Split | Examples | Use |
| --- | --- | ---: | --- |
| MMLU | test | 14,042 | in-domain |
| BoolQ | validation | 3,270 | transfer |
| BBH | 22-category stratified subset | 1,260 | transfer |
| SciQ | test | 1,000 | transfer |
| ARC | Easy + Challenge test | 3,548 | transfer |

Compared with the matched single Actor--Critic pair, SEPAL improves macro
accuracy for all five backbones by 1.06--2.22 percentage points, averaging
+1.81 points, and wins 24 of 25 model--dataset cells. The final vote is better
than the strongest individual role in 21 of 25 cells. Ablations show that
Critic-conditioned revision supplies the largest component gain and that later
revisions have diminishing returns. These are descriptive results from one fixed
trial per cell; the paper does not claim cross-seed significance or
equal-compute superiority.

## Repository layout

```text
code/
  src/                  data, prompts, training, inference, parsing, evaluation
  scripts/acccollab/    original single-pair ACC-Collab stages
  scripts/multi_acccollab/
                        three-role SEPAL pipeline and evaluation entry points
  scripts/ablations/    component and round ablation runners
  tests/                unit and integration tests
configs/                five model directories with ten recorded configs each
results/
  main/                 baseline and matched comparison artifacts
  locked/               CSVs consumed by the paper audit
  ablations/            full-system and policy-lattice manifests
  extended_ablations/   SFT+Trained-C and No-SFT aggregates
  decisions/            compact per-example records for five conditions
reproduce/               audit, workspace preparation, and test helpers
PROVENANCE_MANIFEST.json exact file inventory and SHA-256 values
```

The release contains six reported ablation conditions: `sft_only`,
`sft_base_critic`, `sft_trained_critic`, `no_sft_full`, `Full-R0`, and `Full-R4`.
It contains 50 recorded configurations, 125 compressed decision files, and 660
manifest entries. We do not release model weights, adapters, benchmark question
text, raw generations, or full reasoning traces.

## Audit the released results

These commands use only packaged artifacts. They do not download models, run
inference, or modify `results/`.

```bash
python reproduce/verify_results.py
python reproduce/verify_decisions.py
```

The first verifier checks 25 main rows, 150 component cells, round and diagnostic
summaries, missing-value policy, and the provenance inventory. The second
recomputes 2,427,600 round decisions from 578,000 example records and checks
4,950 aggregate quantities. Save a new report outside `results/`:

```bash
python reproduce/verify_decisions.py --output audit/decision_check.json
```

The release review also compared 1,734,000 stored raw Actor responses with the
full-system decision files and found no mismatches. This verifies released
predictions; it is not a fresh end-to-end training replication.

## Tests

The package requires Python 3.10 or newer:

```bash
python -m pip install -e "code[dev]"
python reproduce/run_tests.py
```

The reviewed Linux audit environment used Python 3.13.13, PyTorch 2.12.1,
Transformers 5.12.1, TRL 1.7.0, PEFT 0.19.1, and datasets 5.0.0. All 290
packaged tests passed there. Tests cover parsing, prompts, voting, configuration,
orchestration, and training setup; they do not launch GPU training.

## Prepare a full rerun

The released configs contain path markers and server-side artifact references.
Prepare an empty workspace:

```bash
python -m pip install pyyaml
python reproduce/prepare_workspace.py \
  --workspace /work/sepal-rerun \
  --model-root /models \
  --data-root /data/acccollab_paper
```

The helper copies `code/`, resolves all 50 configs under `configs/reproduction/`,
and changes filesystem paths only. It never starts training. Supply the five
public checkpoints and official benchmark data at the given roots, then:

```bash
cd /work/sepal-rerun
python -m pip install -e ".[dev]"
```

Run the primary configuration:

```bash
python scripts/multi_acccollab/05_pipeline.py \
  --config configs/reproduction/qwen25_3b/multi_mmlu_train.yaml
```

Use `llama3_8b`, `gemma2_2b`, `phi4_mini`, or `mistral7b_v03` for the other
backbones. `original_mmlu_train.yaml` invokes the matched single-pair ACC-Collab
pipeline; `multi_mmlu_train.yaml` invokes SEPAL. After adapters are available,
validate a transfer evaluation with:

```bash
python scripts/multi_acccollab/06_evaluate_dataset.py \
  --config configs/reproduction/qwen25_3b/multi_eval_boolq.yaml \
  --devices 0,1,2,3 \
  --dry-run
```

Remove `--dry-run` only when checkpoints, adapters, data, and GPUs are ready.
Recorded runs use up to four 80 GB NVIDIA A800 GPUs, base seed 42, role seed
offsets 0/10,000/20,000, a 4,096-token context for Phi, and an 8,192-token
context for the other backbones. Mistral uses five preference trajectories per
training question; the other backbones use one.

## Reproduction boundary

A **paper audit** needs no model weights or accelerator and checks only locked
artifacts. A **full rerun** requires the public checkpoints and official
datasets, the prepared configs, and a record of any hardware, dependency, seed,
or model-revision changes. A different checkpoint revision or split is a new
experiment, not a replacement for a locked paper cell.

All released file sizes and SHA-256 values are recorded in
`PROVENANCE_MANIFEST.json`. The manifest is updated when this documentation is
revised so the numerical audit remains reproducible.
