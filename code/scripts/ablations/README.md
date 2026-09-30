# Multi-A/C component ablations

The paper-reported ablation matrix covers five backbones: Llama-3-8B,
Qwen2.5-3B, Gemma-2-2B, Phi-4-mini, and Mistral-7B-v0.3. It contains six reported variants:

- `sft_only`: the three role-specific SFT Actors answer once, followed by the
  same majority-with-Direct-fallback rule as the main method.
- `sft_base_critic`: each SFT Actor interacts with an untrained base-model
  Critic for all five rounds.
- `sft_trained_critic`: each SFT Actor interacts with its trained Critic, with no Actor DPO.
- `no_sft_full`: role SFT is omitted; role prefixes, Critic and Actor DPO, private revision and voting are retained.
- `Full-R0`: the final Actor policies answer before receiving any
  inference-time Critic message.
- `Full-R4`: the complete system after four Critic-conditioned revisions.

Every reported variant is evaluated on BoolQ, MMLU, BBH, SciQ, and ARC. The
round-wise analysis also records each role, majority coverage, fixed-Direct
fallback, oracle-any-role accuracy, and pairwise role agreement. Existing
Direct, Debate, SoM, matched ACC-Collab, and full Multi-A/C results are reused
rather than regenerated.

All six completed variants appear in the reviewed manuscript. The trained-Critic and No-SFT aggregates are also supplied in `supplementary/results/extended_ablations/`, and their per-example decisions are in `supplementary/results/decisions/`.

## Reproducibility constraints

- Formal coverage and trial counts are authenticated from the completed role
  configs.
- Adapter weights, source configs, base-model metadata, sample identities,
  parser version, batch size, and seed layout enter durable fingerprints.
- Existing logical shards are retained when hardware availability changes, so
  sample partitions and stochastic batch boundaries do not silently change.
- All stages are resumable. A completed artifact is reused only when its
  fingerprint and hashes still match.
- No Judge is used.

## Entrypoints

1. `01_materialize.py`: authenticate source artifacts and create a server-local
   manifest.
2. `04_analyze_existing.py`: run the zero-GPU round and role analysis used for
   Full-R0 through Full-R4.
3. `03_run_policy_matrix.py`: evaluate the SFT-only, SFT+Base-C and SFT+Trained-C policies.
4. `05_run_no_sft_full.py`: run the No-SFT pipeline.
5. `07_run_model.py`: execute the available ablation runners for one model.
6. `08_run_a800.py`: historical four-A800 launcher for the initial
   Llama/Qwen2.5/Gemma matrix.
7. `09_watch_h100.py`: optional H100 watcher; it was not used in the reported
   A800 execution.

For numerical reproduction, the resolved manifests and per-role YAML files in
`results/ablations/` take precedence over these orchestration helpers.
