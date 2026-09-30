# Multi-role ACC-Collab (SFT-10K + majority)

This pipeline trains three independent Actor–Critic pairs. Actor iteration zero is a
role-specific SFT adapter trained from correct self-generated answers on 10,000 sampled
MMLU `auxiliary_train` questions. Critic iteration zero remains the base model. Every role
then executes the unmodified paper-original ACC-Collab order on the full 1,531-example
MMLU validation split:

1. Critic guided-trajectory preference data (K=10, Eq. 4/Eq. 5)
2. Critic DPO from the base model
3. Actor guided-trajectory preference data with the trained Critic
4. Actor DPO continued from its role SFT adapter
5. Five-round, single-trial evaluation

The final stage reads only the three final-round Actor answers. It uses a two-or-three vote
majority and a deterministic Direct-role fallback when no majority exists. No Judge is
loaded or called.

The formal runner uses phase-aware GPU scheduling without changing the ACC-Collab
hyperparameters. Critic/Actor preference data are generated role-by-role with four GPU
shards, matching the original four-GPU ACC-Collab layout. The three single-GPU Critic/Actor
DPO jobs run concurrently. This keeps all four GPUs useful during the dominant generation
stages while preserving each role's independent model, seed, prompts, and training order.

```bash
conda run -n society-rl --no-capture-output python \
  scripts/multi_acccollab/05_pipeline.py \
  --config configs/multi_acccollab/llama3_8b_instruct_mmlu_sft10k.yaml
```

All large outputs remain below
`output/multi_acccollab/llama3_8b_instruct_mmlu_sft10k/`. The generated per-role YAMLs
and manifest under `resolved_role_configs/` record the exact SFT initialization, role
prompts, device, seed, and original ACC-Collab hyperparameters.
