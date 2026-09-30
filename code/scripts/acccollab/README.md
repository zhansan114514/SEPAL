# 原版 ACC-Collab 独立复现

本目录实现论文 **Estornell et al., 2025, “ACC-Collab: An Actor-Critic
Approach to Multi-Agent LLM Collaboration”** 的两智能体原版流程。实现与仓库中现有的
`paired` 多 Actor / 多 Critic 实验隔离：

```text
src/acccollab/       可复用算法与状态管理
scripts/acccollab/   6 个编号阶段脚本与流水线
configs/acccollab/   原版、Smoke、ACC-Collab+ 配置
```

`src/acccollab/` 不依赖 `src.paired`，因此原版实验不会隐式继承多 Actor、投票、Judge、
Actor SFT 或现有改进版 Critic 的行为。

## 1. 忠实复现的协议边界

### 论文明确给出的设置

- **1 个 Actor + 1 个 Critic**，两者从同一个基础语言模型出发。
- 每次交替严格按 **Critic → Actor** 的顺序训练。
- **ACC-Collab** 进行 1 次 Critic→Actor 交替；**ACC-Collab+** 进行 2 次。
- Actor 与 Critic 都使用偏好数据直接进行 DPO；本流程不额外加入 Actor SFT。
- 训练轨迹和测试协作都固定为五轮，即 `t=0,1,2,3,4`。
- Eq. 4 使用一步 Monte Carlo partial-trajectory rollout 估计候选响应的后续正确率。
- Eq. 5 以阈值 `epsilon` 构造偏好对；实现为论文算法中的有序 `if/elif`：当两个条件
  同时成立时，优先选择“guided-positive 优于 natural”的分支，每个样本每轮最多产生
  一个偏好对。
- Appendix A 指定 LoRA size/rank 为 **256**。
- Appendix A 指定 DPO 的 chosen-completion NLL regularization weight 为 **1**。
- 评价指标是五轮协作后 **Actor 最终回答的正确率**。
- Appendix A 报告训练使用单张 NVIDIA H800，并使用 vLLM 做推理。

### 本实现对评价协议的强约束

- Headline 指标只读取 `round=4` 的**单个 Actor 回答**：
  `round_4_single_actor_accuracy`。
- 不进行 majority vote。
- 不调用 Judge，也没有解析失败时的 Judge fallback。
- Critic 回答只作为下一轮 Actor 的反馈，不能直接成为最终答案。
- 每条评估记录都写入并校验：

  ```json
  {
    "decision_rule": "single_final_actor_answer",
    "uses_majority_vote": false,
    "uses_judge_fallback": false
  }
  ```

## 2. Eq. 4 与 Eq. 5 的实现

对每个自然轨迹的 `t=1..4`，分别生成：

```text
natural
guided_positive   （向正确答案引导）
guided_negative   （向错误答案引导）
```

### Critic 偏好数据

对每个候选 Critic feedback，从该 partial trajectory 采样 `K` 个下一步 Actor revision，
以这些 Actor revision 的 gold-answer accuracy 均值作为 Eq. 4 reward。自然 Critic
feedback 始终推进主轨迹，不因 Eq. 5 最终选择了哪个候选而改变。

### Actor 偏好数据

对每个候选 Actor response，采样自然 Critic feedback，再让 Actor 做一次 revision；以
revision 的 gold-answer accuracy 均值作为 Eq. 4 reward。推进自然主轨迹所用的 Critic
feedback 会**独立重新生成**，不会复用 Monte Carlo rollout 中的 Critic 样本。

### Eq. 5

令：

```text
delta_positive = reward(guided_positive) - reward(natural)
delta_negative = reward(natural) - reward(guided_negative)
```

选择规则为：

```python
if delta_positive >= epsilon:
    chosen, rejected = guided_positive, natural
elif delta_negative >= epsilon:
    chosen, rejected = natural, guided_negative
else:
    drop_example()
```

Actor/Critic DPO 的两个 completion 始终共享自然轨迹对应的同一个 prompt。

## 3. 论文参数与公开实现参数的区分

论文没有公开所有超参数。为避免把复现选择误写成论文事实，配置来源区分如下。

### 论文明确值

| 参数 | 值 |
| --- | --- |
| deliberation rounds | 5 |
| LoRA rank | 256 |
| DPO chosen NLL weight | 1.0 |
| ACC-Collab alternations | 1 |
| ACC-Collab+ alternations | 2 |

### 公开代码或本复现的固定选择

| 参数 | 默认值 | 来源/说明 |
| --- | ---: | --- |
| generation temperature | 0.7 | 公开运行脚本 |
| top-p | 0.9 | 公开 vLLM wrapper 的 `SamplingParams`；论文未报告 |
| Eq. 4 rollouts | 10 | 公开数据生成逻辑对应的实现选择；Smoke 为 2 |
| Eq. 5 epsilon | 0.6 | 公开 `DPO.py` 数据阈值 |
| LoRA alpha | 512 | 公开实现采用 `2 * rank` |
| learning rate | `1.41e-5` | 公开 `scripts/run_dpo.sh` |
| epochs | 3 | 公开 `scripts/run_dpo.sh` |
| per-device DPO batch | 2 | 公开 `scripts/run_dpo.sh` |
| gradient accumulation | 2 | 公开 `scripts/run_dpo.sh` |
| weight decay | 0.1 | 公开 `scripts/run_dpo.sh` |
| preference questions | 6000 | 公开 `scripts/run_dpo.sh` |
| DPO beta / token budgets / warmup | 配置中显式给出 | 论文未报告，属于复现选择 |

> **已知冲突：**论文 Appendix A 明确写 LoRA rank 256，但当前公开运行脚本的默认
> `LORA_R=128`。本“忠实论文”配置以论文为准，强制 `r=256`；配置校验会拒绝其他值。

另有两处论文与公开实现不一致，本实现同样以论文公式/算法为准：

1. Actor 候选的 Eq. 4 reward 按“Actor candidate → natural Critic → Actor revision”做一步
   continuation rollout；公开代码直接检查 Actor candidate 本身是否正确。
2. 只在 Algorithm 1 的 `t=1..4` 构造偏好对；公开代码的数据路径还包含 round 0。

重复 prompt 的 Monte Carlo rollout 使用逐请求独立 seed；Critic feedback 与 Actor
revision 使用互不相交的 seed stream。若 Eq. 5 达到阈值但 chosen/rejected 文本相同，则
记录为 `identical_chosen_rejected` 并丢弃，避免向 DPO 注入零信号 pair。

## 4. 配置

```text
configs/acccollab/llama3_8b_instruct_mmlu_original.yaml
    ACC-Collab，1 次交替，6000 个训练问题，完整 14042 个 MMLU test 问题，5 trials。

configs/acccollab/llama3_8b_instruct_mmlu_plus.yaml
    ACC-Collab+，2 次交替，其余协议与原版配置一致，使用独立输出目录。

configs/acccollab/llama3_8b_instruct_mmlu_original_smoke.yaml
    4 个训练问题、8 个测试问题、1 trial；仍保持五轮、rank 256、NLL weight 1，且
    Eq. 4 rollouts=2。Smoke 只检查流程连通性，不能与论文准确率作统计比较。

configs/acccollab/llama3_8b_instruct_mmlu_validation1531.yaml
    本机正式实验配置：使用 MMLU validation 全量 1531 条收集偏好数据，随后训练一次
    Critic→Actor 交替。它是用户指定的实验设计，不是论文的 6000-question 设置。

configs/acccollab/llama3_8b_instruct_mmlu_validation1531_eval_{mmlu,sciq,boolq}.yaml
    复用上述训练 run 的认证 final registry，分别评价 MMLU test 14042、SciQ test 1000
    和 BoolQ validation 3270。BoolQ 的公开 test 无标签，因此不能称为 BoolQ test 准确率。
```

本机离线运行时必须使用 `data.mmlu_load_mode: all`。`train` 会映射到 MMLU 的
`auxiliary_train`；完整测试集大小固定校验为 14042。

## 5. 运行

脚本会通过 `_utils.py` 为子进程设置本机所需的 Hugging Face offline 环境变量。仍应从
仓库根目录、通过 `society-rl` 环境执行。

### Smoke

```bash
conda run -n society-rl --no-capture-output \
  python scripts/acccollab/06_pipeline.py \
  --config configs/acccollab/llama3_8b_instruct_mmlu_original_smoke.yaml
```

### ACC-Collab

```bash
conda run -n society-rl --no-capture-output \
  python scripts/acccollab/06_pipeline.py \
  --config configs/acccollab/llama3_8b_instruct_mmlu_original.yaml
```

### ACC-Collab+

```bash
conda run -n society-rl --no-capture-output \
  python scripts/acccollab/06_pipeline.py \
  --config configs/acccollab/llama3_8b_instruct_mmlu_plus.yaml
```

可用 `--devices 0,1,2,3` 对数据生成和评价做确定性分片；DPO 仍是一角色一任务，使用
`--training-devices` 列表中的第一张卡。论文报告的是单卡执行，因此默认配置均为 `[0]`。

### 只运行部分阶段

流水线顺序为：

```text
iteration-01-critic-data
iteration-01-critic-train
iteration-01-actor-data
iteration-01-actor-train
[iteration-02-critic-data ... iteration-02-actor-train]
evaluate
```

示例：

```bash
# 对每个配置迭代运行 Critic 数据阶段
python scripts/acccollab/06_pipeline.py --config <yaml> --only critic-data

# 只运行第二次 Actor 训练
python scripts/acccollab/06_pipeline.py --config <plus-yaml> \
  --only iteration-02-actor-train

# 重新进入评价阶段；合法 shard checkpoint 仍可恢复
python scripts/acccollab/06_pipeline.py --config <yaml> --only evaluate --force
```

## 6. 断点与产物

每个生成 batch 使用原子 JSONL checkpoint；每个 stage 的 `_SUCCESS` 包含语义指纹、
产物大小和 SHA-256。流水线只有在 pipeline marker 与真实 stage/adapter 产物都匹配当前
配置和 policy identity 时才会跳过。

主要输出：

```text
<output_dir>/
  data/iteration_XX/{critic,actor}_dpo/
    pairs.jsonl
    trajectories.jsonl
    generation_audit.jsonl
    metrics.json
    _SUCCESS
  adapters/iteration_XX/{critic,actor}_adapter/
  registry/iteration_XX.json
  registry/final.json
  eval/
    records.jsonl
    trial_00_metrics.json
    ...
    aggregate_metrics.json
    paper_comparison.json
    metrics.json
    _SUCCESS
  markers/*.json
```

第二次交替不会重新从 base 初始化：Critic(2) 从 Critic(1) LoRA 继续，Actor(2) 从
Actor(1) LoRA 继续。Critic 和 Actor 之间不互相继承 LoRA。

## 7. 与论文结果比较

论文 Table 1 的 Llama-3 五轮结果为：

| 数据集 | ACC-Collab | ACC-Collab+ |
| --- | ---: | ---: |
| BoolQ | `0.887 ± 0.005` | `0.894 ± 0.003` |
| MMLU | `0.644 ± 0.010` | `0.683 ± 0.012` |
| SciQ | `0.952 ± 0.000` | `0.948 ± 0.003` |

这里的 `±` 是论文报告的 **95% confidence interval half-width**，不是标准差。
`eval/paper_comparison.json` 会输出：

- 本次 round-4 单 Actor 平均准确率；
- 相对论文均值的 absolute / relative delta；
- 是否高于论文均值；
- 是否落在论文报告的 95% CI 内。

只有同数据集训练、论文训练规模、完整测试 split 和多 trial 的正式配置才适合做同口径
比较。用 MMLU 训练的 adapter 在 SciQ/BoolQ 上属于**跨域评价**；即使输出对应 Table 1
参考值，也不能宣称复现/超过论文的同任务独立训练结果。Smoke、不同基础模型、不同测试
抽样或不同 trial 数同样应标记为流程验证或消融。

完整逐项审计、论文/发布代码冲突和本轮修复记录见 [`AUDIT.md`](AUDIT.md)。
