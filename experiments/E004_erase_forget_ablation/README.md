# E004：Phase 0b，真实读出路径上擦除与遗忘的贡献

设计：`docs/design_phase0b.md`。实现：`src/diagnostics/gate_ablation.py`（单元测试 `tests/test_gate_ablation.py`，6 项通过）。脚本：`run_ablation.py`。

复现（激活 `/hy-tmp/venv-metis`，仓库根目录）：

```bash
PYTHONPATH=.:src python experiments/E004_erase_forget_ablation/run_ablation.py \
  --checkpoint_path /hy-tmp/metis-models/Metis-4B \
  --memqa_file eval/data/memqa/locomo_evidence_sessions_tps16_20260626.jsonl \
  --max_sessions 100 --seed 0 \
  --out experiments/E004_erase_forget_ablation
```

文件：`config.json`、`per_qa.csv`（逐条 NLL）、`summary.csv`（汇总与配对差）、`state_stats_full.csv`（每层擦除与遗忘的统计）。

## 设置

- 数据：LoCoMo 的 memqa 数据，568 个证据会话中随机抽取 100 个（种子 0），共 247 个问题。
- 记忆：按模式 replay 证据会话的记忆步骤（`commit_mode=user`）。
- 度量：答案 token 的平均 NLL（teacher forcing，不生成）。答案按 `tokenizer.encode(answer, add_special_tokens=False)` 编码，不加 eos。
- 模式：`full`（原始）、`no_erase`（去掉按键擦除）、`no_forget`（α=1）、`none`（不 commit）。
- 正确性检查：`full` 与原始代码在同一会话上的最终状态逐位相等（最大差 0.0）。

## 结果

| 模式 | 平均 NLL | 配对差（相对 full） | 95% bootstrap 区间 |
|---|---|---|---|
| full | 4.3992 | — | — |
| no_erase | 4.3997 | **+0.0005** | **[−0.0029, +0.0042]** |
| no_forget | 4.8292 | **+0.4300** | **[+0.3188, +0.5571]** |
| none | 6.5132 | **+2.1140** | **[+1.8304, +2.4200]** |

## 判定（按设计文档 §6 预先约定的规则）

- **按键擦除**：`no_erase` 的区间包含 0，且点估计仅 +0.0005。按预先约定，**按键擦除对读出几乎没有可测贡献**。思路三（有地址的删除）作为“按键擦除”的增量，在这一设置下缺乏依据，需要重新论证。
- **全局遗忘**：`no_forget` 使 NLL 明显变差（+0.43），区间远离 0。遗忘对读出有实质作用。
- **记忆本身**：`none` 比 `full` 差 +2.11，说明记忆写入对回答问题很重要。

## 与 E002/E003 的关系

写入路径统计（E002/E003）显示，每层的擦除强度约为遗忘的 $10^{-5}$–$10^{-3}$。本实验在读出层面给出一致的结论：擦除的贡献不可测，遗忘有实质作用。

## 必须说明的限制

1. **bf16 掩盖的可能性（已检验，见 E005）。** 状态以 bf16 存储，擦除相对于衰减后状态的平均大小只有约 $10^{-5}$。fp32 对照（E005）显示：状态精度改变后，擦除效应的差异 −0.0045，区间 [−0.0101, +0.0004] 包含 0，因此**没有证据表明 bf16 掩盖了擦除的作用**。fp32 下擦除效应在统计上可测（−0.0041，方向为轻微有害），但量级仅约 0.004 nat/token。详见 `experiments/E005_fp32_control/README.md`。
2. **单一子集。** 只用了种子 0 的 100 个会话。按 CLAUDE.md，结论需要多个种子复核。
3. **NLL 精度。** 模型 logits 为 bf16，逐条 NLL 的分辨率有限。配对差是在 247 个问题上平均后得到的，但微小效应仍可能被量化掩盖。
4. **无生成。** 只测了 teacher-forced 的答案似然，没有测生成答案的正确率。
5. **仅 LoCoMo。** 其他数据集（MemOP 等）未纳入。

## 下一步

1. ~~fp32 状态对照~~：已完成，见 `experiments/E005_fp32_control/`。
2. 用另外 2 个种子的会话子集复核 `no_erase` 与 `no_forget` 的结论。
3. 如果 fp32 下擦除仍无作用，思路三需要改写为“删除与遗忘分离的操作”，而不是按键擦除的改进。
