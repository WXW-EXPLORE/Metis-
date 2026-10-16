# 设计说明：Phase 0b 真实读出路径上的擦除与遗忘贡献

状态：已由用户同意开始（"同意"）。本文件先于代码编写，实现与此一致。
关联：`docs/design_diagnostics.md`（Phase 0）、`experiments/E002_gate_stats`、`experiments/E003_locomo_gate_stats`。

## 1. 目的

E002/E003 显示，训练后 Metis 的按键擦除强度 $\varepsilon$ 远小于遗忘强度 $1-\alpha$。但这只是写入路径的统计，没有说明两者对**读出**的贡献。本阶段回答：

1. 去掉按键擦除（$\beta$ 的擦除项）后，答案的似然变化多大？
2. 去掉遗忘（令 $\alpha=1$）后，答案的似然变化多大？
3. 两者都不用记忆（不 commit），相对基线差多少？

结论用于决定思路三（有地址的删除）是否值得继续投入，以及思路二（快慢双尺度）的优先级。

## 2. 控制论语言

Metis 每次写入是一步 LPV 更新，状态 $M$（每层一个矩阵）：

$$M' = \alpha\,(M-\mathcal E) + \mathcal A,\qquad \mathcal E=\sum_t\beta_t k_t k_t^\top M,\qquad \mathcal A=\sum_t\beta_t k_t v_t^\top$$

- $\alpha$ 是**遗忘**（全局收缩，对应切换系统中的衰减模态）；
- $\mathcal E$ 是**按键擦除**（投影式修正，对应卡尔曼式的修正增益）；
- $\mathcal A$ 是**写入**。

消融只替换 $\alpha$ 或 $\mathcal E$，写入项 $\mathcal A$ 保持不变。这样每次只改一个机制（单变量归因）。

读出为 $y=qM/(q^\top z+1)$，$z$ 是键的累计状态。归一化分母属于读出路径，不在状态递推中，本阶段保持不变。

## 3. 消融模式

| 模式 | 状态更新 | 说明 |
|---|---|---|
| `full` | 原始更新 | 基线，应与原代码逐位一致 |
| `no_erase` | $M'=\alpha M+\mathcal A$，且 $z$ 同样去掉擦除 | 去掉按键擦除 |
| `no_forget` | $M'=M-\mathcal E+\mathcal A$ | 令 $\alpha=1$，去掉全局遗忘 |
| `none` | 不 commit，记忆为空 | 无记忆基线 |

$z$（键的累计状态）的更新与 $M$ 同构，因此 `no_erase` 与 `no_forget` 对 $z$ 也做相同替换。

## 4. 接口

- `src/diagnostics/gate_ablation.py`
  - `gated_delta_update(mod, h_normed, weights, local_memory, mode, stats=None)`：按 `metis/dev_beta/metis_hyper_memory.py:1052` 的公式计算，`mode` 决定是否替换 $\alpha$ 或擦除项。`stats` 可选，用于记录 $\|\alpha\mathcal E\|/\|\alpha M\|$ 等量。
  - `install_mode(model, mode)`：把所有 hyper memory 的 `_apply_gated_delta_rule_update` 替换为指定模式。
- `experiments/E004_erase_forget_ablation/run_ablation.py`：读取 memqa 数据，对每个证据会话做 4 种模式的 replay，对每个 QA 计算答案 token 的 NLL。
- 输出：`experiments/E004_erase_forget_ablation/`，包含 `config.json`、`per_qa.csv`、`summary.csv`、`README.md`。

## 5. 度量

- **主指标**：答案 token 的平均 NLL（只在答案 token 上计算，prompt 部分 label 为 -100）。
- **配对差**：$\Delta_m=\mathrm{NLL}_m-\mathrm{NLL}_{\text{full}}$，按 QA 配对，报告均值、标准误和 bootstrap 95% 区间。
- **状态诊断**：每层、每次写入的 $\|\alpha\mathcal E\|_F/\|\alpha M\|_F$（擦除相对于衰减后状态的大小）和 $(1-\alpha)$。

NLL 的选择理由：不需要生成，计算快，且对答案的微小变化敏感。生成式 QA 准确率作为后续补充，不在本阶段。

## 6. 判定规则（预先约定，避免事后解释）

- 若 $\Delta_{\text{no\_erase}}$ 的 95% 区间不含 0 且明显大于 0：按键擦除对读出有贡献，思路三有实际价值。
- 若 $\Delta_{\text{no\_erase}}$ 的区间包含 0 或接近 0：按键擦除对读出几乎无贡献，思路三的增量需要重新论证。
- $\Delta_{\text{no\_forget}}$ 与 $\Delta_{\text{none}}$ 用于判断遗忘与记忆本身的作用，作为背景参照。

## 7. 单元测试

在 `tests/test_gate_ablation.py` 中，用一个只含线性层的小模块与桩 local memory 测试：

1. `full` 模式与原始公式的输出逐位一致（随机输入，固定种子）。
2. `no_erase` 的新状态等于 $\alpha M+\mathcal A$（与直接计算比较）。
3. `no_forget` 的新状态等于 $M-\mathcal E+\mathcal A$。
4. 线性性：固定 $\alpha,\beta$ 时，状态对 $M$ 仍然线性（$\mathcal E$ 对 $M$ 线性）。
5. 数值稳定性：长序列（例如 $10^4$ 次写入）下状态范数有限。

另外在真实模型上做一次检查：`full` 模式与原始代码在同一会话上的最终状态逐位相等。

## 8. 失效模式与回滚

- **替换错误**：`full` 与原始不一致。由单元测试和真实模型检查捕获。
- **z 处理不一致**：`no_erase` 只改 $M$ 不改 $z$ 会引入偏差。设计要求两者同时替换。
- **批量近似**：代码对所选 token 一次性写入，是并行近似（`metis_hyper_memory.py:1052` 的注释）。消融沿用这一近似，不改为逐 token 顺序写入。
- **样本选择**：先用 100 个会话（随机子集，种子 0）。结论需要多个子集种子复核。
- **回滚**：消融通过替换实例方法实现，不修改仓库中的代码。进程退出即恢复。
- 出现 NaN 或状态发散：立即停止，定位原因，不调参掩盖。

## 9. 不在本阶段范围内

- 不修改 Metis 的任何模块或权重。
- 不修改评测数据、指标定义或划分。
- 不引入新的记忆机制（那是 Phase 1 的内容）。
- 不做生成式准确率。

## 10. 计算预算

- 100 个会话，每个会话 4 次 replay，加上 QA 的前向计算。
- 参考 E003：约 0.5 秒每次 commit。预计约 30–40 分钟。
- 全部 568 个会话约需 3 小时，需要你确认后再扩展。
