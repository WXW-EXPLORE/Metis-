# 设计说明：阶段 1b，原版 Metis 与新结构的合成数据对照训练

状态：**待确认**。按 CLAUDE.md，编码前须获确认。
前提：用户接受单卡 8 个 chunk（512 token）的 episode 长度上限，条件是能对比出效果（2026-10-15）。
依据：`docs/design_holistic.md` §13，`docs/design_0a.md` §13，`experiments/E011_train_feasibility/README.md`。

## 1. 目的

在相同的合成记忆数据、相同的训练步数下，比较：
- 基线：原版 Metis 的记忆结构（GDN 形式，`NormalizedDeltaNetMetisLocalMemory`）；
- 新结构：RLS 记忆（`update_rule="rls"`，单库，见 0a §13），接入 Metis 的 local_memory 接口。

回答的问题：新结构的记忆参数能否训练，并在留出的合成记忆任务上取得不差或更好的结果。这是效果的对比，不是对 Metis 论文结论的复现。

## 2. 数据（与评测集完全分离）

- 合成生成器（阶段 0c）：写入事实、干扰句、更新、删除、提问。
- 训练、验证使用不同的种子与实体表，互不重叠；记录数据哈希。
- 输出格式按 README 的 `messages` 与 `query_turn_id`，并满足 `train/dataset.py` 的任务分配（metadata.type/style），以便复用训练器的数据读取。
- 不使用 LoCoMo、NextMem、MetisTest、MetisOps 等评测数据。LoCoMo 只用于 E002–E005 的诊断，不得用于训练。
- 每个 episode 的查询前至少两次写入（E011 结论）。

## 3. 关键工程问题：RLS 的逐 token 循环太慢

E011 的测量（1024 维，单层，512 token）：RLS 0.35 s，GDN 0.6 s（含预热）。8 chunk × 8 层的 episode 前向加反向需要数十秒，远超骨干的 0.2 s/chunk。

**采用的做法：块形式（信息形式的闭式，精确）。** RLS 在信息形式下是线性的，$\Lambda=\lambda^L\Lambda_0+\sum_t\lambda^{L-1-t}w_t\kappa_t\kappa_t^\top$，$b=\lambda^Lb_0+\sum_t\lambda^{L-1-t}w_t\kappa_tv_t^\top$，$M=\Lambda^{-1}b$。因此块形式与逐 token 形式**精确等价**，不是近似（E012 验证：float64 下相对误差约 $10^{-12}$）。实现为 `BankedLinearMemory.write_chunk`，速度比逐 token 快约 60 倍（1024 维、512 token、单层）。

**精度要求**：求逆在 fp32 下进行；RLS 的 $P$ 与 $M$ 在 1b 中以 fp32 存储（bf16 的舍入会影响协方差，见 E012 README）。

## 4. 接入 Metis

- 替换每个 full-attention 层的 `local_memory`（维度 kv_dim=1024）。接口保持：`read(query)`、`write(new_state, key_state)`、`state`、`key_state`、`reset()`。
- 新增的状态：每层协方差 $P\in\mathbb R^{1024\times1024}$（bf16 约 2 MiB/层，8 层约 16 MiB），显存可接受。
- 写入的键与值沿用 Metis 的 hyper memory（W_k、W_v、token 选择），只把状态递推换成 RLS 的 chunk 版本。
- 门控：RLS 的增益由 $P$ 解析给出，不再需要 `gated_delta_alpha/beta` 的学习参数（设计 0a §12）。这意味着新结构的可训练参数少于基线，需要在报告中说明。
- 训练范围：两个模型都只训练记忆相关参数（`freeze_backbone` 的同一范围），记录各自的参数量。

## 5. 对照的公平性

- **相同数据、相同步数、相同学习率调度、相同随机种子**（3 个种子）。
- **基线从头训练**：原版 Metis 的预训练记忆参数不能直接用于新结构，因此两者都从初始化开始训练。（原版的预训练参数可作为额外的参考，但不计入公平对照。）
- 报告：可训练参数量、单步耗时、峰值显存，以及验证集上的结果。

## 6. 指标

- 主指标：留出合成记忆任务上，答案 token 的平均 NLL（与 E004 相同的度量方式）。
- 辅助：生成式准确率（如果计算预算允许，按固定长度贪婪解码）。
- 监控：α 门控分布（基线）、RLS 的有效极点 $\lambda/(\lambda+\kappa^\top P\kappa)$ 分布（新结构）、协方差迹与条件数、NaN 与梯度范数。
- 统计：3 个种子，报告均值与标准差；差异用配对的 bootstrap 区间判断。

## 7. 判定规则（预先约定）

- 新结构的验证 NLL 明显低于基线（配对区间不含 0 且为负）：支持新结构。
- 差异区间包含 0：不能判断，报告为"无差异"。
- 新结构明显更差：记录为负结果，分析原因（尤其是 RLS 数值稳定性，如协方差的条件数）。

## 8. 计算预算（估计）

- 每个 episode：8 chunk，约 2 s（骨干）加记忆；块形式下记忆部分的速度已由 E012 测得（0.006 s/层/512 token）。
- 每个模型 3 个种子，每个种子约 1000 步，估计每次运行 1–2 小时。两个模型合计约 6–12 小时。
- 显存：8 chunk 约 29 GiB（E011），接近上限。若 RLS 版本显存更高，需要减少 chunk 数或批量。

## 9. 失效模式与回滚

- 块形式与逐 token 形式不一致 → E012 失败；当前通过（相对误差 $10^{-12}$）。
- α 梯度消失（episode 不足两次写入）→ 数据生成器保证每个 episode 满足条件。
- NaN 或协方差发散 → 立即停止，检查 `clamp_count` 与条件数，不调学习率掩盖。
- 显存超限 → 减少 chunk 数，记录为约束。
- 回滚：新结构通过配置开关启用，原版为默认，代码不改动骨干。

## 10. 执行顺序

1. E012（已完成）：块形式与逐 token RLS 的一致性与速度。通过。
2. 合成数据生成器与数据哈希（与评测集分离）。
3. Metis 接入：新 local_memory 类，单元测试与回归（新结构关闭时与原版逐位一致）。
4. 原版与新结构各训练 3 个种子。
5. 留出评测与统计，写入 E013。

## 11. 需要你确认

1. 合成训练数据是否可以（与评测集完全分离）？
2. 采用块形式（精确，已由 E012 验证）作为 RLS 在 Metis 上的实现，并以 fp32 存储 RLS 状态。
3. 基线从头训练（而非沿用原版预训练参数）是否接受？
4. 计算预算约 6–12 小时是否可以？
