# PROGRESS.md（会话交接）

最后更新：2026-10-25（A 解码 b1 初步采用；B pilot 完成，未提交）

## 一、主线目标（锚定）
最终目标：得出一个在相同参数与状态预算下，**优于原版 Metis 记忆（GDN）** 的记忆机制设计。衡量对象是下游任务（LoCoMo 问答的答案 NLL），不是中间指标。
锚点文件：`docs/mainline_anchor.md`。局部步骤必须说明服务最终目标的哪一环，并在结果为负时说明会改变哪个决策。

## 二、当前阶段
**阶段：候选模型的下游对比（测试阶段，不下结论）。**
- 已完成第一组直接对比：候选 RLS 记忆（K1）vs 原版 GDN 记忆（从头训练），同数据、同预算（300 步、学习率 1e-4、3 个种子）。
- 结果：K1 的测试 NLL 高于 GDN，三种子平均 D = K1 − GDN = **+0.283，区间 [0.172, 0.398]**，按登记规则判为"GDN 显著更好"（仅在本预算、本设置下）。
- 结果记录：`docs/design_downstream_gdn_vs_rls.md` 第 7 节。

## 三、已完成（主要）
- **理论与合成数据**：R0/R0b（慢速规律子空间，C4 通过）、R0c（线性预测规律可辨识）、R1/R1b/R1b′（预测+量化创新的率失真收益，N=20000 收敛）、第 3 步（版本一致性，L=50 通过）、第 4 步（观测预测器对观测干预错误跟随，K-C 通过；do(s) 修正版无效应）。
- **真实键**：第 5 / 5b / 5c 步（token 级局部可预测；chunk 级跨对话泛化仅层 7 通过，层 3 未通过）。
- **工程组件**：RLS 块形式与特征值上界（E012、N8、N9）；读出归一化 `1+|q·z|`（修正梯度尖峰，diag_mech、R1 判据）；快照重放删除（G2、G2b、G2c-a 精确）；版本记录（L=50）。
- **训练脚本**：`experiments/E025_G1e_train/train.py` 新增 `--arm GDN`（原版 GDN 对照；K1/K4 行为不变）。
- **测试**：约 82 项单元测试通过（此前版本）；新增 `tests/test_readout_norm.py`、`tests/test_g2c_replay.py`、`tests/test_timescale.py`，均通过。

## 四、关键决策及理由（含否决）
- **读出分母改为 `1+|q·z|`**：原版分母在 1–7% 读出中为负，导致梯度尖峰。
- **删除采用快照重放，否决降秩**：降秩在闭环中不精确（N10、N14）。
- **否决"分库组织收益"作为主张**：真实数据上未稳定（N12、N15、N16）。
- **否决 D1 正交分块作为解耦手段**：串扰降为零但保留下降（N17）；同时确认问题设定有误，库间耦合应视为经验产生的结构而非需消除的对象。
- **预测规律的线性情形**：只能由时间交叉统计量辨识（R0c），边际协方差不足（R0 的局限）。
- **版本问题**：版本漂移代价约 12.7 倍（V-a 通过）；L=500 不足（N20），L=50 通过。
- **参数选择**：凡需要选参数的，一律用预试验（与正式实验种子不重叠，只用拟合/调参划分），选择规则预先写定（如 R1 的 k=3、d=8、R=0.5；step4 的阈值 11.8%）。
- **测试集重复使用**：E028、第 5、5b、5c 步均使用同一测试划分，结论需注明。

## 五、卡点 / 未决
1. **完整候选模型未拼接**：集成设计 `docs/design_integrated.md` 中，慢层与 KV 载体的对接方式未最终确定：(a) 存残差键；(b) 存残差值。用户已同意先测试，我倾向 (b)。
2. **下游对比的结果不利于候选**：RLS 单库在同预算下比 GDN 差（D=+0.283）。原因未知：可能是超参未调（λ=0.9 等）、训练预算不足、或 RLS 结构本身。**未检验**，不得据此下结论。
3. **R1 中分工收益仍低于 oracle**（比值 0.74–0.94 对 0.49–0.67），差距待查。
4. **K-B 无正对照**，不能说明它能识别不变性失败。
5. **训练预算**：每个 GDN/RLS 运行约 20–40 分钟；全部对比需要数小时 GPU。
6. **Git 状态**：当前分支 `feature/control-theory-memory`，本地 `main` 已快进到 `09a050b`。之后的所有新增（E025–E028、R 系列、新文档、测试）**尚未提交**。推送到远程被权限拒绝过（需要用户放开）。

## 六、下一步（具体动作，待用户确认）
1. **[已完成] 接口选 (b)**；集成 v1（`src/holistic/integrated.py`，`tests/test_integrated.py` 7 项通过，全量 98 项通过）。
1a. **[已完成，初步] 解码缺口（E030，`docs/design_E030_decode.md` §8）**：采用 (b1) 读出端补偿。载体级、年龄 0、tune 划分下，所有层 b1 − b0 ≤ 0.028（判据 0.05）；无补偿不可用（N23）；(b2) 只好 0.00–0.03 却需 2× 状态。**未做**：模型级检验（1b）与遗忘后的解码。
1b. **模型级检验（需 GPU）**：默认开关下，集成路径与 `install_rls` 的模型输出逐位一致（设计 §5 第 5 项，未做）。
2. **[已完成，pilot] 诊断差距（E029，`experiments/E029_gap_diagnosis/README.md`，N24）**：λ=0.99 与 lr=3e-4 均无效（d 区间含 0）；GDN 基线同向（d=−0.345）。新观察：RLS 初始 NLL 9.43 对 GDN 7.28，提示结构不匹配（猜想，未检验）。**E031（已完成，不训练）**：无记忆地板 6.05；GDN 初始 7.28、RLS 初始 9.43，两臂的记忆分支都有害（比值 29–38 倍）。S-R 缩放检验无效（推测 `mem_norm` 抵消）。详见 `docs/design_E031_init_mismatch.md` §6。**待你决定**：mem_norm_init 对两臂同时降低（需登记），或先核实 mem_norm 位置。
3. [已完成] 提交改动，分支 `feature/control-theory-memory`（已快进到 `main` 的 `f580739`）。新增提交 `89ecc7b`（文档）、`99996bf`（代码与测试）、`b74dae7`（实验记录）。**未推送**：远程 feature 分支仍停在 `09a050b`，推送需你确认。测试：`PYTHONPATH=src python -m unittest discover -s tests`，98 项通过（约 100 秒）。

## 七、文件索引
- 主线锚点：`docs/mainline_anchor.md`；主线记录：`docs/MASTER_PLAN.md`（§0 为提议章节，§8 修改记录）。
- 负结果：`docs/negative_results.md`（N1–N22）。
- 集成设计：`docs/design_integrated.md`；下游对比：`docs/design_downstream_gdn_vs_rls.md`。
- 理论：`docs/theory_innovations.md`、`docs/theory_regularity.md`、`docs/theory_coupling.md`（已废弃）。
- 合成数据实验：`experiments/R0_regularity/`、`R0c_predictive/`、`R1_division/`（含 pilot 与 run_*.py）、`step4_knowledge/`。
- 真实键实验：`experiments/E028_H11_timescale/`（录制文件 `/hy-tmp/e028_keys.pt`，约 27 GB）、`experiments/step5_real/`。
- 训练实验：`experiments/E025_G1e_train/`（`runs_abs/` 为 K1/K4；`runs_gdn/` 为 GDN 基线）。
- 设计文档：`docs/design_*.md`（R0、R0b、R0c、R1、step4、step5、step5b、step5c、downstream、integrated 等）。
- 模型与数据：`/hy-tmp/metis-models/Metis-4B`；`eval/data/memqa/`（LoCoMo，不入库）。

**2026-10-26 新发现（E031b / E032，需你决定）**：`reinit_memory` 把 mem_norm 权重设为 1.0，而 Qwen3_5RMSNorm 的有效缩放是 1+weight，因此 E013–E031 全部使用缩放 2.0。缩放 1.0 时初始有害大幅减小（GDN −0.31 超出地板，RLS +0.15），两臂初始差距由 2.15 缩小到 0.46。训练后的差距是否随之改变尚未检验。E032：集成路径与 install_rls 损失逐位一致（✅ 设计 §5 第 5 项）。详见 `docs/design_E031_init_mismatch.md` §7–§8。
