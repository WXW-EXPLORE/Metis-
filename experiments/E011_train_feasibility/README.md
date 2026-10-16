# E011：阶段 1a 训练可行性（Metis-4B，记忆参数训练，骨干冻结）

脚本：`probe.py`。结果：`c2_L512.json`、`c3_L512.json`、`c8_L512.json`、`c8_L512_gc.json`（梯度检查点，失败时无 json）。
复现：`PYTHONPATH=.:src python experiments/E011_train_feasibility/probe.py --chunks 8 --chunk_len 512 --steps 2 --out ...`

**输入为随机 token，只测可行性，不代表训练效果。** 损失值无意义。

## 设置

- 训练器的方式（`train/trainer.py:compute_loss`）：按顺序前向各个记忆 chunk（`commit_memory=True`，梯度经记忆写入回传），最后一个 chunk 计算标签损失。
- 冻结：`train.train_utils.freeze_backbone`。可训练参数 **125,925,392**（占总量 4.97B 的 2.5%），80 个张量。
- 单卡 RTX 4080 SUPER（32 GB），bf16，batch = 1，chunk 长度 512。

## 结果

| chunks | 峰值显存 | 稳定单步耗时 | 记忆参数梯度 |
|---|---|---|---|
| 2 | 15.3 GiB | 0.38 s | 64/80 非零（α 门控 16 个张量为 0） |
| 3 | 17.6 GiB | 0.58 s | 80/80 非零 |
| 8 | 29.2 GiB | 1.58 s | 80/80 非零 |
| 8 + 梯度检查点 | 失败 | — | — |

首步约 45 s，为内核编译与预热，不计入。

## 发现

1. **α 门控在只有一次写入后的查询中梯度为零（已验证，机制明确）。** 旧状态为零时，$\alpha(M-\mathcal E)$ 对 $\alpha$ 的导数为零。因此 **episode 内查询前至少需要两次写入，α 才能学到东西。** 3 个及以上 chunk 时全部 80 个张量都有梯度。这一约束直接影响训练数据的组织。
2. **显存随 chunk 数线性增长**，约 +2.3 GiB/chunk（512 token）。无梯度检查点时，单卡上限约 8–9 个 chunk。
3. **梯度检查点与记忆写入不兼容（已验证）。** 反向重算时写入的记忆状态与前向不一致，PyTorch 报 `Recomputed values ... have different metadata`。需要自定义检查点（保存每段的记忆状态快照），或限制 episode 长度。
4. **耗时**：约 0.2 s/chunk（512 token）。
5. 对照 LoCoMo：每个记忆步骤约 1900 字符（约 500 token），每个证据会话约 7 步（E003），即在 8 chunk 的显存范围内。长 episode 才需要处理检查点问题。

## 对设计的含义

- 训练阶段的 episode 结构：至少两次写入后再查询（否则 α 不学习）。
- 显存预算：单卡可直接训练 ~8 个 chunk 的 episode。更长的需要自定义检查点（另立阶段实现）。
- 训练需要记忆任务的数据；训练数据暂缺，合成数据需要与评测集分离（设计 0a/阶段 1）。
- 这些都是阶段 1a 的结论（已验证）；训练后的收敛性、α 分布的变化、以及是否优于原版 Metis 都尚未验证（阶段 1b）。
