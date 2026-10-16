# E003：LoCoMo 上训练后 Metis-4B 的写入门控 α、β 分布

脚本：`experiments/E002_gate_stats/collect_gates.py`（`--memqa_file` 模式）。汇总：`gate_summary.csv`。配置：`config.json`。原始数组：`/hy-tmp/metis-outputs/E003/gates_raw.npz`（未入库）。

复现（激活 `/hy-tmp/venv-metis`，在仓库根目录）：

```bash
PYTHONPATH=. python experiments/E002_gate_stats/collect_gates.py \
  --checkpoint_path /hy-tmp/metis-models/Metis-4B \
  --memqa_file eval/data/memqa/locomo_evidence_sessions_tps16_20260626.jsonl \
  --seed 0 \
  --out experiments/E003_locomo_gate_stats \
  --raw_dir /hy-tmp/metis-outputs/E003
```

## 数据与局限

- 数据：LoCoMo（`locomo10.json`，10 段对话），经 `eval.data.prepare --dataset locomo` 转换为 `memqa` 格式（1527 行 QA）。
- 去重：同一（对话, 证据会话集）的记忆步骤只写入一次，共 568 个会话、2115 次 commit。每个 commit 是一段 16 轮对话的记忆文本，作为 user 消息写入（`commit_mode=user`）。
- 这是 LoCoMo 上的写入路径统计，**不是** MemOP 评测。MetisTest 与 MetisOps 未提供，无法复现完整的 MemOP 表。
- flash-linear-attention 未安装，运行时走 torch 回退实现。
- 其余局限与 E002 相同：bf16 量化；统计的是写入路径，不是读出。

## 门控的计算方式

与 E002 相同（见 [E002 README](../E002_gate_stats/README.md)）：
- $\alpha_{\text{write}}$ 是选中 token 的 α 门控按权重的平均，每次 commit 一个值。
- $\beta_t = w_t\cdot 0.9\cdot\sigma_\beta$，逐 token。
- 键范数 $\|k\|^2 = 1/kv\_dim = 1/1024$。
- 每次写入的擦除强度 $\varepsilon=\big(\sum_t\beta_t\big)/1024$，遗忘强度为 $1-\alpha_{\text{write}}$。

## 结果

### 每层汇总（均值，2115 次写入）

| 层 | $\bar\alpha_{\text{write}}$ | α 的 5%–95% | $\bar\beta_{\text{gate}}$ | 每次写入选中 token 数 | $\varepsilon$ / 遗忘 |
|---|---|---|---|---|---|
| 3 | 0.262 | 0.22–0.30 | 0.041 | 359 | 4.5e-05 |
| 7 | 0.490 | 0.38–0.56 | 0.099 | 400 | 1.6e-04 |
| 11 | 0.585 | 0.49–0.73 | 0.063 | 389 | 1.3e-04 |
| 15 | 0.417 | 0.32–0.52 | 0.131 | 400 | 1.6e-04 |
| 19 | 0.622 | 0.44–0.82 | 0.451 | 410 | 8.2e-04 |
| 23 | 0.581 | 0.28–0.91 | 0.749 | 356 | 1.2e-03 |
| 27 | 0.599 | 0.26–0.93 | 0.777 | 369 | 1.6e-03 |
| 31 | 0.714 | 0.47–0.95 | 0.159 | 330 | 4.2e-04 |

### 与 E002（合成数据）的比较

- **选中 token 数不同。** 合成数据约 16 个，LoCoMo 约 330–410 个。这是 AlphaTopP 选择（累计概率阈值 0.9）的结果：输入越长，选中的 token 越多。
- **α 的层间模式相似，但数值不同。** 合成数据的第 3 层 α 约 0.12，LoCoMo 约 0.26；第 31 层两者分别约 0.90 与 0.71。
- **擦除仍远弱于遗忘。** 擦除与遗忘之比在两个数据集上都是 $10^{-5}$–$10^{-3}$ 量级，最高在第 27 层，约 $1.6\times10^{-3}$。

## 观察（初步，仅限 LoCoMo 写入路径）

1. 按键擦除的强度远弱于遗忘，在所有层都成立。这与 E002 的结论一致，说明它不依赖合成数据。
2. 选中 token 数随输入长度显著增加，但擦除与遗忘的比例仍然很小，因为每个 token 的 β 很小。
3. 中后层（19–27）的 β_gate 较大（0.45–0.78），同时 α 的分布很宽（第 23、27 层 5% 分位数约 0.26–0.28）。这些层的遗忘在不同写入之间差异很大。

## 猜想（需要验证）

- 中后层的大 β_gate 与宽 α 分布，可能是模型用来区分"需要覆盖"与"需要保留"的写入。但目前没有读出层面的证据。
- 擦除的相对强度很小，可能是因为键范数 $1/\sqrt{kv\_dim}$ 把擦除项压低了。这意味着即使 β 大，按键擦除的作用也有限。这一点需要在 Phase 0b 中，用读出误差直接检验。

## 复跑：fast path 与 torch 回退

- 目录：`fastpath/`。fla 与 causal-conv1d==1.6.0 均已安装，`is_fast_path_available = True`，运行时没有回退警告（`/hy-tmp/venv-metis`，Triton 3.1.0）。
- 之前的 `fla_rerun/` 没有真正启用 fast path（缺少 causal-conv1d），因此它只能确认可复现，不能说明内核的影响。`fastpath/` 才是有效的对照。

| 层 | $\bar\alpha_{\text{write}}$ 回退 → fast path | $\bar\beta_{\text{gate}}$ 回退 → fast path | 选中 token 数 回退 → fast path |
|---|---|---|---|
| 3 | 0.2616 → 0.2617 | 0.0411 → 0.0411 | 359.3 → 359.2 |
| 7 | 0.4896 → 0.4896 | 0.0994 → 0.0994 | 399.6 → 399.5 |
| 11 | 0.5845 → 0.5846 | 0.0626 → 0.0626 | 389.0 → 389.1 |
| 15 | 0.4174 → 0.4175 | 0.1313 → 0.1313 | 399.9 → 400.7 |
| 19 | 0.6221 → 0.6219 | 0.4506 → 0.4505 | 409.9 → 410.0 |
| 23 | 0.5809 → 0.5811 | 0.7490 → 0.7490 | 356.0 → 356.2 |
| 27 | 0.5990 → 0.5991 | 0.7774 → 0.7774 | 369.4 → 369.3 |
| 31 | 0.7141 → 0.7140 | 0.1587 → 0.1587 | 330.1 → 330.3 |

**结论（已验证）**：内核实现对门控分布的影响可以忽略，α 的差异在 0.001 以内，β 基本相同。后续统计可以使用 torch 回退或 fast path，结果一致。

## 下一步

1. Phase 0b：在 Metis 的真实读出路径上，测擦除项与遗忘项对读出的贡献，包括归一化分母的作用。
2. 其他数据集（NextMem、ATM、MemDaily、MetisTest、MetisOps）如需纳入统计，需要先在 `eval/data/raw/` 下准备好源文件。
3. 其他数据集（NextMem、ATM、MemDaily、MetisTest、MetisOps）当前均未在 `eval/data/raw/` 中。若要扩展统计，需要先准备这些数据。
