# E002：训练后 Metis-4B 的写入门控 α、β 分布

脚本：`collect_gates.py`。汇总：`gate_summary.csv`。配置：`config.json`。原始数组：`/hy-tmp/metis-outputs/E002/gates_raw.npz`（未入库）。

复现（激活 `/hy-tmp/venv-metis`，在仓库根目录）：

```bash
PYTHONPATH=. python experiments/E002_gate_stats/collect_gates.py \
  --checkpoint_path /hy-tmp/metis-models/Metis-4B \
  --questions_file example/multi_turn_questions.json \
  --num_synthetic 200 --seed 0 \
  --out experiments/E002_gate_stats \
  --raw_dir /hy-tmp/metis-outputs/E002
```

## 数据与局限（先读这一节）

- 204 个会话、1017 次 commit（`exchange` 模式），覆盖 8 个 hyper memory（第 3、7、11、15、19、23、27、31 层）。
- 数据：`example/multi_turn_questions.json` 的 4 个会话 + 200 个合成记忆会话（人名、食物、城市、年龄、干扰句）。**不是论文的评测数据**（评测数据未随仓库发布）。
- 助手回复为固定文本（"Got it." 等），未经模型生成。统计的是写入路径的门控，不包含生成质量。
- 门控由 bf16 计算，sigmoid 输出有量化台阶（例如 β_gate 的最小步长约 0.001）。
- 统计时 flash-linear-attention 是否已安装，运行时未记录。若未安装，Qwen3.5 线性注意力走 torch 回退实现，隐状态会有小幅数值差异。
- 统计的是 commit 写入路径，不是推理时的读出。

## 门控的计算方式（与代码一致）

来源：`metis/dev_beta/metis_hyper_memory.py:1052` `_apply_gated_delta_rule_update`。

- $\alpha_{\text{write}}=\sum_t w_t\,\sigma(\cdot)_\alpha/\sum_t w_t$，每次 commit 一个值。
- $\beta_t=w_t\cdot 0.9\cdot\sigma(\cdot)_\beta$，逐 token。`update_ratio=0.9`。
- 写入键 $k=\mathrm{normalize}(W_k h)/\sqrt{kv\_dim}$，$kv\_dim=1024$（由 `W_k` 形状 `[1024, 2560]` 确认），所以 $\|k\|^2=1/1024$。
- 更新：$M\leftarrow\alpha(M-\sum_t\beta_t k_tk_t^\top M)+\sum_t\beta_tk_tv_t^\top$。

因此每次写入中：
- **遗忘强度**为 $1-\alpha_{\text{write}}$，作用在整个状态上；
- **擦除强度**约为 $\varepsilon=\big(\sum_t\beta_t\big)\cdot\|k\|^2=\big(n_{\text{sel}}\cdot\overline{\beta_{\text{tok}}}\big)/1024$。

## 结果

### 每层汇总（均值，来自 `gate_summary.csv`）

| 层 | $\bar\alpha_{\text{write}}$ | 遗忘 $1-\bar\alpha$ | $\sum_t\beta_t$ / 次写入 | 擦除 $\varepsilon$ | $\varepsilon$ / 遗忘 |
|---|---|---|---|---|---|
| 3 | 0.124 | 0.876 | 0.0104 | 1.0e-05 | 1.2e-05 |
| 7 | 0.149 | 0.851 | 0.0376 | 3.7e-05 | 4.3e-05 |
| 11 | 0.692 | 0.308 | 0.0202 | 2.0e-05 | 6.4e-05 |
| 15 | 0.660 | 0.340 | 0.0185 | 1.8e-05 | 5.3e-05 |
| 19 | 0.798 | 0.202 | 0.0808 | 7.9e-05 | 3.9e-04 |
| 23 | 0.694 | 0.306 | 0.1822 | 1.8e-04 | 5.8e-04 |
| 27 | 0.768 | 0.232 | 0.2584 | 2.5e-04 | 1.1e-03 |
| 31 | 0.903 | 0.097 | 0.0874 | 8.5e-05 | 8.8e-04 |

每个 token 的 $\beta_{\text{gate}}$ 均值：第 3 层 0.013，第 27 层 0.42。选中的 token 数每次写入约 15–20（`n_selected`）。

### 观察（初步，样本限于上述数据）

1. **α 随层强烈变化。** 第 3、7 层的 $\bar\alpha\approx0.12$–$0.15$，即每次写入约遗忘 85%；中后层 $\bar\alpha\approx0.66$–$0.90$。分布很宽（例如第 19 层 α_gate 的 5%–95% 分位数为 0.14–1.0），没有退化为常数 0 或 1。
2. **擦除远弱于遗忘。** 在所有层，按键擦除的强度只有遗忘的约 $10^{-5}$–$10^{-3}$。也就是说，在这些样本上，训练后的 Metis 几乎不做按键擦除，"删除"实际上主要由 $\alpha$ 的全局衰减完成。
3. **β_gate 不是零。** 个别 token 的 β_gate 可以超过 0.9（第 27 层 95% 分位数 0.97）。但这些 token 的权重只占很小一部分，平均擦除仍然很弱。

### 与 E001 的关系

- E001 是固定 $\lambda$ 的参考线性记忆，键为单位向量。Metis 的键范数只有 $1/32$，擦除项相应缩小。因此 E001 中按键擦除的误伤数值，**不能**直接套用到 Metis。
- Metis 的遗忘是输入相关的 $\alpha_t$，且每次 commit 作用一次，不是每个 token 一次。
- 读出有归一化分母（`y = qM/(q·z+1)`），绝对尺度会被部分抵消，所以 α 的数值不等于"每轮保留的记忆比例"。这一点需要单独验证（见下一步）。

## 对研究方向的含义（猜想，需验证）

- 思路三（有地址的删除）的空间比预想的小：在当前训练得到的 Metis 中，按键擦除的强度远弱于遗忘。若要让删除真正起作用，需要先确认这种弱擦除是否影响读出质量。
- 思路二（快慢双尺度）有依据：α 在层间的差异很大（0.12 与 0.90 并存），说明模型已经在不同层学到了不同的遗忘速度。但这是否构成“快慢两组”，还要看同一层内是否也有两个时间尺度。
- 思路一（分库编目）：E002 没有测路由，本次统计不涉及。

## 下一步

1. **Phase 0b**：在 Metis 的真实读出路径上，测擦除项和遗忘项各自对读出的贡献，包括归一化分母的影响。
2. **用真实评测数据重做统计**：本次是合成数据和固定回复，需要用 `eval/` 中的任务数据（需先按 `eval/data/README.md` 准备）复核。
3. **同层多尺度检验**：在同一层内，检查 α 是否呈双峰分布，以判断是否存在快慢两组。
