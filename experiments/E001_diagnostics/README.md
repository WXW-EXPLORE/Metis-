# E001：Phase 0 合成诊断（参考线性记忆）

设计：`docs/design_diagnostics.md`。代码：`src/diagnostics/`。测试：`tests/test_synthetic_memory.py`（13 项通过）。

复现：

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m diagnostics.run_diagnostics \
    --config experiments/E001_diagnostics/config.json \
    --out experiments/E001_diagnostics
```

文件：

- `config.json`：全部超参数与种子（5 个种子，$d=d_v=64$，$A=512$）。
- `raw.csv`：逐条原始结果。
- `decay_tau.csv`：有效时间常数。
- `crosstalk_summary.csv`：串扰，按 $(\lambda, N, \rho)$ 汇总。
- `deletion_summary.csv`：删除，按 $(\text{方法}, \lambda, N, \rho)$ 汇总。

## 结果

### 1. 衰减（已验证）

| $\lambda$ | $\tau_{\text{eff}}$ 测量 | $-1/\ln\lambda$ 设计值 |
|---|---|---|
| 0.9 | 9.491 | 9.491 |
| 0.99 | 99.499 | 99.499 |
| 0.999 | 999.50 | 999.50 |

这是实现正确性的检查，不是新发现：振幅严格为 $(1-\lambda)\lambda^{\text{age}}$，拟合结果与理论一致。

### 2. 串扰（初步观察）

归一化串扰 $\text{crosstalk\_over\_fresh}$（均值，5 个种子）：

| $\lambda$ | $N$ | $\rho=0.1$ | $\rho=0.3$ | $\rho=0.9$ |
|---|---|---|---|---|
| 0.999 | 8 | 0.26 | 0.78 | 2.34 |
| 0.999 | 32 | 0.55 | 1.64 | 4.92 |
| 0.999 | 128 | 1.02 | 3.05 | 9.15 |
| 0.999 | 256 | 1.32 | 3.98 | 11.9 |
| 0.99 | 8 | 0.25 | 0.75 | 2.26 |
| 0.99 | 32 | 0.48 | 1.44 | 4.32 |
| 0.99 | 128 | 0.65 | 1.95 | 5.86 |
| 0.99 | 256 | 0.66 | 1.99 | 5.96 |

观察：

- 串扰与 $\rho$ 近似成正比。
- 对小 $N$，串扰随 $\sqrt N$ 增长（$\lambda=0.999$ 时 $N$ 从 8 到 32，增长约 2.1 倍）。这与设计说明的预期一致。
- 对 $\lambda=0.99$，串扰在 $N\approx128$ 后饱和，对应有效记忆时域 $1/(1-\lambda)=100$。$\lambda=0.999$ 的时域为 1000，在本网格内还未饱和。
- 这一形式与 $\rho\sqrt{\sum_j\lambda^{2a_j}}$ 相符，偏差约 10%，未细查原因。

**衰减与串扰的比较（同一目标年龄 $A=512$）：**

- $\lambda=0.99$：目标只剩 $\lambda^{512}\approx0.6\%$，衰减占主导。
- $\lambda=0.999$：目标保留约 60%，但 $\rho\ge0.1$、$N\ge32$ 时，串扰的范数已经超过目标（$\rho=0.1$、$N=256$ 时，串扰 0.0106，目标 0.0048）。此时串扰占主导。

**猜想（未验证）**：Metis 的实际 $\lambda$ 决定主导项。需要先从 Metis 配置中读取真实的 $\lambda$ 分布，才能判断应先做思路一还是思路二。

### 3. 删除误伤（已验证，形式上）

按键擦除（GDN 式，$\beta=1$）的误伤（均值，5 个种子）：

| $N$ | $\rho=0$ | $\rho=0.1$ | $\rho=0.3$ | $\rho=0.6$ | $\rho=0.9$ |
|---|---|---|---|---|---|
| 8 | 0.000 | 0.025 | 0.226 | 0.717 | 0.967 |
| 256 | 0.000 | 0.070 | 0.561 | 0.963 | 0.999 |

- 目标残留为 $10^{-14}$ 量级，即目标被完全删除。
- $\rho=0$ 时误伤为零，与理论一致。
- 误伤随 $\rho$ 单调上升，并随 $N$ 增加。

全局缩放基线 $M\leftarrow\gamma M$：目标残留与误伤恒等于 $\gamma$ 和 $1-\gamma$，即 $\text{collateral}=1-\text{residual}$，无选择性。

**解读（观察）**：按键擦除的误伤来自键之间的相关性。这说明思路三的增量不在于“按键擦除”本身，而在于 §5 所述的操作分离（删除、遗忘、修改分开），与设计说明一致。

## 局限

- 这是参考线性记忆，不是 Metis 模型。结论只说明机制的形式，不说明 Metis 的性能。
- 只用随机高斯值，未考虑真实语义数据中的值相关性。
- 键由构造得到，$\rho$ 是设定值，不是从真实 embedding 测量的。
- 5 个种子，只报告均值与标准差，未做显著性检验。

## Metis 实际机制（代码核对，已验证）

来源：`metis/dev_beta/metis_hyper_memory.py:988`（`GatedDeltaRuleMixin`）与 `metis/dev_beta/metis_local_memory.py:137`（`NormalizedDeltaNetMetisLocalMemory`）。

- **遗忘率不是固定的 $\lambda$。** 写入时 $\alpha_t=\sigma(w_\alpha^\top x_t+b_\alpha)$ 由线性层产生，且与输入相关。初始化时 $b_\alpha$ 对应 $\alpha=1$，即初始无遗忘。训练后的分布需要从 checkpoint 中读取。
- **$\beta_t$ 同样是输入相关的门控**，初始化为 $\beta=1$。
- **更新规则**为 $M_t=\alpha_t(I-\beta_t k_tk_t^\top)M_{t-1}+\beta_t k_tv_t^\top$，即与 E001 中按键擦除的形式相同。
- **批量近似**：代码对所选 token 集合使用 $\sum_t\beta_tk_tk_t^\top$ 一次性写入，是并行近似，不是严格的顺序递推。诊断若要与 Metis 对齐，需要用同样的近似。
- **读出有归一化分母**：$y=qM/(q^\top z+1)$，其中 $z$ 是键的累计状态。这一项在读出路径上，不在状态递推中，符合 CLAUDE.md 的约束，但它是非线性的，诊断中的线性读出没有包含它。

**对 E001 结论的影响**：E001 用固定 $\lambda$ 的参考线性记忆，因此“$\lambda=0.99$ 与 $0.999$”的对比只说明机制形式，不能直接套用到 Metis。判断思路一与思路二的先后，需要 Metis 训练后的 $\alpha_t$、$\beta_t$ 分布。

## 下一步

1. 下载 Metis 的 checkpoint（你说明可自行下载），统计训练后 $\alpha_t$、$\beta_t$ 的分布，据此决定 Phase 1 的顺序。
2. 用 Metis 实际的键分布（而非构造的 $\rho$）重新测量串扰与误伤。
3. 诊断若要更接近 Metis，需加入归一化读出与批量写入近似，这属于 Phase 0b，单独编号。
