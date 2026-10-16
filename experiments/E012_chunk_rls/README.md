# E012：RLS 块形式的数值一致性与速度

脚本：`run.py`。结果：`raw.csv`、`summary.json`。
复现（仓库根目录）：`PYTHONPATH=src python experiments/E012_chunk_rls/run.py --out experiments/E012_chunk_rls`

## 背景

设计 1b 原先假设 chunk 级更新是近似（E012 用于测误差）。推导后发现：RLS 在信息形式下是线性的（$\Lambda_t=\lambda\Lambda_{t-1}+w_t\kappa_t\kappa_t^\top$，$b_t=\lambda b_{t-1}+w_t\kappa_tv_t^\top$，$M=\Lambda^{-1}b$），因此一个 chunk 的闭式更新与逐 token 更新**精确等价**，只需一次求逆（`BankedLinearMemory.write_chunk`）。E012 的作用是验证这一点。

## 结果（已验证）

**一致性**（float64，$d=64$，256 步，$\lambda\in\{0.9,0.99,0.999\}$，chunk 长度 $L\in\{1,8,32,128\}$，5 个种子）：

| 量 | 最坏相对误差 |
|---|---|
| 状态 $M$ | $1.2\times10^{-12}$ |
| 协方差 $P$ | $6.9\times10^{-13}$ |
| 键质量 $z$ | $1.0\times10^{-15}$ |
| 读出 | $1.2\times10^{-12}$ |

最大条件数 $\kappa(P)$ 为 $2.4\times10^4$。

**一致性（Metis 维度，fp32，$d=1024$，$L=512$，单层）**：状态 $2.5\times10^{-5}$，$P$ $8.2\times10^{-6}$，$z$ $2.7\times10^{-7}$。这是 fp32 舍入量级，不是方法误差。

**速度**（$d=1024$，单层，512 token，GPU bf16 输入，状态 fp32 计算）：

| 方式 | 时间 |
|---|---|
| 逐 token（`write`） | 0.35 s |
| 块形式（`write_chunk`） | 0.006 s |

约 60 倍。

## 注意

1. 速度测试用的是 bf16 输入，但块形式的计算在 fp32 下进行（PyTorch 不支持 bf16 求逆）。bf16 下的数值一致性尚未单独测量。
2. 状态若以 bf16 存储，会有 E004 中讨论过的舍入问题（相对精度约 $4\times10^{-3}$）。RLS 的协方差对舍入更敏感，**1b 中 RLS 的 $P$ 与 $M$ 应以 fp32 存储**，这一点需要在 1b 设计中写明。
