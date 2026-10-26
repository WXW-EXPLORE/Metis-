# E032：模型级检验（不训练）

设计依据：`docs/design_integrated.md` §5 第 5 项；`docs/design_E031_init_mismatch.md` §7。脚本：`run_check.py`，结果：`result.json`，日志：`run.log`。

## 结果

| 检验 | 结果 | 判定 |
|---|---|---|
| 集成路径（install_integrated）与 install_rls 的输出，默认开关，abs 分母 | 损失 9.1875 与 9.1875，逐位一致 | ✅ 通过 |
| 同上，orig 分母 | 损失 9.3125 与 9.3125，逐位一致 | ✅ 通过 |
| mem_norm 的类 | `Qwen3_5RMSNorm`，前向 `x̂·(1+weight)` | ✅ 核实 |
| mem_norm 对输入乘 0.1 后的输出差 | 最大绝对差 0.031，输出最大值 8.56（bf16 舍入量级） | ✅ 缩放被归一化抵消 |

## 说明

- 比较的是一个 pilot 片段（第 5 个对话的第一个 QA 行）的答案 NLL（标量），不是逐 logit 比较。逐位一致的强度低于 logit 级比较，但足以确认默认开关下两条路径的数值一致。
- 发现：`reinit_memory` 把 `mem_norm.weight` 设为 1.0，使有效缩放为 2.0（见 E031 §7）。本检验没有修改它。
- 早期两次运行因外部 timeout 被终止，没有结果文件；本目录的结果来自第三次运行。
