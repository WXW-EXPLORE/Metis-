"""E005：比较 bf16 与 fp32 状态精度下的配对差（逐 QA 配对）。

输入：E004（bf16）与 E005（fp32）的 per_qa.csv。
输出：compare_bf16_fp32.csv，包含
    precision_effect       full(fp32) - full(bf16)
    erase_effect_fp32      no_erase(fp32) - full(fp32)
    erase_effect_bf16      no_erase(bf16) - full(bf16)
    erase_effect_diff      erase_effect_fp32 - erase_effect_bf16
每项给出均值、配对标准误与 bootstrap 95% 区间。

用法（仓库根目录）：
    python experiments/E005_fp32_control/compare.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BF16 = ROOT / "experiments/E004_erase_forget_ablation/per_qa.csv"
FP32 = ROOT / "experiments/E005_fp32_control/per_qa.csv"
OUT = ROOT / "experiments/E005_fp32_control/compare_bf16_fp32.csv"


def load(path):
    table = defaultdict(dict)
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[row["instance_id"]][row["mode"]] = float(row["nll"])
    return table


def bootstrap(values, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    means = np.array([rng.choice(values, size=values.size, replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def stat(name, values):
    values = np.asarray(values)
    lo, hi = bootstrap(values)
    se = values.std(ddof=1) / np.sqrt(values.size)
    return {"contrast": name, "n": int(values.size), "mean": float(values.mean()),
            "se": float(se), "ci95_lo": lo, "ci95_hi": hi}


def main():
    b = load(BF16)
    f = load(FP32)
    common = [k for k in f if k in b and all(m in f[k] and m in b[k] for m in ("full", "no_erase"))]
    rows = []
    rows.append(stat("precision_effect (full fp32 - full bf16)",
                     [f[k]["full"] - b[k]["full"] for k in common]))
    rows.append(stat("erase_effect_fp32 (no_erase - full, fp32)",
                     [f[k]["no_erase"] - f[k]["full"] for k in common]))
    rows.append(stat("erase_effect_bf16 (no_erase - full, bf16)",
                     [b[k]["no_erase"] - b[k]["full"] for k in common]))
    rows.append(stat("erase_effect_diff (fp32 - bf16)",
                     [(f[k]["no_erase"] - f[k]["full"]) - (b[k]["no_erase"] - b[k]["full"]) for k in common]))

    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['contrast']}: n={r['n']} mean={r['mean']:+.5f} "
              f"se={r['se']:.5f} CI95=[{r['ci95_lo']:+.5f}, {r['ci95_hi']:+.5f}]")


if __name__ == "__main__":
    main()
