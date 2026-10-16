"""E005：汇总 fp32 对照在三个会话子集（种子 0、1、2）上的擦除效应。

输入：
    experiments/E005_fp32_control/per_qa.csv        （种子 0）
    experiments/E005_fp32_control/seed1/per_qa.csv  （种子 1）
    experiments/E005_fp32_control/seed2/per_qa.csv  （种子 2）
输出：aggregate_seeds.csv
    每个种子的 no_erase - full（逐 QA 配对均值与 bootstrap 区间）；
    三个子集合并后按唯一 QA 去重的估计。
注意：不同子集的会话可能重叠，合并估计不是独立样本，只作参考。

用法（仓库根目录）：
    python experiments/E005_fp32_control/aggregate_seeds.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/E005_fp32_control"
SOURCES = {
    "seed0": BASE / "per_qa.csv",
    "seed1": BASE / "seed1/per_qa.csv",
    "seed2": BASE / "seed2/per_qa.csv",
}
OUT = BASE / "aggregate_seeds.csv"


def load(path):
    table = defaultdict(dict)
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[row["instance_id"]][row["mode"]] = float(row["nll"])
    return table


def paired(table, a, b):
    """返回 table 中同时具有 a、b 且有限的 QA 的 (a - b) 列表。"""
    vals = []
    for v in table.values():
        if a in v and b in v and np.isfinite(v[a]) and np.isfinite(v[b]):
            vals.append(v[a] - v[b])
    return np.asarray(vals)


def bootstrap(values, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=values.size, replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def row(name, values):
    lo, hi = bootstrap(values)
    return {"subset": name, "n": int(values.size), "mean": float(values.mean()),
            "se": float(values.std(ddof=1) / np.sqrt(values.size)),
            "ci95_lo": lo, "ci95_hi": hi}


def main():
    rows = []
    merged = {}
    for name, path in SOURCES.items():
        table = load(path)
        rows.append(row(f"{name} no_erase-full", paired(table, "no_erase", "full")))
        for qa, modes in table.items():
            merged.setdefault(qa, modes)
    rows.append(row("pooled_unique_qa no_erase-full", paired(merged, "no_erase", "full")))

    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['subset']}: n={r['n']} mean={r['mean']:+.5f} se={r['se']:.5f} "
              f"CI95=[{r['ci95_lo']:+.5f}, {r['ci95_hi']:+.5f}]")


if __name__ == "__main__":
    main()
