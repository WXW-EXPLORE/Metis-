"""E027：分库比较（只比较路由中心，不重新运行模型）。

问题：不同的分库（种子、原型学习的读出版本、随机中心）是否真的是不同的划分？

方法：对两组中心 A、B（各 K=4，单位向量），枚举 K! 种库编号排列，取使匹配后余弦最大的排列，
报告匹配后的平均余弦与最小余弦。接近 1 表示两组中心是同一划分（仅编号不同）。
每层分别计算，再对 8 个记忆层取均值。

比较对象：
  CL_abs(s)   ：abs 读出版本的竞争学习原型，种子 s（experiments/E025_G1e_train/abs/prototypes.pt）
  CL_orig(s)  ：orig 读出版本的竞争学习原型，种子 s（experiments/E025_G1e_train/prototypes.pt）
  RND_abs(s)  ：abs 版本中的随机中心，种子 s（同一文件，C_rand）

用法（仓库根目录）：PYTHONPATH=.:src python experiments/E027_partition_compare/compare.py --out experiments/E027_partition_compare
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ABS = ROOT / "experiments/E025_G1e_train/abs/prototypes.pt"
ORIG = ROOT / "experiments/E025_G1e_train/prototypes.pt"
SEEDS = [0, 1, 2]


def matched(A, B):
    """A, B: (K, d) 单位向量。返回最优排列下的逐库余弦（数组）。"""
    K = A.shape[0]
    S = (A @ B.T).numpy()
    best, best_val = None, -np.inf
    for perm in itertools.permutations(range(K)):
        val = sum(S[i, perm[i]] for i in range(K))
        if val > best_val:
            best, best_val = [S[i, perm[i]] for i in range(K)], val
    return np.array(best)


def compare(pa, pb, a_key, b_key, a_seed, b_seed, blocks):
    per_layer = []
    for li in range(len(blocks)):
        A = pa[a_key][a_seed][li]; B = pb[b_key][b_seed][li]
        m = matched(A, B)
        per_layer.append({"mean": float(m.mean()), "min": float(m.min())})
    return {"mean_over_layers": float(np.mean([x["mean"] for x in per_layer])),
            "min_over_layers": float(np.min([x["min"] for x in per_layer])),
            "per_layer": per_layer}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pa = torch.load(ABS, weights_only=False)
    po = torch.load(ORIG, weights_only=False)
    blocks = pa["blocks"]
    # 统一为 dict 形式，便于比较
    P = {"abs": {"CL": pa["C_cl"], "RND": pa["C_rand"]},
         "orig": {"CL": po["C_cl"], "RND": po["C_rand"]}}

    comparisons = {}
    # 1. 同一种子，CL 的 abs 与 orig 版本
    for s in SEEDS:
        comparisons[f"CL_abs_s{s} vs CL_orig_s{s}"] = compare(
            {"x": P["abs"]["CL"]}, {"y": P["orig"]["CL"]}, "x", "y", s, s, blocks)
    # 2. abs 版本中，不同种子的 CL 原型（独立性检查）
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        comparisons[f"CL_abs_s{a} vs CL_abs_s{b}"] = compare(
            {"x": P["abs"]["CL"]}, {"y": P["abs"]["CL"]}, "x", "y", a, b, blocks)
    # 3. 随机中心：不同种子之间
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        comparisons[f"RND_abs_s{a} vs RND_abs_s{b}"] = compare(
            {"x": P["abs"]["RND"]}, {"y": P["abs"]["RND"]}, "x", "y", a, b, blocks)
    # 4. CL 与随机中心（同一种子）
    for s in SEEDS:
        comparisons[f"CL_abs_s{s} vs RND_abs_s{s}"] = compare(
            {"x": P["abs"]["CL"]}, {"y": P["abs"]["RND"]}, "x", "y", s, s, blocks)

    result = {"blocks": blocks, "note": "匹配余弦 ≈ 1 表示同一划分（仅库编号不同）；越低表示划分越不同。",
              "comparisons": {k: {kk: vv for kk, vv in v.items() if kk != "per_layer"} for k, v in comparisons.items()}}
    (out / "partition_compare.json").write_text(json.dumps(result, indent=1))
    print(f"{'comparison':40s} {'mean_cos':>9s} {'min_cos':>9s}")
    for k, v in result["comparisons"].items():
        print(f"{k:40s} {v['mean_over_layers']:9.4f} {v['min_over_layers']:9.4f}")


if __name__ == "__main__":
    main()
