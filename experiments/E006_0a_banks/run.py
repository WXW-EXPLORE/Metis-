"""E006：0a 参考记忆的验证（分库 A、路由 E、多时间尺度 B、显式删除 C）。

基线为 Metis 的 GDN 形式（K=1，beta=1，带按键擦除）。每个实验只改变一个机制。
等参数：臂 (i) 为 K 库且 d_b=d/K（总状态与基线相同），臂 (ii) 为 d_b=d（仅作参考）。

用法（仓库根目录）：
    PYTHONPATH=src python experiments/E006_0a_banks/run.py --out experiments/E006_0a_banks
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from holistic.banked_memory import BankedLinearMemory

D = 64
DV = 64
SEEDS = [0, 1, 2, 3, 4]


def unit_keys_with_rho(gen, d, n, rho):
    """目标键 k_c 与 n 个干扰键，干扰键与 k_c 的余弦严格为 rho。返回 (k_c, keys)。"""
    k_c = F.normalize(torch.randn(d, generator=gen, dtype=torch.float64), dim=0)
    noise = torch.randn(n, d, generator=gen, dtype=torch.float64)
    noise = noise - (noise @ k_c)[:, None] * k_c[None, :]
    noise = F.normalize(noise, dim=-1)
    keys = rho * k_c[None, :] + float(np.sqrt(max(0.0, 1 - rho ** 2))) * noise
    return k_c, keys


def make_memory(arm, seed, alphas=None, top_k=None):
    """arm: 'K1' 基线；'i' 为 K=4、d_b=d/4；'ii' 为 K=4、d_b=d。"""
    if arm == "K1":
        mem = BankedLinearMemory(D, DV, n_banks=1, alphas=alphas or [0.99], seed=seed)
    elif arm == "i":
        mem = BankedLinearMemory(D, DV, n_banks=4, bank_key_dim=D // 4,
                                 alphas=alphas or [0.99] * 4, top_k=top_k, seed=seed)
    elif arm == "ii":
        mem = BankedLinearMemory(D, DV, n_banks=4, bank_key_dim=D,
                                 alphas=alphas or [0.99] * 4, top_k=top_k, seed=seed)
    else:
        raise ValueError(arm)
    return mem.double()


def m1_crosstalk(seed, arm, top_k, rho, n):
    """目标写入后再写 n 个干扰（与目标余弦为 rho），读出目标键的余弦。"""
    gen = torch.Generator().manual_seed(1000 + seed)
    k_c, dk = unit_keys_with_rho(gen, D, n, rho)
    v_c = torch.randn(DV, generator=gen, dtype=torch.float64)
    dv_ = torch.randn(n, DV, generator=gen, dtype=torch.float64)
    keys = torch.cat([k_c[None], dk], dim=0)[None]
    vals = torch.cat([v_c[None], dv_], dim=0)[None]
    mem = make_memory(arm, seed, top_k=top_k)
    mem.record = True
    mem.write(keys, vals)
    y = mem.read(k_c[None])[0]
    cos = F.cosine_similarity(y, v_c, dim=0).item()
    summary = mem.routing_summary()
    ent = summary["entropy"] if summary else float("nan")
    util = summary["utilization"] if summary else [1.0]
    return {"recall_cos": cos, "routing_entropy": ent, "max_bank_util": max(util)}


def m2_age(seed, alphas, top_k, age):
    """目标写入后经过 age 步无关写入，读出目标键的余弦。"""
    gen = torch.Generator().manual_seed(2000 + seed)
    k_c = F.normalize(torch.randn(D, generator=gen, dtype=torch.float64), dim=0)
    v_c = torch.randn(DV, generator=gen, dtype=torch.float64)
    filler_k = F.normalize(torch.randn(age, D, generator=gen, dtype=torch.float64), dim=-1)
    filler_v = torch.randn(age, DV, generator=gen, dtype=torch.float64)
    keys = torch.cat([k_c[None], filler_k], dim=0)[None]
    vals = torch.cat([v_c[None], filler_v], dim=0)[None]
    n_banks = len(alphas)
    if n_banks == 1:
        mem = BankedLinearMemory(D, DV, n_banks=1, alphas=alphas, seed=seed).double()
    else:
        mem = BankedLinearMemory(D, DV, n_banks=n_banks, alphas=alphas, top_k=top_k, seed=seed).double()
    mem.write(keys, vals)
    y = mem.read(k_c[None])[0]
    return {"recall_cos": F.cosine_similarity(y, v_c, dim=0).item()}


def m3_delete(seed, arm, top_k, rho, n):
    """写入目标与 n 个相关条目，显式删除目标，测其余条目的误伤与目标残留。"""
    gen = torch.Generator().manual_seed(3000 + seed)
    k_c, dk = unit_keys_with_rho(gen, D, n, rho)
    v_c = torch.randn(DV, generator=gen, dtype=torch.float64)
    dv_ = torch.randn(n, DV, generator=gen, dtype=torch.float64)
    keys = torch.cat([k_c[None], dk], dim=0)[None]
    vals = torch.cat([v_c[None], dv_], dim=0)[None]
    mem = make_memory(arm, seed, top_k=top_k)
    mem.write(keys, vals)
    before = mem.read(dk[None])[0].clone()           # (n, dv)，同一记忆对 n 个查询读出
    mem.delete(k_c[None], beta_d=1.0)
    after = mem.read(dk[None])[0]
    residual = mem.read(k_c[None])[0].norm().item()
    collateral = ((after - before).norm(dim=-1) / before.norm(dim=-1).clamp_min(1e-12)).mean().item()
    return {"collateral": collateral, "target_residual": residual}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    # M1：分库与路由（臂 i/ii，softmax 与 top-1）
    configs_m1 = [("K1", None), ("i", None), ("i", 1), ("ii", None)]
    for arm, top_k in configs_m1:
        for rho in [0.0, 0.1, 0.3, 0.6]:
            for n in [8, 32, 128]:
                for seed in SEEDS:
                    r = m1_crosstalk(seed, arm, top_k, rho, n)
                    rows.append({"exp": "M1", "arm": arm, "top_k": top_k or "softmax",
                                 "rho": rho, "n": n, "seed": seed, **r})
        print(f"M1 done arm={arm} top_k={top_k}", flush=True)

    # M2：多时间尺度（单一 α 与快慢两库）
    configs_m2 = [("single_0.9", [0.9]), ("single_0.99", [0.99]), ("single_0.999", [0.999]),
                  ("fast_slow_top1", [0.9, 0.999])]
    for name, alphas in configs_m2:
        for age in [1, 4, 16, 64, 256]:
            for seed in SEEDS:
                r = m2_age(seed, alphas, top_k=1, age=age)
                rows.append({"exp": "M2", "arm": name, "top_k": "", "rho": "", "n": age,
                             "seed": seed, **r})
    print("M2 done", flush=True)

    # M3：显式删除的误伤（K=1 与分库，全部与 top-1 分库）
    configs_m3 = [("K1", None), ("i", None), ("i", 1)]
    for arm, top_k in configs_m3:
        for rho in [0.0, 0.1, 0.3, 0.6]:
            for n in [8, 32, 128]:
                for seed in SEEDS:
                    r = m3_delete(seed, arm, top_k, rho, n)
                    rows.append({"exp": "M3", "arm": arm, "top_k": top_k or "softmax",
                                 "rho": rho, "n": n, "seed": seed, **r})
    print("M3 done", flush=True)

    fields = sorted({k for r in rows for k in r})
    with (out / "raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # 汇总：按实验与配置求均值、标准差
    groups = {}
    for r in rows:
        key = (r["exp"], r["arm"], str(r["top_k"]), str(r["rho"]), str(r["n"]))
        groups.setdefault(key, []).append(r)
    value_keys = ["recall_cos", "routing_entropy", "max_bank_util", "collateral", "target_residual"]
    summary = []
    for key, rs in groups.items():
        row = dict(zip(["exp", "arm", "top_k", "rho", "n"], key))
        for vk in value_keys:
            vals = [float(x[vk]) for x in rs if vk in x and x[vk] != ""]
            if vals:
                row[f"{vk}_mean"] = float(np.mean(vals))
                row[f"{vk}_std"] = float(np.std(vals))
        row["seeds"] = len(rs)
        summary.append(row)
    sfields = sorted({k for r in summary for k in r})
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(summary)

    config = {"D": D, "DV": DV, "seeds": SEEDS, "torch": torch.__version__,
              "baseline": "GDN form, K=1, beta=1, alpha=0.99 (M1, M3); alphas per M2 config",
              "arms": {"K1": "K=1, d_b=d", "i": "K=4, d_b=d/4 (equal total state)",
                       "ii": "K=4, d_b=d (capacity-confounded reference)"}}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"wrote {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
