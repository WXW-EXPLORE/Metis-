"""E008：等状态 GDN 对照（E008a）与 N1 键表示（E008b，分块 vs 随机投影）。

E008a：RLS K1 的状态为 M(64x64)+P(64x64)=2*64^2。GDN K=2 满维度库的状态为 2*64^2，等大。
       GDN 两库 alpha 与 RLS 的 lambda 相同（0.99 与 0.999），路由 top-1（随机固定中心）。
E008b：RLS 等参数臂 (i)（K=4, d_b=16），比较 proj_mode='random'（N1 的原方案）与 'block'（坐标分块）。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from holistic.banked_memory import BankedLinearMemory  # noqa: E402

D, DV = 64, 64
SEEDS = [0, 1, 2, 3, 4]


def unit_keys_with_rho(gen, d, n, rho):
    k_c = F.normalize(torch.randn(d, generator=gen, dtype=torch.float64), dim=0)
    noise = torch.randn(n, d, generator=gen, dtype=torch.float64)
    noise = noise - (noise @ k_c)[:, None] * k_c[None, :]
    noise = F.normalize(noise, dim=-1)
    keys = rho * k_c[None, :] + float(np.sqrt(max(0.0, 1 - rho ** 2))) * noise
    return k_c, keys


def m1(seed, mem, rho, n):
    gen = torch.Generator().manual_seed(1000 + seed)
    k_c, dk = unit_keys_with_rho(gen, D, n, rho)
    v_c = torch.randn(DV, generator=gen, dtype=torch.float64)
    dv_ = torch.randn(n, DV, generator=gen, dtype=torch.float64)
    keys = torch.cat([k_c[None], dk], dim=0)[None]
    vals = torch.cat([v_c[None], dv_], dim=0)[None]
    mem.write(keys, vals)
    y = mem.read(k_c[None])[0]
    return F.cosine_similarity(y, v_c, dim=0).item()


def m2(seed, mem, age):
    gen = torch.Generator().manual_seed(2000 + seed)
    k_c = F.normalize(torch.randn(D, generator=gen, dtype=torch.float64), dim=0)
    v_c = torch.randn(DV, generator=gen, dtype=torch.float64)
    fk = F.normalize(torch.randn(age, D, generator=gen, dtype=torch.float64), dim=-1)
    fv = torch.randn(age, DV, generator=gen, dtype=torch.float64)
    keys = torch.cat([k_c[None], fk], dim=0)[None]
    vals = torch.cat([v_c[None], fv], dim=0)[None]
    mem.write(keys, vals)
    y = mem.read(k_c[None])[0]
    return F.cosine_similarity(y, v_c, dim=0).item()


def build(kind, seed):
    if kind == "rls_K1":
        return BankedLinearMemory(D, DV, n_banks=1, alphas=[0.999], update_rule="rls", seed=seed).double()
    if kind == "gdn_K2_top1_a999":
        return BankedLinearMemory(D, DV, n_banks=2, alphas=[0.999, 0.999], top_k=1, seed=seed).double()
    if kind == "gdn_K2_top1_a99":
        return BankedLinearMemory(D, DV, n_banks=2, alphas=[0.99, 0.99], top_k=1, seed=seed).double()
    if kind.startswith("rls_armi_"):
        mode = kind.split("_")[-1]
        return BankedLinearMemory(D, DV, n_banks=4, bank_key_dim=D // 4, top_k=1,
                                  alphas=[0.99] * 4, update_rule="rls", proj_mode=mode, seed=seed).double()
    raise ValueError(kind)


def main():
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    # E008a
    for kind in ["rls_K1", "gdn_K2_top1_a99", "gdn_K2_top1_a999"]:
        for rho in [0.0, 0.1, 0.3, 0.6]:
            for n in [32, 128]:
                for seed in SEEDS:
                    rows.append({"exp": "E008a_M1", "config": kind, "param": f"rho={rho}", "n": n,
                                 "seed": seed, "value": m1(seed, build(kind, seed), rho, n)})
        for age in [4, 16, 64, 256]:
            for seed in SEEDS:
                rows.append({"exp": "E008a_M2", "config": kind, "param": "", "n": age,
                             "seed": seed, "value": m2(seed, build(kind, seed), age)})
        print("E008a done", kind, flush=True)
    # E008b
    for kind in ["rls_armi_random", "rls_armi_block"]:
        for rho in [0.0, 0.1, 0.3, 0.6]:
            for n in [32, 128]:
                for seed in SEEDS:
                    rows.append({"exp": "E008b_M1", "config": kind, "param": f"rho={rho}", "n": n,
                                 "seed": seed, "value": m1(seed, build(kind, seed), rho, n)})
        print("E008b done", kind, flush=True)

    with (out / "raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    groups = {}
    for r in rows:
        groups.setdefault((r["exp"], r["config"], r["param"], str(r["n"])), []).append(r["value"])
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp", "config", "param", "n", "mean", "std", "seeds"])
        for k, v in groups.items():
            w.writerow([*k, f"{np.mean(v):.4f}", f"{np.std(v):.4f}", len(v)])
    (out / "config.json").write_text(json.dumps({
        "D": D, "DV": DV, "seeds": SEEDS,
        "E008a": "RLS K1 (2*64^2 state) vs GDN K=2 full banks top-1 (2*64^2 state), equal bytes",
        "E008b": "RLS arm i (K=4, d_b=16, top-1) with random vs block coordinate projection"}, indent=2))
    print("wrote", len(rows), flush=True)


if __name__ == "__main__":
    main()
