"""E012：RLS 的块形式与逐 token 形式的数值一致性，以及在 Metis 维度下的速度。

用法（仓库根目录）：PYTHONPATH=src python experiments/E012_chunk_rls/run.py --out experiments/E012_chunk_rls
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from holistic.banked_memory import BankedLinearMemory


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def accuracy_block(args, out_rows):
    # 数值一致性：d=64，多个 chunk 大小，lambda 与种子变化
    for lam in [0.9, 0.99, 0.999]:
        for L in [1, 8, 32, 128]:
            for seed in range(5):
                torch.manual_seed(seed)
                N = 256
                k = torch.randn(1, N, 64, dtype=torch.float64)
                v = torch.randn(1, N, 64, dtype=torch.float64)
                beta = torch.rand(1, N, dtype=torch.float64) + 0.5
                seq = BankedLinearMemory(64, 64, n_banks=1, alphas=[lam], update_rule="rls", p0=1.0).double()
                chk = BankedLinearMemory(64, 64, n_banks=1, alphas=[lam], update_rule="rls", p0=1.0).double()
                seq.write(k, v, beta=beta)
                for s in range(0, N, L):
                    chk.write_chunk(k[:, s:s + L], v[:, s:s + L], beta=beta[:, s:s + L])
                q = F.normalize(torch.randn(1, 8, 64, dtype=torch.float64), dim=-1)
                out_rows.append({"exp": "consistency", "lam": lam, "chunk_len": L, "seed": seed,
                                 "state_rel": rel(chk.state, seq.state), "P_rel": rel(chk.P, seq.P),
                                 "z_rel": rel(chk.z, seq.z),
                                 "readout_rel": rel(chk.read(q), seq.read(q)),
                                 "cond_P": torch.linalg.cond(seq.P[0, 0]).item()})
    print("consistency done", flush=True)


def speed_block(args, out_rows):
    # 速度：Metis 维度 kv_dim=1024，单层，512 token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    torch.manual_seed(0)
    k = torch.randn(1, 512, 1024, device=dev, dtype=dt)
    v = torch.randn(1, 512, 1024, device=dev, dtype=dt)
    for mode in ["sequential", "chunk"]:
        for rep in range(2):
            m = BankedLinearMemory(1024, 1024, n_banks=1, alphas=[0.99], update_rule="rls").to(dev, dt)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            if mode == "sequential":
                m.write(k, v)
            else:
                m.write_chunk(k, v)
            if dev == "cuda":
                torch.cuda.synchronize()
            out_rows.append({"exp": "speed", "mode": mode, "rep": rep, "seconds": time.time() - t0, "device": dev})
            print(f"speed {mode} rep={rep}: {out_rows[-1]['seconds']:.3f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    accuracy_block(args, rows)
    speed_block(args, rows)
    fields = sorted({k for r in rows for k in r})
    with (out / "raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    cons = [r for r in rows if r["exp"] == "consistency"]
    worst = {k: max(r[k] for r in cons) for k in ["state_rel", "P_rel", "z_rel", "readout_rel"]}
    worst["max_cond_P"] = max(r["cond_P"] for r in cons)
    spd = [r for r in rows if r["exp"] == "speed" and r["rep"] == 1]
    summary = {"worst_consistency": worst,
               "speed_seconds": {r["mode"]: r["seconds"] for r in spd}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
