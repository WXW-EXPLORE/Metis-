"""E026 探索性诊断（不改变 G2c 判定）：G2c-b 的下降是否来自测量问题。

问题：主实验中，删除后的自检索用的是原始运行的键值。被重放的 chunk（k > j）在删除后的键会改变（闭环，N10），
因此 "删除后 − 删除前" 的差可能混入测量假象。

本诊断把它拆开：
  D_lt       ：k < j（不受影响）。键相同，只改变状态。这是删除对早期条目的纯影响。
  D_gt_stale ：k > j，用原始运行的键值（与主实验相同的口径）。
  D_gt_live  ：k > j，用重放时实际写入的键值（与删除后的状态一致）。
  key_shift  ：k > j 的键在重放前后的平均差（确认闭环效应是否存在）。
输出标注为 exploratory。

用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E026_G2c_composition/diag_remeasure.py --seed 0 --out experiments/E026_G2c_composition/diag_remeasure
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import holistic.banked_metis as B


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


G = _load("e26run", "experiments/E026_G2c_composition/run.py")


def chunk_sr(log, states, commit, n_layers, K):
    """单个 chunk（commit 号）的自检索余弦均值，使用给定的日志与状态。"""
    vals = []
    for l in range(n_layers):
        for b in range(K):
            _, kp, v, mask, _lam = log[(commit * n_layers + l) * K + b]
            sel = mask > 0
            if int(sel.sum()) == 0:
                continue
            M, _P, z = states[l][b]
            kq = kp[sel].double(); vq = v[sel].double()
            out = (kq @ M[0].double()) / (kq @ z[0].double() + 1.0)
            vals.append(F.cosine_similarity(out, vq, dim=-1))
    return float(torch.cat(vals).mean()) if vals else float("nan")


def key_shift(log_a, commit_a, log_b, commit_b, n_layers, K):
    """两个日志中同一 chunk 的键差（同层同库，逐 token 的 max 绝对差）；形状不同则返回 nan。"""
    diffs = []
    for l in range(n_layers):
        for b in range(K):
            ka = log_a[(commit_a * n_layers + l) * K + b][1]
            kb = log_b[(commit_b * n_layers + l) * K + b][1]
            if ka.shape != kb.shape:
                return float("nan")
            diffs.append(float((ka - kb).abs().max()))
    return float(np.mean(diffs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=G.EPS_N)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model, tok, dev = G.load_model()
    G.install_arm(model, dev, "K4_CL_cons", args.seed, "abs")
    n_layers = sum(1 for b in model.model.metis_blocks if b.local_memory is not None)
    K = G.K4
    rows = []
    for ep in G.test_episodes(args.n):
        chunks = ep["memory_chunks"]; N = len(chunks)
        if N < 2:
            continue
        with torch.no_grad():
            model.reset()
            B.CAPTURE["on"], B.CAPTURE["log"] = True, []
            snaps = {}
            G.run_chunks(model, tok, dev, chunks, 0, snapshots=snaps)
            log_full = B.CAPTURE["log"]
            full = G.get_states(model, dev)
            for j in range(N):
                G.set_states(model, snaps[j])
                B.CAPTURE["log"] = []
                G.run_chunks(model, tok, dev, chunks, start=j + 1)
                log_replay = B.CAPTURE["log"]
                replay = G.get_states(model, dev)
                B.CAPTURE["log"] = []
                lt, gs, gl, ks = [], [], [], []
                for k in range(N):
                    if k == j:
                        continue
                    before = chunk_sr(log_full, full, k, n_layers, K)
                    if k < j:
                        lt.append(chunk_sr(log_full, replay, k, n_layers, K) - before)
                    else:
                        rc = k - j - 1
                        gs.append(chunk_sr(log_full, replay, k, n_layers, K) - before)
                        gl.append(chunk_sr(log_replay, replay, rc, n_layers, K) - before)
                        ks.append(key_shift(log_full, k, log_replay, rc, n_layers, K))
                rows.append({"episode_id": ep["episode_id"], "N": N, "j": j,
                             "D_lt": float(np.nanmean(lt)) if lt else float("nan"),
                             "D_gt_stale": float(np.nanmean(gs)) if gs else float("nan"),
                             "D_gt_live": float(np.nanmean(gl)) if gl else float("nan"),
                             "key_shift": float(np.nanmean(ks)) if ks else float("nan")})
                print(json.dumps(rows[-1]), flush=True)
            B.CAPTURE["on"], B.CAPTURE["log"] = False, None
            model.reset()

    per_ep = {}
    for r in rows:
        per_ep.setdefault(r["episode_id"], []).append(r)
    summary = {"exploratory": True, "seed": args.seed, "n_episodes": len(per_ep), "n_deletions": len(rows)}
    rng = np.random.default_rng(0)
    for key in ["D_lt", "D_gt_stale", "D_gt_live", "key_shift"]:
        ep_means = np.array([np.nanmean([x[key] for x in v]) for v in per_ep.values()])
        ep_means = ep_means[np.isfinite(ep_means)]
        boots = np.array([rng.choice(ep_means, size=ep_means.size, replace=True).mean() for _ in range(5000)])
        summary[key] = {"mean": float(ep_means.mean()), "ci95": [float(np.quantile(boots, 0.025)),
                                                                  float(np.quantile(boots, 0.975))]}
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
