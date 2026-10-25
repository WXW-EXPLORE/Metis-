"""E026 G2c-b'（第二次登记，见 docs/design_G2c.md §10）：修正后的测量口径下的不误伤检验。

  D_lt       ：k < j 的 chunk，键不变，删除前后的自检索余弦之差（完整状态 → 重放状态，同一键值）。
  D_gt_live  ：k > j 的 chunk，删除前（完整键值，完整状态）→ 删除后（重放中实际写入的键值，重放状态）。
  D_gt_stale ：k > j 的 chunk，使用原始键值（仅作参照，不参与判定，等同 G2c-b 的旧口径）。

判定（K4_CL_cons）：D_lt 与 D_gt_live 的 episode 级 95% bootstrap 区间下界均 > −0.01 → G2c-b' 通过。
K1 作为参照计算，不参与判定。

用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E026_G2c_composition/run_g2cb_v2.py --arm K4_CL_cons --seed 0 --out experiments/E026_G2c_composition/g2cb_v2/K4_CL_cons_s0
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import holistic.banked_metis as B


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


G = _load("e26run", "experiments/E026_G2c_composition/run.py")
D = _load("e26diag", "experiments/E026_G2c_composition/diag_remeasure.py")


def bci(x, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(5000)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["K1_full", "K4_CL_cons"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=G.EPS_N)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model, tok, dev = G.load_model()
    G.install_arm(model, dev, args.arm, args.seed, "abs")
    n_layers = sum(1 for b in model.model.metis_blocks if b.local_memory is not None)
    K = 1 if args.arm == "K1_full" else G.K4

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
                lt, gl, gs, ks = [], [], [], []
                for k in range(N):
                    if k == j:
                        continue
                    before = D.chunk_sr(log_full, full, k, n_layers, K)
                    if k < j:
                        lt.append(D.chunk_sr(log_full, replay, k, n_layers, K) - before)
                    else:
                        rc = k - j - 1
                        gl.append(D.chunk_sr(log_replay, replay, rc, n_layers, K) - before)
                        gs.append(D.chunk_sr(log_full, replay, k, n_layers, K) - before)
                        ks.append(D.key_shift(log_full, k, log_replay, rc, n_layers, K))
                rows.append({"episode_id": ep["episode_id"], "N": N, "j": j,
                             "D_lt": float(np.nanmean(lt)) if lt else float("nan"),
                             "D_gt_live": float(np.nanmean(gl)) if gl else float("nan"),
                             "D_gt_stale": float(np.nanmean(gs)) if gs else float("nan"),
                             "key_shift": float(np.nanmean(ks)) if ks else float("nan")})
                print(json.dumps(rows[-1]), flush=True)
            B.CAPTURE["on"], B.CAPTURE["log"] = False, None
            model.reset()

    per_ep = {}
    for r in rows:
        per_ep.setdefault(r["episode_id"], []).append(r)
    summary = {"arm": args.arm, "seed": args.seed, "n_episodes": len(per_ep), "n_deletions": len(rows)}
    for key in ["D_lt", "D_gt_live", "D_gt_stale", "key_shift"]:
        ep_means = [float(np.nanmean([x[key] for x in v])) for v in per_ep.values()]
        m, lo, hi = bci(ep_means)
        summary[key] = {"mean": m, "ci95": [lo, hi]}
    if args.arm == "K4_CL_cons":
        summary["G2c_b_prime_passed"] = bool(summary["D_lt"]["ci95"][0] > -0.01
                                             and summary["D_gt_live"]["ci95"][0] > -0.01)
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
