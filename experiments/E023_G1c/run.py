"""E023（门控 G1c）：数据自适应的路由原型（竞争学习 + 内稳态），等总状态。设计：docs/design_G1c.md（预先登记）。

臂：K1_full、K1_pca256、K4_random、K4_CL（β=0）、K4_CL_cons（β=1，主臂）、K4_balanced_free（内容无关的等数量分配）。
拟合集 seed 9000 前 10 条；测试集 seed 9500 的 20 条（新）。负载按种子单独报告。

用法（仓库根目录）：PYTHONPATH=.:src python experiments/E023_G1c/run.py --out experiments/E023_G1c
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri
import holistic.rls_metis as R
from holistic.synthetic_memory_data import make_split
from train.train_utils import freeze_backbone
_s = importlib.util.spec_from_file_location("e18", ROOT / "experiments/E018_G1_equal_state/run.py")
E18 = importlib.util.module_from_spec(_s); _s.loader.exec_module(E18)

K, KD, LAM = 4, 256, 0.9
ETA, RHO = 0.05, 0.01


def train_prototypes(keys, beta, seed):
    """竞争学习（β 为内稳态强度）。keys: (N, 1024) 单位向量，按顺序单次遍历。"""
    g = torch.Generator().manual_seed(seed)
    C = F.normalize(torch.randn(K, keys.shape[1], generator=g), dim=-1)
    u = torch.full((K,), 1.0 / K)
    for k in keys:
        s = C @ k - beta * (u - 1.0 / K)
        b = int(s.argmax())
        C[b] = F.normalize(C[b] + ETA * (k - C[b]), dim=0)
        onehot = torch.zeros(K); onehot[b] = 1.0
        u = (1 - RHO) * u + RHO * onehot
    return C, u


def route_scores(keys, C, u, beta):
    return (keys @ C.T - beta * (u - 1.0 / K)).argmax(-1)


def simulate(test_seqs, Q, route_per_episode):
    """route_per_episode: 列表，每个元素为该 episode 全部 token（按 commit 顺序拼接）的库索引。"""
    per_ep = []
    for seq, bidx in zip(test_seqs, route_per_episode):
        dv = seq[0][1].shape[-1]
        kd = Q.shape[1]
        Ms = [torch.zeros(1, kd, dv) for _ in range(K)]
        Ps = [torch.eye(kd).expand(1, kd, kd).clone() for _ in range(K)]
        zs = [torch.zeros(1, kd, 1) for _ in range(K)]
        items, off = [], 0
        for k, v, w in seq:
            T = k.shape[0]
            b_idx = bidx[off:off + T]; off += T
            kp = F.normalize(k @ Q, dim=-1)                          # 单位键（与 E018/E019 相同的约定）
            for b in range(K):
                mask = (b_idx == b).float() * w
                Ms[b], Ps[b], zs[b] = E18.rls_commit_update(Ms[b], Ps[b], zs[b], kp[None], v[None], mask[None], LAM)
            items.append((kp, v, w, b_idx))
        cos = []
        for kp, v, w, b_idx in items:
            for t in range(kp.shape[0]):
                if w[t] == 0:
                    continue
                b = int(b_idx[t]); q = kp[t]
                num = (q[None] @ Ms[b][0]).squeeze(0)
                den = (q[None] @ zs[b][0]).squeeze() + 1.0
                cos.append(F.cosine_similarity(num / den, v[t], dim=0).item())
        per_ep.append(float(np.mean(cos)))
    return per_ep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load); model.eval(); freeze_backbone(model)
    R.install_rls(model, lam=LAM)
    fit = make_split("val", 10, seed=9000); test = make_split("val", 20, seed=9500)
    layers = [3, 7]
    fit_d = E18.record(model, tok, dev, fit, layers); test_d = E18.record(model, tok, dev, test, layers)
    R.uninstall_rls(model)

    result = {"layers": {}}
    for layer in layers:
        seqs_f, seqs_t = fit_d[layer], test_d[layer]
        Xf = torch.cat([F.normalize(k, dim=-1) for seq in seqs_f for (k, _, _) in seq], 0)
        _, _, Vh = torch.linalg.svd(Xf, full_matrices=False); Q = Vh[:KD].T.contiguous()
        Xt_list = [F.normalize(k, dim=-1) for seq in seqs_t for (k, _, _) in seq]
        lens = [sum(k.shape[0] for (k, _, _) in seq) for seq in seqs_t]
        Xt = torch.cat(Xt_list, 0)
        per_arm, loads = {}, {}

        def ep_split(bidx_all):
            out_l, off = [], 0
            for L in lens:
                out_l.append(bidx_all[off:off + L]); off += L
            return out_l

        # K1_full / K1_pca256
        per_arm["K1_full"] = simulate(seqs_t, torch.eye(1024), [torch.zeros(L, dtype=torch.long) for L in lens])
        per_arm["K1_pca256"] = simulate(seqs_t, Q, [torch.zeros(L, dtype=torch.long) for L in lens])
        # 随机固定中心（3 种子）
        rnd_runs = []
        for s in range(args.seeds):
            g = torch.Generator().manual_seed(1000 + s)
            Cr = F.normalize(torch.randn(K, KD, generator=g) @ Q.T, dim=-1)
            bidx = (Xt @ Cr.T).argmax(-1)
            rnd_runs.append(simulate(seqs_t, Q, ep_split(bidx)))
            loads.setdefault("K4_random", []).append(torch.bincount(bidx, minlength=K).float().div(bidx.numel()).tolist())
        per_arm["K4_random"] = np.mean(np.array(rnd_runs), axis=0).tolist()
        # 竞争学习（β=0）与内稳态（β=1）
        for name, beta in [("K4_CL", 0.0), ("K4_CL_cons", 1.0)]:
            runs = []
            for s in range(args.seeds):
                C, u = train_prototypes(Xf, beta, seed=2000 + s)
                # 测试时：原型冻结，使用拟合结束时的使用率
                bidx = route_scores(Xt, C, u, beta)
                runs.append(simulate(seqs_t, Q, ep_split(bidx)))
                loads.setdefault(name, []).append(torch.bincount(bidx, minlength=K).float().div(bidx.numel()).tolist())
            per_arm[name] = np.mean(np.array(runs), axis=0).tolist()
        # 内容无关的等数量分配（每 episode 严格均衡，3 种子）
        bal_runs = []
        for s in range(args.seeds):
            g = torch.Generator().manual_seed(3000 + s)
            route_eps, all_b = [], []
            for L in lens:
                base = torch.arange(L) % K
                perm = base[torch.randperm(L, generator=g)]
                route_eps.append(perm); all_b.append(perm)
            bal_runs.append(simulate(seqs_t, Q, route_eps))
            loads.setdefault("K4_balanced_free", []).append(torch.bincount(torch.cat(all_b), minlength=K).float().div(sum(lens)).tolist())
        per_arm["K4_balanced_free"] = np.mean(np.array(bal_runs), axis=0).tolist()

        def paired(a, b):
            d = np.array(per_arm[a]) - np.array(per_arm[b]); rng = np.random.default_rng(0)
            means = np.array([rng.choice(d, size=d.size, replace=True).mean() for _ in range(5000)])
            return {"mean_diff": float(d.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}

        maxshare = {n: [float(max(v)) for v in runs] for n, runs in loads.items()}
        result["layers"][f"layer{layer}"] = {
            "means": {k: float(np.mean(v)) for k, v in per_arm.items()},
            "per_seed_max_share": maxshare,
            "median_max_share_CL_cons": float(np.median(maxshare["K4_CL_cons"])),
            "K4_CL_cons_minus_K1_full": paired("K4_CL_cons", "K1_full"),
            "K4_CL_cons_minus_balanced_free": paired("K4_CL_cons", "K4_balanced_free"),
            "K4_CL_cons_minus_random": paired("K4_CL_cons", "K4_random"),
            "K4_CL_cons_minus_CL": paired("K4_CL_cons", "K4_CL"),
            "per_episode": per_arm,
        }
        print(layer, {k: round(v, 4) for k, v in result["layers"][f"layer{layer}"]["means"].items()}, flush=True)

    passed = True
    for lr in result["layers"].values():
        ok = (lr["median_max_share_CL_cons"] < 0.40
              and lr["K4_CL_cons_minus_K1_full"]["ci95"][0] > 0
              and lr["K4_CL_cons_minus_balanced_free"]["ci95"][0] > 0)
        lr["G1c_layer_passed"] = bool(ok)
        passed = passed and ok
    result["G1c_passed"] = bool(passed)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_episode"} for k, v in result["layers"].items()}, indent=1), flush=True)
    print("G1c_passed =", passed, flush=True)


if __name__ == "__main__":
    main()
