"""E030（A）：残差值存储的解码缺口。设计：docs/design_E030_decode.md

载体级（不训练模型）。键来自 E028 录制（/hy-tmp/e028_keys.pt，单位键 kn）。
拟合（M_v、B、读出系数）用 fit 划分；评测用 tune 划分；test 不读取。
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from holistic.rls_metis import rls_commit_update  # noqa: E402

LAM = 0.9
DEV = "cuda"
VARIANTS = {
    "b0": ["m0"],               # 纯存储基线：存 v
    "b1": ["m1", "comp1"],      # 残差 + 读出端补偿 M_v (q B)
    "b1naive": ["m1"],          # 残差，无补偿（展示缺口）
    "b2": ["m1", "comp2"],      # 残差 + 存前驱键 M_v (q M2)（2× 状态）
}


def ridge(XtX, XtY, eps=1e-3):
    d = XtX.shape[0]
    reg = eps * torch.trace(XtX) / d
    return torch.linalg.solve(XtX + reg * torch.eye(d, device=DEV), XtY)


def fit_maps(episodes, layer, kv):
    XtX_p = torch.zeros(kv, kv, device=DEV)
    XtY_p = torch.zeros(kv, kv, device=DEV)
    XtX_b = torch.zeros(kv, kv, device=DEV)
    XtY_b = torch.zeros(kv, kv, device=DEV)
    for ep in episodes:
        K, V = _flat(ep, layer)
        Kp = torch.cat([torch.zeros(1, kv, device=DEV), K[:-1]], 0)
        XtX_p += Kp.T @ Kp
        XtY_p += Kp.T @ V
        XtX_b += K.T @ K
        XtY_b += K.T @ Kp
    Mv = ridge(XtX_p, XtY_p)     # kn_{t-1} -> v_t
    Bm = ridge(XtX_b, XtY_b)     # kn_t -> kn_{t-1}
    return Mv, Bm


def _flat(ep, layer):
    Ks, Vs = [], []
    for chunk in ep["chunks"]:
        k, v = chunk[layer]
        Ks.append(k.float().to(DEV))
        Vs.append(v.float().to(DEV))
    return torch.cat(Ks, 0), torch.cat(Vs, 0)


def run_episode(ep, layer, kv, Mv, Bm, acc, mode, errs):
    """按 chunk 提交 RLS 并在提交后读出本 chunk（年龄 0）。
    mode='fit'：累加 Gram 与交叉项；mode='eval'：记录逐 token 相对误差（系数已冻结于 acc）。"""
    d = kv
    M = torch.zeros(1, d, 3 * d, device=DEV)
    P = (kv * torch.eye(d, device=DEV))[None].clone()
    z = torch.zeros(1, d, 1, device=DEV)
    K, V = _flat(ep, layer)
    Kp = torch.cat([torch.zeros(1, d, device=DEV), K[:-1]], 0)
    R = V - Kp @ Mv
    s = 0
    for chunk in ep["chunks"]:
        n = chunk[layer][0].shape[0]
        e = s + n
        Kc, Vc, Rc, Kpc = K[s:e], V[s:e], R[s:e], Kp[s:e]
        VC = torch.cat([Vc, Rc, Kpc], -1)[None]
        M, P, z = rls_commit_update(M, P, z, (Kc / kv ** 0.5)[None], VC, torch.ones(1, n, device=DEV),
                                    LAM, p_max=kv * kv)
        m = Kc @ M[0]                              # 年龄 0 读出：单位查询
        terms = {
            "m0": m[:, :d],
            "m1": m[:, d:2 * d],
            "comp1": (Kc @ Bm) @ Mv,
            "comp2": m[:, 2 * d:] @ Mv,
        }
        for name, tn in VARIANTS.items():
            T = [terms[t] for t in tn]
            if mode == "fit":
                G = acc[name]["G"]; h = acc[name]["h"]
                for i in range(len(T)):
                    h[i] += (T[i] * Vc).sum()
                    for j in range(len(T)):
                        G[i, j] += (T[i] * T[j]).sum()
            else:
                pred = sum(a * t for a, t in zip(acc[name]["coef"], T))
                errs[name].append(((pred - Vc).norm(dim=-1) / Vc.norm(dim=-1).clamp(min=1e-12)).cpu())
        if mode == "eval":
            # 补偿误差（诊断）：M_v (Kprev - B K) 的相对大小
            comp_err = ((Kpc - Kc @ Bm) @ Mv).norm(dim=-1) / Vc.norm(dim=-1).clamp(min=1e-12)
            errs["comp_err"].append(comp_err.cpu())
        s = e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="/hy-tmp/e028_keys.pt")
    ap.add_argument("--out", default=str(ROOT / "experiments/E030_decode/result.json"))
    ap.add_argument("--max_episodes", type=int, default=None, help="试验用：每个划分只取前若干个 episode")
    args = ap.parse_args()

    data = torch.load(args.keys, mmap=True, weights_only=False, map_location="cpu")
    kv = data["kv"]
    fit = data["data"]["fit"]
    tune = data["data"]["tune"]
    if args.max_episodes:
        fit, tune = fit[: args.max_episodes], tune[: args.max_episodes]
    layers = list(range(len(data["blocks"])))
    result = {"blocks": data["blocks"], "n_fit": len(fit), "n_tune": len(tune), "lam": LAM, "layers": {}}

    for li in layers:
        Mv, Bm = fit_maps(fit, li, kv)
        d = kv
        acc = {name: {"G": torch.zeros(len(tn), len(tn), device=DEV, dtype=torch.float64),
                      "h": torch.zeros(len(tn), device=DEV, dtype=torch.float64)}
               for name, tn in VARIANTS.items()}
        # 第一遍（fit）：系数用 float64 累加，保证 Gram 精度
        for ep in fit:
            run_episode(ep, li, kv, Mv, Bm, acc, "fit", None)
        for name in VARIANTS:
            G = acc[name]["G"].cpu().double(); h = acc[name]["h"].cpu().double()
            coef = torch.linalg.solve(G + 1e-9 * torch.eye(G.shape[0], dtype=G.dtype) * G.trace(), h)
            acc[name]["coef"] = [c.float().to(DEV) for c in coef]
        errs = {name: [] for name in VARIANTS}
        errs["comp_err"] = []
        for ep in tune:
            run_episode(ep, li, kv, Mv, Bm, acc, "eval", errs)
        stat = {}
        for name in list(VARIANTS) + ["comp_err"]:
            e = torch.cat(errs[name])
            stat[name] = {"median": float(e.median()), "mean": float(e.mean()), "n": int(e.numel())}
        stat["coef"] = {name: [float(c) for c in acc[name]["coef"]] for name in VARIANTS}
        stat["b1_minus_b0_median"] = stat["b1"]["median"] - stat["b0"]["median"]
        stat["b1_ok"] = stat["b1_minus_b0_median"] <= 0.05
        result["layers"][str(data["blocks"][li])] = stat
        print(f"layer {data['blocks'][li]}: b0 {stat['b0']['median']:.4f}  b1 {stat['b1']['median']:.4f}  "
              f"b1naive {stat['b1naive']['median']:.4f}  b2 {stat['b2']['median']:.4f}  "
              f"comp {stat['comp_err']['median']:.4f}  b1_ok {stat['b1_ok']}", flush=True)

    result["b1_adequate_all_layers"] = all(v["b1_ok"] for v in result["layers"].values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print("b1 adequate at all layers:", result["b1_adequate_all_layers"], flush=True)


if __name__ == "__main__":
    main()
