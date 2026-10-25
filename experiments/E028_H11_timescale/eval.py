"""E028（H11 / G4）：多时间尺度的离线评估（真实写入键，零样本，等总状态）。设计：docs/design_H11_timescale.md（§10 修订）。

臂：
  S1   ：单库，1024 行，kappa = kn/sqrt(1024)，P0 = 1024 I，λ = 0.9（与 RLS 基线一致）。
  T2L  ：两组各 512 行。路由：竞争学习原型（K=2，fit 数据，β=1，固定）按完整键 kn 分配；
         写入键为 kp = normalize(kn·Q512)/sqrt(512)（Q512 为 fit 的 PCA 基），P0 = 512 I。
         快组 λ_f，慢组 λ_s。
  T2R  ：同 T2L，但路由中心为随机（K=2，PCA 空间内随机，3 个种子）。

度量（保留曲线的 AUC）：每个 chunk 写入后，经过 a 次之后的写入（a ∈ {1,2,4,8,16}），用其自身键评估自检索余弦
（kp·M/(kp·z+1) 与 v 的余弦，逐 token，取均值）。每个 episode 对各年龄取平均（只用存在的配对），再对年龄取均值。

划分（设计 §10）：调参集 = 第 4–5 个对话（λ 网格选择）；测试集 = 第 6–10 个对话（判据）。
层：记忆层 3 与 7（blocks[0]、blocks[1]）。

用法（仓库根目录）：
    PYTHONPATH=.:src python experiments/E028_H11_timescale/eval.py --keys /hy-tmp/e028_keys.pt --out experiments/E028_H11_timescale/results
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from holistic.rls_metis import rls_commit_update

AGES = [1, 2, 4, 8, 16]
LAYERS = [0, 1]          # blocks[0]=3, blocks[1]=7
K = 2
KV = 1024
KD = 512
ETA, RHO = 0.05, 0.01
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def cl_prototypes(keys, seed, beta=1.0):
    """竞争学习（与 E023 相同）：在全键 kn 上学习 K 个原型，单次遍历。"""
    g = torch.Generator().manual_seed(seed)
    C = F.normalize(torch.randn(K, keys.shape[1], generator=g), dim=-1)
    u = torch.full((K,), 1.0 / K)
    for k in keys:
        s = C @ k - beta * (u - 1.0 / K)
        b = int(s.argmax())
        C[b] = F.normalize(C[b] + ETA * (k - C[b]), dim=0)
        oh = torch.zeros(K); oh[b] = 1.0
        u = (1 - RHO) * u + RHO * oh
    return C


def random_centers(Q, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    return F.normalize(torch.randn(K, Q.shape[1], generator=g) @ Q.T, dim=-1)


def episode_layer(ep, li):
    """返回 (chunks)：每个 chunk 为 (kn (T,1024) float, v (T,1024) float)，已在设备上。"""
    out = []
    for chunk in ep["chunks"]:
        kn, v = chunk[li]
        out.append((kn.to(DEV, torch.float32), v.to(DEV, torch.float32)))
    return out


def run_arm(chunks, arm, lams=None, Q=None, centers=None):
    """在一个 episode 的一个层上运行一个臂，返回：每个年龄的 (chunk 级) 平均余弦列表，以及组占比计数。"""
    N = len(chunks)
    if arm == "S1":
        states = [(torch.zeros(1, KV, KV, device=DEV), (KV * torch.eye(KV, device=DEV))[None].clone(),
                   torch.zeros(1, KV, 1, device=DEV))]
    elif arm == "S1p":   # 对照：单库，PCA-512 键，512 行（投影成本）
        states = [(torch.zeros(1, KD, KV, device=DEV), (KD * torch.eye(KD, device=DEV))[None].clone(),
                   torch.zeros(1, KD, 1, device=DEV))]
    else:
        states = [(torch.zeros(1, KD, KV, device=DEV), (KD * torch.eye(KD, device=DEV))[None].clone(),
                   torch.zeros(1, KD, 1, device=DEV)) for _ in range(K)]
    keys_written = []      # 每个 chunk 的 (kp, v, group_index) 以便评估
    counts = np.zeros(K if arm.startswith("T") else 1)
    per_age = {a: [] for a in AGES}
    for t in range(N):
        kn, v = chunks[t]
        if arm in ("S1", "S1p"):
            kp = kn / (KV ** 0.5) if arm == "S1" else F.normalize(kn @ Q, dim=-1) / (KD ** 0.5)
            g_idx = torch.zeros(kn.shape[0], dtype=torch.long, device=DEV)
            M, P, z = states[0]
            M, P, z = rls_commit_update(M, P, z, kp[None], v[None], torch.ones(1, kn.shape[0], device=DEV),
                                        0.9, p_max=(KV * KV if arm == "S1" else KD * KD))
            states[0] = (M, P, z)
        else:
            kp = F.normalize(kn @ Q, dim=-1) / (KD ** 0.5)
            g_idx = (kn @ centers.T).argmax(-1)
            new = []
            for g in range(K):
                w = (g_idx == g).float()[None]
                M, P, z = states[g]
                M, P, z = rls_commit_update(M, P, z, kp[None], v[None], w, lams[g], p_max=KD * KD)
                new.append((M, P, z))
            states = new
            counts += np.bincount(g_idx.cpu().numpy(), minlength=K)
        keys_written.append((kp, v, g_idx))
        # 评估：对于每个年龄 a，chunk c = t − a 的自检索
        for a in AGES:
            c = t - a
            if c < 0:
                continue
            kp_c, v_c, g_c = keys_written[c]
            cos_all = []
            for g in range(len(states)):
                sel = (g_c == g) if arm.startswith("T") else torch.ones_like(g_c, dtype=torch.bool)
                if int(sel.sum()) == 0:
                    continue
                M, _P, z = states[g]
                kq = kp_c[sel]
                out = (kq @ M[0]) / (kq @ z[0] + 1.0)
                cos_all.append(F.cosine_similarity(out, v_c[sel], dim=-1))
            if cos_all:
                per_age[a].append(float(torch.cat(cos_all).mean()))
    return per_age, counts


def episode_auc(per_age):
    vals = [np.mean(v) for a, v in per_age.items() if len(v) > 0]
    return float(np.mean(vals)) if vals else float("nan")


def fit_basis_and_protos(fit_eps, li):
    keys = torch.cat([kn.to(DEV, torch.float32) for ep in fit_eps for (kn, _) in
                      [c[li] for c in ep["chunks"]]], 0)
    _, _, Vh = torch.linalg.svd(keys, full_matrices=False)
    Q = Vh[:KD].T.contiguous()
    C_cl = cl_prototypes(keys.cpu(), seed=2000).to(DEV)
    return Q, C_cl


def bci(x, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(5000)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def evaluate(split_eps, li, arm_cfg):
    """arm_cfg: dict(name, arm, lams, Q, centers)；返回每个 episode 的 AUC 与组计数。"""
    aucs, counts = [], np.zeros(K)
    for ep in split_eps:
        chunks = episode_layer(ep, li)
        per_age, cnt = run_arm(chunks, arm_cfg["arm"], arm_cfg.get("lams"), arm_cfg.get("Q"), arm_cfg.get("centers"))
        aucs.append(episode_auc(per_age))
        if arm_cfg["arm"].startswith("T"):
            counts += cnt
    return np.array(aucs), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_tune", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    d = torch.load(args.keys, weights_only=False, mmap=True)
    fit = d["data"]["fit"]; tune = d["data"]["tune"]; test = d["data"]["test"]
    if args.n_tune: tune = tune[: args.n_tune]
    if args.n_test: test = test[: args.n_test]

    lam_grid = [(lf, ls) for lf in (0.3, 0.5, 0.7) for ls in (0.95, 0.99, 0.995)]
    layer_cfg = {}
    for li in LAYERS:
        Q, C_cl = fit_basis_and_protos(fit, li)
        layer_cfg[li] = {"Q": Q, "C_cl": C_cl,
                         "C_rand": {s: random_centers(Q.cpu(), s).to(DEV) for s in (0, 1, 2)}}

    # ---- 调参集：S1 与 T2L 网格 ----
    tune_res = {}
    for li in LAYERS:
        s1, _ = evaluate(tune, li, {"arm": "S1"})
        for lf, ls in lam_grid:
            t2, _ = evaluate(tune, li, {"arm": "T2L", "lams": [lf, ls], "Q": layer_cfg[li]["Q"],
                                        "centers": layer_cfg[li]["C_cl"]})
            diff = t2 - s1
            tune_res.setdefault(f"{lf},{ls}", {})[li] = bci(diff)
            print("tune", li, lf, ls, [round(x, 4) for x in bci(diff)], flush=True)
    score = {k: np.mean([v[li][0] for li in LAYERS]) for k, v in tune_res.items()}
    best = max(score, key=score.get)
    lf, ls = [float(x) for x in best.split(",")]
    best_half = np.mean([(tune_res[best][li][2] - tune_res[best][li][1]) / 2 for li in LAYERS])
    detectable = score[best] > best_half
    print("best lambda on tune:", lf, ls, "mean diff", round(score[best], 5), "half-width", round(best_half, 5),
          "detectable", detectable, flush=True)

    # ---- 测试集：判据 ----
    test_res = {"S1": {}, "T2L": {}, "T2R": {}}
    for li in LAYERS:
        s1, _ = evaluate(test, li, {"arm": "S1"})
        t2l, cnt = evaluate(test, li, {"arm": "T2L", "lams": [lf, ls], "Q": layer_cfg[li]["Q"],
                                       "centers": layer_cfg[li]["C_cl"]})
        t2r = []
        for s in (0, 1, 2):
            r, _ = evaluate(test, li, {"arm": "T2R", "lams": [lf, ls], "Q": layer_cfg[li]["Q"],
                                       "centers": layer_cfg[li]["C_rand"][s]})
            t2r.append(r)
        t2r = np.mean(np.stack(t2r), axis=0)
        test_res["S1"][li] = s1.tolist(); test_res["T2L"][li] = t2l.tolist(); test_res["T2R"][li] = t2r.tolist()
        share = cnt / cnt.sum()
        test_res.setdefault("share_T2L", {})[li] = share.tolist()
        print("test layer", li, "S1", np.nanmean(s1), "T2L", np.nanmean(t2l), "T2R", np.nanmean(t2r), flush=True)

    judge = {}
    if detectable:
        ga = {li: bci(np.array(test_res["T2L"][li]) - np.array(test_res["S1"][li])) for li in LAYERS}
        gb = {li: bci(np.array(test_res["T2L"][li]) - np.array(test_res["T2R"][li])) for li in LAYERS}
        judge["G4_a"] = {str(li): ga[li] for li in LAYERS}
        judge["G4_a_passed"] = bool(all(ga[li][1] > 0 for li in LAYERS))
        judge["G4_b"] = {str(li): gb[li] for li in LAYERS}
        judge["G4_b_passed"] = bool(all(gb[li][1] > 0 for li in LAYERS))
        judge["G4_c_min_share"] = float(min(min(test_res["share_T2L"][li]) for li in LAYERS))
        judge["G4_c_passed"] = bool(judge["G4_c_min_share"] >= 0.2)
    else:
        judge["note"] = "调参集上最佳差值未超过检出下限：不进入测试集判定（设计 §10）。"

    res = {"lambda_selected": [lf, ls], "tune_mean_diff": float(score[best]), "tune_half_width": float(best_half),
           "tune_detectable": bool(detectable), "tune_grid": {k: {str(li): tune_res[k][li] for li in LAYERS}
                                                             for k in tune_res},
           "test": test_res, "judgement": judge, "n_tune": len(tune), "n_test": len(test)}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else o))
    print(json.dumps(judge, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
