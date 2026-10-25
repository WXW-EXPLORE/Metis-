"""模块 D1（结构性正交分块）：设计 docs/design_D_decoupling.md。

臂（等总状态 1024 行，λ 沿用 E028 锁定值）：
  T2L ：两组共用 PCA-512 基（E028 的结果从 result.json 读取，不重算）。
  D1  ：两组使用正交的 PCA 分块：组 1 用前 512 个主成分，组 2 用后 512 个主成分。路由与 T2L 相同（竞争学习原型）。
  S1  ：单库，全键（从 result.json 读取）。

度量：
  保留曲线 AUC（与 E028 相同）；G5-a/b 为配对差的 episode 级 bootstrap 区间；
  G5-c 组间串扰：把两组的键（投影后）提升回 1024 维，取两组键之间的平均绝对余弦（随机抽样配对）；
  G5-d 组占比（token）的最小值。

判据（运行前登记，docs/design_D_decoupling.md §4）：
  G5-a：D1 − T2L 的区间下界 > 0（两层）；G5-b：D1 − S1 的区间下界 > 0（两层）；
  G5-c：D1 的组间串扰均值 < T2L 的组间串扰均值（两层）；G5-d：D1 的最小组占比 ≥ 0.2。

注意：测试集已在 G4 中用过（E028）。本次沿用同一划分，结论需注明这一点。

用法（仓库根目录）：
    PYTHONPATH=.:src python experiments/E028_H11_timescale/eval_d1.py --keys /hy-tmp/e028_keys.pt --result experiments/E028_H11_timescale/results/result.json --out experiments/E028_H11_timescale/results_d1
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from holistic.rls_metis import rls_commit_update


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


E = _load("e28eval", "experiments/E028_H11_timescale/eval.py")
K, KV, KD = E.K, E.KV, E.KD
DEV = E.DEV


def fit_basis_split(fit_eps, li):
    keys = torch.cat([kn.to(DEV, torch.float32) for ep in fit_eps for (kn, _) in
                      [c[li] for c in ep["chunks"]]], 0)
    _, _, Vh = torch.linalg.svd(keys, full_matrices=False)
    Qa = Vh[:KD].T.contiguous()          # 前 512 个主成分（与 T2L 的基相同）
    Qb = Vh[KD:2 * KD].T.contiguous()    # 后 512 个主成分（与 Qa 正交）
    C_cl = E.cl_prototypes(keys.cpu(), seed=2000).to(DEV)
    return Qa, Qb, C_cl


def run_d1(chunks, lam, Qa, Qb, centers):
    """两组、两个正交子空间。返回：per-age 余弦、组计数、以及用于串扰的 (lifted kn, group) 样本。"""
    Qs = [Qa, Qb]
    states = [(torch.zeros(1, KD, KV, device=DEV), (KD * torch.eye(KD, device=DEV))[None].clone(),
               torch.zeros(1, KD, 1, device=DEV)) for _ in range(K)]
    written = []
    counts = np.zeros(K)
    per_age = {a: [] for a in E.AGES}
    for t in range(len(chunks)):
        kn, v = chunks[t]
        g_idx = (kn @ centers.T).argmax(-1)
        kps = [F.normalize(kn @ Qs[g], dim=-1) / (KD ** 0.5) for g in range(K)]
        new = []
        for g in range(K):
            w = (g_idx == g).float()[None]
            M, P, z = states[g]
            M, P, z = rls_commit_update(M, P, z, kps[g][None], v[None], w, lam[g], p_max=KD * KD)
            new.append((M, P, z))
        states = new
        counts += np.bincount(g_idx.cpu().numpy(), minlength=K)
        written.append((kps, v, g_idx))
        for a in E.AGES:
            c = t - a
            if c < 0:
                continue
            kps_c, v_c, g_c = written[c]
            cos_all = []
            for g in range(K):
                sel = (g_c == g)
                if int(sel.sum()) == 0:
                    continue
                M, _P, z = states[g]
                kq = kps_c[g][sel]
                out = (kq @ M[0]) / (kq @ z[0] + 1.0)
                cos_all.append(F.cosine_similarity(out, v_c[sel], dim=-1))
            if cos_all:
                per_age[a].append(float(torch.cat(cos_all).mean()))
    return per_age, counts


def crosstalk(test_eps, li, centers, proj_fn, n_pairs=20000, seed=0):
    """组间串扰：各组键经 proj_fn 提升回 1024 维后，两组之间的平均绝对余弦（随机配对）。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    lifted = {0: [], 1: []}
    for ep in test_eps:
        for kn, _ in [c[li] for c in ep["chunks"]]:
            kn = kn.to(DEV, torch.float32)
            gi = (kn @ centers.T).argmax(-1)
            for gg in (0, 1):
                sel = gi == gg
                if int(sel.sum()) > 0:
                    lifted[gg].append(proj_fn(kn[sel], gg))
    A = torch.cat(lifted[0], 0); B = torch.cat(lifted[1], 0)
    ia = torch.randint(0, A.shape[0], (n_pairs,), generator=g).to(DEV)
    ib = torch.randint(0, B.shape[0], (n_pairs,), generator=g).to(DEV)
    cos = F.cosine_similarity(A[ia], B[ib], dim=-1).abs()
    return float(cos.mean())


def bci(x, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(5000)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    main_res = json.loads(Path(args.result).read_text())
    lam = main_res["lambda_selected"]
    d = torch.load(args.keys, weights_only=False, mmap=True)
    fit, test = d["data"]["fit"], d["data"]["test"]

    res = {"lambda": lam, "layers": {}, "judgement": {}}
    for li in E.LAYERS:
        Qa, Qb, C_cl = fit_basis_split(fit, li)
        Q_shared = torch.cat([Qa, Qb], 1)   # 只用于说明；T2L 的基是 Qa（前 512）
        per_d1, counts_d1, per_t2l_eps = [], np.zeros(K), []
        aucs_d1 = []
        for ep in test:
            chunks = E.episode_layer(ep, li)
            pa, cnt = run_d1(chunks, lam, Qa, Qb, C_cl)
            aucs_d1.append(E.episode_auc(pa))
            counts_d1 += cnt
        aucs_d1 = np.array(aucs_d1)
        s1 = np.array(main_res["test"]["S1"][str(li)])
        t2l = np.array(main_res["test"]["T2L"][str(li)])

        # 串扰：T2L 用共享基 Qa（两组都投影到同一 512 子空间，提升回 1024）；D1 用各自的 Qa / Qb
        xt_t2l = crosstalk(test, li, C_cl, lambda x, gg: (x @ Qa) @ Qa.T)
        xt_d1 = crosstalk(test, li, C_cl, lambda x, gg: (x @ (Qa if gg == 0 else Qb)) @ (Qa if gg == 0 else Qb).T)

        share = counts_d1 / counts_d1.sum()
        res["layers"][str(li)] = {
            "D1_mean_auc": float(np.nanmean(aucs_d1)), "T2L_mean_auc": float(np.nanmean(t2l)),
            "S1_mean_auc": float(np.nanmean(s1)),
            "D1_minus_T2L": bci(aucs_d1 - t2l), "D1_minus_S1": bci(aucs_d1 - s1),
            "crosstalk_T2L": xt_t2l, "crosstalk_D1": xt_d1,
            "D1_group_share": share.tolist(),
        }
        print("layer", li, json.dumps(res["layers"][str(li)], default=float), flush=True)

    L = res["layers"]
    J = res["judgement"]
    J["G5_a_passed"] = bool(all(L[k]["D1_minus_T2L"][1] > 0 for k in L))
    J["G5_b_passed"] = bool(all(L[k]["D1_minus_S1"][1] > 0 for k in L))
    J["G5_c_passed"] = bool(all(L[k]["crosstalk_D1"] < L[k]["crosstalk_T2L"] for k in L))
    J["G5_d_min_share"] = float(min(min(L[k]["D1_group_share"]) for k in L))
    J["G5_d_passed"] = bool(J["G5_d_min_share"] >= 0.2)
    J["all_passed"] = bool(J["G5_a_passed"] and J["G5_b_passed"] and J["G5_c_passed"] and J["G5_d_passed"])
    (out / "result_d1.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(J, indent=1), flush=True)


if __name__ == "__main__":
    main()
