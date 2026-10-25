"""第 5 步预试验（只用 fit/tune，不碰 test）：选择序列形式（chunk 均值 / token 逐个）与 PCA 维度 d。
选择规则（预先写定）：在 tune 上最大化 S5-b 的边际（NMSE_marg − NMSE_learn）；并报告 S5-a（NMSE_base − NMSE_learn）。"""
import json
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
d = torch.load("/hy-tmp/e028_keys.pt", weights_only=False, mmap=True)
LAYERS = [0, 1]          # 层 3、层 7
def pairs_from(eps, li, form):
    seqs = []
    for ep in eps:
        if form == "chunk":
            seq = [c[li][0].float().mean(0).numpy() for c in ep["chunks"]]
            seqs.append(np.array(seq))
        else:  # token
            for c in ep["chunks"]:
                kn = c[li][0].float().numpy()
                if len(kn) > 1: seqs.append(kn)
    return seqs
def pca_basis(X, k):
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k].T
def lms(seqs, g0=0.1, tau=100.0, eps=1e-6):
    D = seqs[0].shape[1]; M = np.zeros((D, D)); t = 0
    for seq in seqs:
        for i in range(1, len(seq)):
            t += 1; g = g0 / (1 + t / tau)
            e = seq[i] - M @ seq[i-1]
            M += g * np.outer(e, seq[i-1]) / (eps + seq[i-1] @ seq[i-1])
    return M
def nmse(seqs, pred_fn, mu):
    num = den = 0.0
    for seq in seqs:
        for i in range(1, len(seq)):
            p = pred_fn(seq[i-1]); num += np.sum((seq[i] - p) ** 2); den += np.sum(seq[i] ** 2)
    return num / den
if __name__ == "__main__":
    res = {}
    for form in ["chunk", "token"]:
        for li in LAYERS:
            fit = pairs_from(d["data"]["fit"], li, form); tune = pairs_from(d["data"]["tune"], li, form)
            Xf = np.vstack(fit)
            for k in [8, 32]:
                mu, B = pca_basis(Xf, k)
                proj = lambda seqs: [(s - mu) @ B for s in seqs]
                fp, tp = proj(fit), proj(tune)
                M = lms(fp)
                mean_vec = np.vstack(fp).mean(0)
                base = nmse(tp, lambda x: np.zeros_like(x), None)
                marg = nmse(tp, lambda x: mean_vec, None)
                learn = nmse(tp, lambda x: M @ x, None)
                res[f"{form}_L{li}_d{k}"] = {"base": base, "marg": marg, "learn": learn,
                                              "S5a_gap": base - learn, "S5b_gap": marg - learn}
                print(form, li, k, {x: round(y, 4) for x, y in res[f"{form}_L{li}_d{k}"].items()}, flush=True)
    # 选择：对每个 (form, d)，两层的 S5-b 平均，取最大
    agg = {}
    for form in ["chunk", "token"]:
        for k in [8, 32]:
            vals = [res[f"{form}_L{li}_d{k}"]["S5b_gap"] for li in LAYERS]
            agg[f"{form}_d{k}"] = float(np.mean(vals))
    best = max(agg, key=agg.get)
    out = ROOT / "experiments/step5_real/pilot.json"
    out.write_text(json.dumps({"results": res, "S5b_mean_by_config": agg, "selected": best}, indent=1, default=float))
    print("selected", best, round(agg[best], 4))
