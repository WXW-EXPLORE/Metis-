"""第 5b 步：跨对话泛化（chunk 级，真实键）。配置与判据见 docs/design_step5b_crossconv.md。"""
import importlib.util, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot5", ROOT / "pilot.py"); P5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P5)
DATA = P5.d["data"]; K = 8
def chunk_seqs(eps, li):
    return [np.array([c[li][0].float().mean(0).numpy() for c in ep["chunks"]]) for ep in eps if len(ep["chunks"]) > 1]
def lms_pairs(seqs, g0=0.1, tau=100.0, eps=1e-6):
    D = seqs[0].shape[1]; M = np.zeros((D, D)); t = 0
    for seq in seqs:
        for i in range(1, len(seq)):
            t += 1; g = g0 / (1 + t / tau); e = seq[i] - M @ seq[i-1]
            M += g * np.outer(e, seq[i-1]) / (eps + seq[i-1] @ seq[i-1])
    return M
def per_ep_nmse(seqs, pred):
    out = []
    for seq in seqs:
        num = den = 0.0
        for i in range(1, len(seq)):
            num += np.sum((seq[i] - pred(seq[i-1])) ** 2); den += np.sum(seq[i] ** 2)
        if den > 0: out.append(num / den)
    return np.array(out)
def boot(x, seed=0):
    r = np.random.default_rng(seed); m = np.array([r.choice(x, size=len(x), replace=True).mean() for _ in range(5000)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
if __name__ == "__main__":
    res = {}; passA = passB = True
    rng = np.random.default_rng(0)
    for li in P5.LAYERS:
        fit = chunk_seqs(DATA["fit"], li); test = chunk_seqs(DATA["test"], li)
        Xf = np.vstack(fit); mu, B = P5.pca_basis(Xf, K)
        fp = [(s - mu) @ B for s in fit]; tp = [(s - mu) @ B for s in test]
        M = lms_pairs(fp); mean_vec = np.vstack(fp).mean(0)
        shuf = [s[rng.permutation(len(s))] for s in fp]                 # 时间打乱对照（同边际，去掉时间结构）
        Msh = lms_pairs(shuf)
        L = per_ep_nmse(tp, lambda x: M @ x); Bm = per_ep_nmse(tp, lambda x: mean_vec)
        SH = per_ep_nmse(tp, lambda x: Msh @ x)
        xa = boot(Bm - L); xb = boot(SH - L)
        res[f"L{li}"] = {"NMSE_learn": float(L.mean()), "NMSE_mean": float(Bm.mean()), "NMSE_shuffled": float(SH.mean()),
                         "X_a_gain_vs_mean": xa, "X_b_gain_vs_shuffled": xb, "n_episodes": int(len(L))}
        passA &= xa[1] > 0; passB &= xb[1] > 0
        print(li, res[f"L{li}"], flush=True)
    res["X_a_passed"] = bool(passA); res["X_b_passed"] = bool(passB); res["all_passed"] = bool(passA and passB)
    out = ROOT / "results_crossconv"; out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float))
