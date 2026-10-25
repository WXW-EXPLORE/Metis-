"""第 5c 步：闭式岭回归的 chunk 级跨对话预测（docs/design_step5c_ridge.md，预先登记）。"""
import importlib.util, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c5b", ROOT / "run_crossconv.py"); C5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(C5)
P5, DATA, K = C5.P5, C5.DATA, C5.K
def ridge(seqs, lam_rel):
    X0 = np.vstack([s[:-1] for s in seqs]); X1 = np.vstack([s[1:] for s in seqs])
    S0 = X0.T @ X0 / len(X0); S1 = X1.T @ X0 / len(X0)
    s = np.trace(S0) / S0.shape[0]
    return S1 @ np.linalg.inv(S0 + lam_rel * s * np.eye(S0.shape[0]))
def shuffled(seqs, rng):
    return [s[rng.permutation(len(s))] for s in seqs]
def gain_and_nmse(tp, mean_vec, pred_mat):
    B = C5.per_ep_nmse(tp, lambda x: mean_vec); L = C5.per_ep_nmse(tp, lambda x: pred_mat @ x)
    return B, L
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    lams = [1e-3, 1e-2, 1e-1, 1.0]
    tune_sel = {}
    for li in P5.LAYERS:
        fit = C5.chunk_seqs(DATA["fit"], li); tune = C5.chunk_seqs(DATA["tune"], li)
        Xf = np.vstack(fit); mu, Bp = P5.pca_basis(Xf, K)
        fp = [(s - mu) @ Bp for s in fit]; tp = [(s - mu) @ Bp for s in tune]
        mean_vec = np.vstack(fp).mean(0)
        for lam in lams:
            Bm, L = gain_and_nmse(tp, mean_vec, ridge(fp, lam))
            tune_sel.setdefault(lam, []).append(float((Bm - L).mean()))
    best_lam = max(lams, key=lambda l: np.mean(tune_sel[l]))
    res = {"tune_gain_by_lambda": {str(l): float(np.mean(v)) for l, v in tune_sel.items()}, "selected_lambda": best_lam}
    passA = passB = True
    for li in P5.LAYERS:
        fit = C5.chunk_seqs(DATA["fit"], li); test = C5.chunk_seqs(DATA["test"], li)
        Xf = np.vstack(fit); mu, Bp = P5.pca_basis(Xf, K)
        fp = [(s - mu) @ Bp for s in fit]; tp = [(s - mu) @ Bp for s in test]
        mean_vec = np.vstack(fp).mean(0)
        Lm = ridge(fp, best_lam); Sm = ridge(shuffled(fp, rng), best_lam)
        Bm, L = gain_and_nmse(tp, mean_vec, Lm)
        SH = C5.per_ep_nmse(tp, lambda x: Sm @ x)
        xa = C5.boot(Bm - L); xb = C5.boot(SH - L)
        res[f"L{li}"] = {"NMSE_learn": float(L.mean()), "NMSE_mean": float(Bm.mean()), "NMSE_shuffled": float(SH.mean()),
                         "X_a_gain_vs_mean": xa, "X_b_gain_vs_shuffled": xb, "n_episodes": int(len(L))}
        passA &= xa[1] > 0; passB &= xb[1] > 0
    res["X_a_passed"] = bool(passA); res["X_b_passed"] = bool(passB); res["all_passed"] = bool(passA and passB)
    out = ROOT / "results_ridge"; out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float))
