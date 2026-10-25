"""第 5 步正式判定（test，E028 划分；配置由 pilot.py 选定：token 逐个、d=8；层 3、层 7）。
S5-a：NMSE_base − NMSE_learn 的 episode 级 95% 区间下界 > 0；S5-b：NMSE_marg − NMSE_learn 的下界 > 0。"""
import importlib.util, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot5", ROOT / "pilot.py"); P5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(P5)
import torch
DATA = P5.d["data"]
def per_episode(eps, li, M, mean_vec, B, mu):
    out = []
    for ep in eps:
        num_b = num_m = num_l = den = 0.0
        for c in ep["chunks"]:
            kn = c[li][0].float().numpy()
            if len(kn) < 2: continue
            x = (kn - mu) @ B
            for i in range(1, len(x)):
                num_b += np.sum(x[i] ** 2); num_m += np.sum((x[i] - mean_vec) ** 2)
                num_l += np.sum((x[i] - M @ x[i-1]) ** 2); den += np.sum(x[i] ** 2)
        if den > 0: out.append((num_b / den, num_m / den, num_l / den))
    return np.array(out)
def boot(x, seed=0):
    r = np.random.default_rng(seed); m = np.array([r.choice(x, size=len(x), replace=True).mean() for _ in range(5000)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
if __name__ == "__main__":
    res = {}; passA = passB = True
    for li in P5.LAYERS:
        fit = P5.pairs_from(DATA["fit"], li, "token")
        Xf = np.vstack(fit); mu, B = P5.pca_basis(Xf, 8)
        fp = [(s - mu) @ B for s in fit]; M = P5.lms(fp); mean_vec = np.vstack(fp).mean(0)
        E = per_episode(DATA["test"], li, M, mean_vec, B, mu)
        a = boot(E[:, 0] - E[:, 2]); b = boot(E[:, 1] - E[:, 2])
        res[f"L{li}"] = {"base": float(E[:, 0].mean()), "marg": float(E[:, 1].mean()), "learn": float(E[:, 2].mean()),
                         "S5a_gap": a, "S5b_gap": b, "n_episodes": int(len(E))}
        passA &= a[1] > 0; passB &= b[1] > 0
        print(li, res[f"L{li}"], flush=True)
    res["S5_a_passed"] = bool(passA); res["S5_b_passed"] = bool(passB)
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float))
