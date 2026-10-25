"""R1b 预试验：选择 γ₀、τ（种子 900–902，与正式实验无重叠）。判据：N=4000 时收敛到 M* 的相对误差最小且不发散。"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot", ROOT / "pilot.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
D, R_NOISE, N, EPS = 8, 0.5, 4000, 1e-6

def m_star(rng_seed):
    rng = np.random.default_rng(rng_seed); C = P.make_C(rng, D)
    X, _ = P.simulate(rng, C, R_NOISE, 200000)
    x0, x1 = X[:-1], X[1:]
    return (x1.T @ x0) @ np.linalg.inv(x0.T @ x0), C, X

def lms(X, g0, tau):
    M = np.zeros((D, D)); err = []
    for t in range(1, len(X)):
        x_prev, x = X[t-1], X[t]
        g = g0 / (1.0 + t / tau)
        e = x - M @ x_prev
        M = M + g * np.outer(e, x_prev) / (EPS + x_prev @ x_prev)
    return M

if __name__ == "__main__":
    res = {}
    for g0 in [0.02, 0.05, 0.1, 0.2]:
        for tau in [100.0, 1000.0]:
            errs = []
            for s in [900, 901, 902]:
                Ms, C, _ = m_star(s)
                rng = np.random.default_rng(s + 1)
                X, _ = P.simulate(rng, C, R_NOISE, N)
                M = lms(X, g0, tau)
                errs.append(np.linalg.norm(M - Ms) / np.linalg.norm(Ms))
            res[f"g0={g0},tau={tau}"] = float(np.mean(errs))
            print(f"g0={g0} tau={tau} rel_err={np.mean(errs):.4f}", flush=True)
    out = ROOT / "pilot_r1b.json"; out.write_text(json.dumps(res, indent=1)); print("best", min(res, key=res.get))
