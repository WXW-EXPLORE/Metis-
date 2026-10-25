"""R1b 正式实验：慢层在线学习 M（归一化 LMS，γ₀=0.1, τ=100，由 pilot_r1b.py 选定），快层量化创新，闭环解码。
判据见 docs/design_R1_division.md §10（运行前登记，未修改）。种子 0–19。"""
import importlib.util, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
P = _load("pilot", "pilot.py"); B = _load("pilot_r1b", "pilot_r1b.py")
import sys
D, R_NOISE, K_RANGE, G0, TAU = 8, 0.5, 3.0, 0.1, 100.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000   # R1b: 4000；R1b′: 20000（docs §12）
CKPTS = [200, 1000, 4000, 20000] if N >= 20000 else [200, 1000, 4000]
SEEDS = list(range(20))

def run_seed(seed):
    rng = np.random.default_rng(seed)
    C = P.make_C(rng, D)
    Ms, _, _ = B.m_star(seed)                       # M*（真实协方差，仅判据用）
    X, _ = P.simulate(rng, C, R_NOISE, N)
    M = np.zeros((D, D)); preds = np.zeros((N, D)); innov = np.zeros((N, D)); ck = {}
    for t in range(1, N):
        preds[t] = M @ X[t-1]
        innov[t] = X[t] - preds[t]
        g = G0 / (1.0 + t / TAU)
        M = M + g * np.outer(innov[t], X[t-1]) / (B.EPS + X[t-1] @ X[t-1])
        if t + 1 in CKPTS:
            ck[t + 1] = float(np.linalg.norm(M - Ms) / np.linalg.norm(Ms))
    out = {"seed": seed, "rel_err": ck}
    denom = np.mean(X ** 2)
    raw_scale = K_RANGE * np.sqrt(np.mean(X ** 2)); inn_scale = K_RANGE * np.sqrt(np.mean(innov ** 2))
    out["pred_nmse_learned"] = float(np.mean((X - preds) ** 2) / denom)
    Mpred = Ms
    out["pred_nmse_star"] = float(np.mean((X[1:] - X[:-1] @ Ms.T) ** 2) / denom)
    for b in P.BITS:
        qX, _ = P.quantize(X, b, raw_scale)
        out[f"F_{b}"] = float(np.mean((X - qX) ** 2) / denom)
        # 闭环解码：x̂_t = M_{t-1} x̂_{t-1} + q(r_t)，使用编码器同样的 M 版本（版本成本不计入）
        Mt = np.zeros((D, D)); xr = np.zeros(D); err = np.zeros((N, D)); 
        Mt_hist = [np.zeros((D, D))]
        Mcur = np.zeros((D, D)); rec = np.zeros((N, D)); rec[0] = X[0]
        Mseq = [np.zeros((D, D))] * N
        M2 = np.zeros((D, D))
        for t in range(1, N):
            Mseq[t] = M2.copy()
            innov_t = X[t] - M2 @ X[t-1]
            g = G0 / (1.0 + t / TAU)
            M2 = M2 + g * np.outer(innov_t, X[t-1]) / (B.EPS + X[t-1] @ X[t-1])
        # 注：上面重算 M 序列以保证编码器/解码器的版本一致
        M2 = np.zeros((D, D))
        for t in range(1, N):
            pred_t = M2 @ rec[t-1]                         # 解码器：基于已解码的 x̂_{t-1}
            innov_true = X[t] - (Mseq[t] @ X[t-1])        # 编码器：基于真实 x_{t-1} 的创新
            qI, _ = P.quantize(np.array([innov_true]), b, inn_scale)
            rec[t] = pred_t + qI[0]
            g = G0 / (1.0 + t / TAU)
            M2 = M2 + g * np.outer(innov_true, X[t-1]) / (B.EPS + X[t-1] @ X[t-1])
        out[f"FS_{b}"] = float(np.mean((X - rec) ** 2) / denom)
    return out

def bci(x, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = np.array([r.choice(x, size=x.size, replace=True).mean() for _ in range(5000)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))

if __name__ == "__main__":
    rows = [run_seed(s) for s in SEEDS]
    summ = {"params": {"g0": G0, "tau": TAU, "N": N, "bits": P.BITS, "n_seeds": len(SEEDS)}, "per_bits": {}}
    rel = {n: float(np.mean([r["rel_err"][n] for r in rows])) for n in CKPTS}
    mono = all(rel[CKPTS[i]] >= rel[CKPTS[i+1]] for i in range(len(CKPTS)-1))
    R1d = bool(rel[N] < 0.1 and mono)
    ga, ra = True, True
    for b in P.BITS:
        diff = [r[f"F_{b}"] - r[f"FS_{b}"] for r in rows]
        ratio = float(np.mean([r[f"FS_{b}"] / r[f"F_{b}"] for r in rows]))
        m, lo, hi = bci(diff)
        summ["per_bits"][str(b)] = {"F": float(np.mean([r[f"F_{b}"] for r in rows])),
                                    "FS_learned": float(np.mean([r[f"FS_{b}"] for r in rows])),
                                    "F_minus_FS": {"mean": m, "ci95": [lo, hi]}, "ratio": ratio}
        ga &= lo > 0; ra &= ratio < 1
    pl = float(np.mean([r["pred_nmse_learned"] for r in rows])); ps = float(np.mean([r["pred_nmse_star"] for r in rows]))
    summ["rel_err_by_N"] = rel; summ["R1_d_passed"] = R1d; summ["R1_d_monotone"] = bool(mono)
    summ["R1_a_passed"] = bool(ga); summ["R1_b_passed"] = bool(ra)
    summ["R1_c_pred_nmse_learned"] = pl; summ["R1_c_pred_nmse_star"] = ps
    summ["R1_c_passed"] = bool(pl < 1.0 and abs(pl - ps) / ps < 0.10)
    out = ROOT / ("results_r1b" if N == 4000 else f"results_r1b_N{N}"); out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); (out / "rows.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(summ, indent=1), flush=True)
