"""第 3 步：版本一致性（docs/design_R1_division.md §14，预先登记）。"""
import importlib.util, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
P = _load("pilot", "pilot.py")
D, R_NOISE, K_RANGE, G0, TAU = 8, 0.5, 3.0, 0.1, 100.0
import sys
N, TS, BITS = 20000, 10000, 4
L = int(sys.argv[1]) if len(sys.argv) > 1 else 500   # 块长（docs §14 为 500；§16 为 50、100）
EPS = 1e-6
SEEDS = list(range(20))

def run_seed(seed):
    rng = np.random.default_rng(seed)
    C1 = P.make_C(rng, D)
    Qr, _ = np.linalg.qr(rng.standard_normal((D, D)))
    C2 = Qr @ C1                                   # 知识更新：观测映射改变
    # 生成两段环境的序列（与 P.simulate 相同动力学，C 不同；状态在段间连续）
    L_w = np.linalg.cholesky(P.Q + 1e-12 * np.eye(P.M))
    s = rng.standard_normal(P.M); X = np.zeros((N, D))
    for t in range(N):
        s = P.A @ s + L_w @ rng.standard_normal(P.M)
        C = C1 if t < TS else C2
        X[t] = C @ s + np.sqrt(R_NOISE) * rng.standard_normal(D)
    # 编码：学习 M，存储量化创新（用于前半段）
    M = np.zeros((D, D)); Mseq = np.zeros((N, D, D)); innov = np.zeros((N, D))
    for t in range(1, N):
        Mseq[t] = M
        innov[t] = X[t] - M @ X[t-1]
        g = G0 / (1.0 + t / TAU)
        M = M + g * np.outer(innov[t], X[t-1]) / (EPS + X[t-1] @ X[t-1])
    M_final = M
    inn_scale = K_RANGE * np.sqrt(np.mean(innov[1:] ** 2))
    qI = np.zeros_like(innov)
    for t in range(1, N):
        qI[t], _ = P.quantize(innov[t:t+1], BITS, inn_scale)
        qI[t] = qI[t]
    # 解码前半段：闭环 x̂_t = M_{t-1}^{v} x̂_{t-1} + q(r_t)
    def decode(Mfun, T):
        rec = np.zeros((T, D)); rec[0] = X[0]
        for t in range(1, T):
            rec[t] = Mfun(t) @ rec[t-1] + qI[t]
        return rec
    seg = TS
    rec_V0 = decode(lambda t: M_final, seg)
    rec_V1 = decode(lambda t: Mseq[t], seg)
    blocks = np.arange(seg) // L
    Mblock = {}
    # V3：每块使用块首时刻的参数版本
    rec_V3 = np.zeros((seg, D)); rec_V3[0] = X[0]
    for t in range(1, seg):
        b = blocks[t]
        ver = Mseq[(b * L) if b * L > 0 else 1]
        rec_V3[t] = ver @ rec_V3[t-1] + qI[t]
    denom = np.mean(X[:seg] ** 2)
    out = {"seed": seed,
           "V0": float(np.mean((X[:seg] - rec_V0) ** 2) / denom),
           "V1": float(np.mean((X[:seg] - rec_V1) ** 2) / denom),
           "V3": float(np.mean((X[:seg] - rec_V3) ** 2) / denom),
           "V3_overhead_bits_per_coord_step": float(np.log2(seg / L) / L / D)}
    return out

def bci(x, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = np.array([r.choice(x, size=x.size, replace=True).mean() for _ in range(5000)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))

if __name__ == "__main__":
    rows = [run_seed(s) for s in SEEDS]
    V0 = np.array([r["V0"] for r in rows]); V1 = np.array([r["V1"] for r in rows]); V3 = np.array([r["V3"] for r in rows])
    ratio_V0V1 = float(V0.mean() / V1.mean())
    ratio_V3V1 = float(V3.mean() / V1.mean())
    oh = rows[0]["V3_overhead_bits_per_coord_step"]
    summ = {"V0_mean": float(V0.mean()), "V1_mean": float(V1.mean()), "V3_mean": float(V3.mean()),
            "V0_over_V1": ratio_V0V1, "V0_over_V1_ci95": bci(V0 / V1)[1:],
            "V3_over_V1": ratio_V3V1, "V3_overhead": oh,
            "V-a_passed": bool(ratio_V0V1 >= 1.5), "V-b_passed": bool(ratio_V3V1 <= 1.1 and oh <= 0.02),
            "params": {"d": D, "R": R_NOISE, "b": BITS, "N": N, "T_s": TS, "L": L, "n_seeds": len(SEEDS)}}
    out = ROOT / ("results_version" if L == 500 else f"results_version_L{L}"); out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); (out / "rows.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(summ, indent=1), flush=True)
