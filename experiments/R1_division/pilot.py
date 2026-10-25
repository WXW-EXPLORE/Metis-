"""R1 预试验：为正式实验选择参数（观测维度 d、观测噪声 R、量化范围 k）。

使用与正式实验不同的种子（900+），只用于参数选择，不用于判定 R1-a…R1-d。
慢层此处使用真实参数（oracle）：预试验的目的是确定量化与可区分性的尺度，而不是学习效果。

选择规则（预先声明，本文件即登记）：
  (i)  量化范围 k：在 b=8 时，原始观测与创新的裁剪率均 < 1%，取满足条件的最小 k。
  (ii) (d, R)：在 b=4 时，FS 与 F 的 MSE 比值 ρ 满足 0.5 ≤ ρ ≤ 0.9
       （低于 0.5 说明收益过于显著、可能源于实现偏差；高于 0.9 说明效应不可检出）。
       在满足条件的组合中取 d 最小者。

用法：python experiments/R1_division/pilot.py --out experiments/R1_division/pilot
"""
import argparse, json
from pathlib import Path
import numpy as np

M = 4
A = 0.9 * np.block([[np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]]), np.zeros((2, 2))],
                    [np.zeros((2, 2)), np.array([[np.cos(1.2), -np.sin(1.2)], [np.sin(1.2), np.cos(1.2)]])]])
Q = np.eye(M) - A @ A.T          # 使 Σ_s = I 稳态
SEEDS = [900, 901, 902]
N = 4000
BITS = [2, 4, 6, 8]


def make_C(rng, d):
    G = rng.standard_normal((d, M))
    Qc, _ = np.linalg.qr(G)
    return Qc


def kalman_steady(C, R, iters=300):
    P = np.eye(M)
    for _ in range(iters):
        Pp = A @ P @ A.T + Q
        S = C @ Pp @ C.T + R * np.eye(C.shape[0])
        K = Pp @ C.T @ np.linalg.inv(S)
        P = (np.eye(M) - K @ C) @ Pp
    Pp = A @ P @ A.T + Q
    S = C @ Pp @ C.T + R * np.eye(C.shape[0])
    K = Pp @ C.T @ np.linalg.inv(S)
    return K, S


def simulate(rng, C, R, n):
    s = rng.standard_normal(M)
    xs, ss = [], []
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(M))
    for _ in range(n):
        s = A @ s + L @ rng.standard_normal(M)
        ss.append(s.copy())
        xs.append(C @ s + np.sqrt(R) * rng.standard_normal(C.shape[0]))
    return np.array(xs), np.array(ss)


def quantize(v, bits, scale):
    levels = 2 ** bits
    lo, hi = -scale, scale
    q = np.clip(v, lo, hi)
    idx = np.round((q - lo) / (hi - lo) * (levels - 1))
    return lo + idx * (hi - lo) / (levels - 1), np.mean(np.abs(v) > scale)


def run_one(d, R, k, seed):
    rng = np.random.default_rng(seed)
    C = make_C(rng, d)
    K, S = kalman_steady(C, R)
    X, _ = simulate(rng, C, R, N)
    # 稳态卡尔曼预测（线性递推）：x̂_{t|t-1} = C·ŝ_{t|t-1}
    sh = np.zeros(M); xpred = np.zeros((N, d)); innov = np.zeros((N, d))
    for t in range(N):
        xp = C @ sh
        xpred[t] = xp
        innov[t] = X[t] - xp
        sh = A @ (sh + K @ innov[t])   # x̂_{t+1|t} = A(x̂_{t|t-1} + K r_t)
    raw_scale = k * np.sqrt(np.mean(X ** 2))
    inn_scale = k * np.sqrt(np.mean(innov ** 2))
    out = {}
    for b in BITS:
        qX, cX = quantize(X, b, raw_scale)
        qI, cI = quantize(innov, b, inn_scale)
        mse_F = float(np.mean((X - qX) ** 2) / np.mean(X ** 2))
        mse_FS = float(np.mean((X - (xpred + qI)) ** 2) / np.mean(X ** 2))   # 重建 = 预测 + 量化创新
        out[b] = {"mse_F": mse_F, "mse_FS": mse_FS, "clip_raw": float(cX), "clip_inn": float(cI)}
    return out, float(np.mean(innov ** 2) / np.mean(X ** 2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res = {}
    # (i) 量化范围 k：在 d=16,R=0.1 下，b=8 的裁剪率
    for k in [2.0, 3.0, 4.0, 5.0]:
        r, _ = run_one(16, 0.1, k, SEEDS[0])
        res.setdefault("k_scan", {})[str(k)] = {"clip_raw_b8": r[8]["clip_raw"], "clip_inn_b8": r[8]["clip_inn"]}
    # (ii) (d, R) 网格，k 取 (i) 的结果（稍后确定；此处先用 4）
    grid = {}
    for d in [8, 16, 32]:
        for R in [0.05, 0.1, 0.5]:
            rhos = []
            for s in SEEDS:
                r, ratio_var = run_one(d, R, 4.0, s)
                rhos.append(r[4]["mse_FS"] / r[4]["mse_F"])
            grid[f"d={d},R={R}"] = {"rho_b4_mean": float(np.mean(rhos)), "rho_b4_sd": float(np.std(rhos)),
                                    "innov_over_raw_var": ratio_var}
            print(f"d={d} R={R} rho_b4={np.mean(rhos):.3f}±{np.std(rhos):.3f} innov/raw={ratio_var:.3f}", flush=True)
    res["grid"] = grid
    (out / "pilot.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["k_scan"], indent=1), flush=True)


if __name__ == "__main__":
    main()
