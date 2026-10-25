"""R0c：同边际协方差、不同动态的预测规律检验（设计：docs/design_R0c_predictive.md，已确认）。

模型：s_{t+1} = A s_t + w_t，w_t ~ N(0, I − A Aᵀ)，x_t = s_t。边际 N(0, I)。
A_1 = 0.9·O（O 为两个平面旋转的块对角正交阵），A_2 = Pᵀ A_1 P（P 随机正交）：谱相同、特征向量不同。

学习器（线性闭式）：
  Σ̂0 = mean(x xᵀ)，Σ̂1 = mean(x_{t+1} x_tᵀ)，Â = Σ̂1 Σ̂0⁻¹。
  预测：x̂_{t+1} = Â x_t。NMSE = E‖x_{t+1} − x̂‖² / E‖x_{t+1}‖²。
对照：
  NULL：把训练样本的时间顺序打乱（x_a 与 x_b 独立配对），保留边际、去除动态。
  ORACLE：真实 A，理论 NMSE = ‖Q‖/d = 0.19。
  MARG：只用边际统计量的预测器（零预测），NMSE = 1。

判据（设计 §5，已确认阈值）：
  P1：A_1 与 A_2 的边际估计 Σ̂0 之差与抽样噪声同量级（同边际）。
  P2：NULL − L1 在 N=5000 的 95% 区间下界 > 0，对 A_1 与 A_2 均成立。
  P3：L1 的 NMSE 随 N 单调下降（均值），且 N=20000 时与 ORACLE 的相对差 < 5%。
  P4：ρ(A) = 0.9 < 1。

用法：python experiments/R0c_predictive/run.py --out experiments/R0c_predictive/results
"""
import argparse, json
from pathlib import Path
import numpy as np

D = 4
NS = [200, 1000, 5000, 20000]
N_TEST = 20000
BURN = 200
SEEDS = list(range(20))
ORACLE = 0.19   # = ‖I − AAᵀ‖ / d = 0.19 for A = 0.9·orthogonal


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def make_A1():
    O = np.zeros((D, D))
    O[:2, :2] = rot(0.5)
    O[2:, 2:] = rot(1.2)
    return 0.9 * O


def simulate(rng, A, n):
    Q = np.eye(D) - A @ A.T
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(D))
    s = rng.standard_normal(D)
    for _ in range(BURN):
        s = A @ s + L @ rng.standard_normal(D)
    X = np.empty((n + 1, D))
    X[0] = s
    for t in range(1, n + 1):
        X[t] = A @ X[t - 1] + L @ rng.standard_normal(D)
    return X


def fit(X):
    x0, x1 = X[:-1], X[1:]
    S0 = x0.T @ x0 / len(x0)
    S1 = x1.T @ x0 / len(x0)
    return S0, S1 @ np.linalg.inv(S0)


def nmse(Ahat, X):
    x0, x1 = X[:-1], X[1:]
    err = np.sum((x1 - x0 @ Ahat.T) ** 2, axis=1).mean()
    return float(err / np.sum(x1 ** 2, axis=1).mean())


def bci(x, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(5000)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    A1 = make_A1()
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        P, _ = np.linalg.qr(rng.standard_normal((D, D)))
        A2 = P.T @ A1 @ P
        r = {"seed": seed, "rho": {"A1": float(np.max(np.abs(np.linalg.eigvals(A1)))),
                                   "A2": float(np.max(np.abs(np.linalg.eigvals(A2))))}}
        for name, A in [("A1", A1), ("A2", A2)]:
            X_test = simulate(rng, A, N_TEST)
            Ssum = {}
            for N in NS:
                Xtr = simulate(rng, A, N)
                S0, Ahat = fit(Xtr)
                Ssum[N] = S0
                r[f"{name}_L1_{N}"] = nmse(Ahat, X_test)
                r[f"{name}_err_{N}"] = float(np.linalg.norm(Ahat - A) / np.linalg.norm(A))
                if N == 5000:
                    # NULL：同一训练样本，独立打乱配对（同边际，无动态）
                    idx = rng.permutation(len(Xtr)); jdx = rng.permutation(len(Xtr))
                    Xnull_pairs = np.stack([Xtr[idx, :], Xtr[jdx, :]], 1)
                    S1n = Xnull_pairs[:, 1, :].T @ Xnull_pairs[:, 0, :] / len(idx)
                    Anull = S1n @ np.linalg.inv(S0)
                    r[f"{name}_NULL_5000"] = nmse(Anull, X_test)
            r[f"{name}_oracle"] = nmse(A, X_test)
            r[f"{name}_S0_5000"] = Ssum[5000].tolist()
        rows.append(r)
        print(f"seed {seed}: A1 L1@20000={r['A1_L1_20000']:.4f} NULL={r['A1_NULL_5000']:.4f} "
              f"oracle={r['A1_oracle']:.4f}", flush=True)

    # P1：同边际。A1 与 A2 的 Σ̂0 差的范数 vs 同一 A 下两次独立估计的差的范数
    S_A1 = np.array([r["A1_S0_5000"] for r in rows]); S_A2 = np.array([r["A2_S0_5000"] for r in rows])
    d_cross = np.linalg.norm(S_A1 - S_A2, axis=(1, 2)).mean()
    d_within = np.linalg.norm(S_A1 - S_A1[np.roll(np.arange(len(rows)), 1)], axis=(1, 2)).mean()
    P1 = bool(d_cross < 1.5 * d_within)
    # P2：NULL − L1 at 5000
    g = {name: bci([r[f"{name}_NULL_5000"] - r[f"{name}_L1_5000"] for r in rows]) for name in ("A1", "A2")}
    P2 = bool(all(g[n][1] > 0 for n in g))
    # P3：均值单调下降，且 20000 与 oracle 相对差 < 5%
    means = {n: float(np.mean([r[f"A1_L1_{n}"] for r in rows])) for n in NS}
    vals = [means[n] for n in NS]
    mono = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    orc = float(np.mean([r["A1_oracle"] for r in rows]))
    rel = abs(means[20000] - orc) / orc
    rel2 = abs(float(np.mean([r["A2_L1_20000"] for r in rows])) - orc) / orc
    P3 = bool(mono and rel < 0.05 and rel2 < 0.05)
    P4 = bool(all(r["rho"]["A1"] < 1 and r["rho"]["A2"] < 1 for r in rows))
    summary = {"P1_same_marginal": {"d_cross": float(d_cross), "d_within": float(d_within), "passed": P1},
               "P2_gain_vs_null_5000": {k: {"mean": v[0], "ci95": [v[1], v[2]]} for k, v in g.items()},
               "P2_passed": P2,
               "P3_L1_nmse_mean": means, "P3_oracle": orc, "P3_rel_diff_A1": rel, "P3_rel_diff_A2": rel2,
               "P3_monotone": bool(mono), "P3_passed": P3,
               "P4_passed": P4,
               "R0c_passed": bool(P1 and P2 and P3 and P4), "n_seeds": len(SEEDS)}
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
