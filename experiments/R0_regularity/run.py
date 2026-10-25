"""R0：合成数据上的规律子空间形成（设计：docs/design_R0_regularity.md）。纯数值模拟，不涉及 Metis 模型。

慢速规则：Oja 子空间规则 + QR 回缩。U ← qr(U + η x cᵀ)，c = Uᵀx。
判据 T1–T4 在运行前登记（见设计文档 §5）。

用法：python experiments/R0_regularity/run.py --out experiments/R0_regularity/results
"""
import argparse, json
from pathlib import Path
import numpy as np

D, M = 64, 4
SIG = np.array([3.0, 2.0, 1.5, 1.0])
NOISE = 0.3
ETA = 0.01
N_TRAIN, N_TEST, N_DRIFT = 5000, 2000, 5000
CKPTS = [100, 500, 1000, 2000, 5000]
SEEDS = list(range(20))


def make_A(rng):
    G = rng.standard_normal((D, M))
    Q, _ = np.linalg.qr(G)
    return Q


def sample_regular(rng, A, n):
    z = rng.standard_normal((n, M)) * SIG
    return z @ A.T + NOISE * rng.standard_normal((n, D))


def sample_null(rng, n):
    return rng.standard_normal((n, D))


def oja(U, X):
    """对样本 X（按顺序）做 Oja 子空间更新；返回最终 U 与在若干检查点的 U 列表。"""
    snaps = {}
    for i, x in enumerate(X, start=1):
        c = U.T @ x
        U = U + ETA * np.outer(x, c)
        Q, R = np.linalg.qr(U)
        signs = np.sign(np.diag(R)); signs[signs == 0] = 1.0
        U = Q * signs
        if i in CKPTS:
            snaps[i] = U.copy()
    return U, snaps


def recon_err(U, X):
    P = X @ U
    return float(np.mean(np.sum(X * X, 1) - np.sum(P * P, 1)) / np.mean(np.sum(X * X, 1)))


def max_angle_cos(U, V):
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return float(s.min())


def bci(x, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(5000)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        A = make_A(rng)
        U0 = make_A(np.random.default_rng(5000 + seed))   # 随机初始化（也作为"随机 U"基线）
        X_tr = sample_regular(rng, A, N_TRAIN)
        X_te = sample_regular(rng, A, N_TEST)
        U_L, snaps = oja(U0.copy(), X_tr)
        # T1
        e_rand = recon_err(U0, X_te)
        e_learn = recon_err(U_L, X_te)
        e_oracle = recon_err(A, X_te)
        # T3 漂移：在学习后的 U 上继续输入无规律经历
        U_drift, _ = oja(U_L.copy(), sample_null(rng, N_DRIFT))
        # T4 学习曲线
        ang = {n: max_angle_cos(snaps[n], A) for n in CKPTS}
        # T2 无规律对照
        X_null_tr = sample_null(rng, N_TRAIN)
        X_null_te = sample_null(rng, N_TEST)
        U_nL, _ = oja(make_A(np.random.default_rng(7000 + seed)), X_null_tr)
        U_n0 = make_A(np.random.default_rng(7000 + seed))
        rows.append({"seed": seed, "err_random": e_rand, "err_learned": e_learn, "err_oracle": e_oracle,
                     "D_T1": e_rand - e_learn, "gap_oracle": e_rand - e_oracle,
                     "drift_angle": max_angle_cos(U_L, U_drift),
                     "angle_curve": ang,
                     "D_null": recon_err(U_n0, X_null_te) - recon_err(U_nL, X_null_te)})
        print(f"seed {seed}: D_T1={rows[-1]['D_T1']:.4f} oracle_gap={rows[-1]['gap_oracle']:.4f} "
              f"angle@5000={ang[5000]:.4f} drift={rows[-1]['drift_angle']:.4f} D_null={rows[-1]['D_null']:.5f}",
              flush=True)

    D_T1 = [r["D_T1"] for r in rows]
    gap = [r["gap_oracle"] for r in rows]
    D_null = [r["D_null"] for r in rows]
    m1, lo1, hi1 = bci(D_T1)
    mg = float(np.mean(gap))
    frac_of_oracle = m1 / mg
    mn, lon, hin = bci(D_null)
    drift_mean = float(np.mean([r["drift_angle"] for r in rows]))
    curve = {n: float(np.mean([r["angle_curve"][n] for r in rows])) for n in CKPTS}
    curve_vals = [curve[n] for n in CKPTS]
    T1 = bool(lo1 > 0 and frac_of_oracle >= 0.5)
    T2 = bool(lon <= 0 <= hin and abs(mn) < 0.25 * abs(m1))
    T3 = bool(drift_mean >= 0.95)
    T4 = bool(all(curve_vals[i] <= curve_vals[i + 1] + 1e-9 for i in range(len(curve_vals) - 1))
              and curve[5000] >= 0.95)
    summary = {
        "T1": {"D_mean": m1, "ci95": [lo1, hi1], "fraction_of_oracle_gap": frac_of_oracle, "passed": T1},
        "T2": {"D_null_mean": mn, "ci95": [lon, hin], "passed": T2},
        "T3": {"drift_angle_mean": drift_mean, "passed": T3},
        "T4": {"angle_curve_mean": curve, "passed": T4},
        "R0_passed": bool(T1 and T2 and T3 and T4),
        "n_seeds": len(SEEDS),
        "params": {"D": D, "M": M, "sigma": SIG.tolist(), "noise": NOISE, "eta": ETA,
                   "N_train": N_TRAIN, "N_test": N_TEST, "N_drift": N_DRIFT},
    }
    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=float))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
