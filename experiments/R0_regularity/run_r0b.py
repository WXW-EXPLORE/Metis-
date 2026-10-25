"""R0b：数据规模 × 学习率调度（设计：docs/design_R0b_scale_schedule.md，预先登记）。

复用 run.py 的数据生成与度量函数；学习率调度：fixed（η=0.01）或 decay（η_t = 0.01/(1 + t/1000)）。
用法：python experiments/R0_regularity/run_r0b.py --n_train 50000 --sched decay --out experiments/R0_regularity/results_r0b/C4
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


R = _load("r0run", ROOT / "run.py")
ETA0, TAU = 0.01, 1000.0
CKPTS = [100, 500, 1000, 2000, 5000, 20000, 50000]
T4_TOL = 0.001   # 预先登记


def oja_sched(U, X, sched):
    snaps = {}
    eta_last = ETA0
    for i, x in enumerate(X, start=1):
        eta = ETA0 if sched == "fixed" else ETA0 / (1.0 + (i - 1) / TAU)
        eta_last = eta
        c = U.T @ x
        U = U + eta * np.outer(x, c)
        Q, Rm = np.linalg.qr(U)
        signs = np.sign(np.diag(Rm)); signs[signs == 0] = 1.0
        U = Q * signs
        if i in CKPTS:
            snaps[i] = U.copy()
    return U, snaps, eta_last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, required=True)
    ap.add_argument("--sched", choices=["fixed", "decay"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in R.SEEDS:
        rng = np.random.default_rng(1000 + seed)
        A = R.make_A(rng)
        U0 = R.make_A(np.random.default_rng(5000 + seed))
        X_tr = R.sample_regular(rng, A, args.n_train)
        X_te = R.sample_regular(rng, A, R.N_TEST)
        U_L, snaps, eta_final = oja_sched(U0.copy(), X_tr, args.sched)
        e_rand, e_learn, e_oracle = R.recon_err(U0, X_te), R.recon_err(U_L, X_te), R.recon_err(A, X_te)
        # T3：无关经历，学习率冻结在最终值（decay 条件）或保持 η0（fixed 条件）
        U_d = U_L.copy()
        X_drift = R.sample_null(rng, R.N_DRIFT)
        for x in X_drift:
            c = U_d.T @ x
            U_d = U_d + eta_final * np.outer(x, c)
            Q, Rm = np.linalg.qr(U_d)
            s = np.sign(np.diag(Rm)); s[s == 0] = 1.0
            U_d = Q * s
        # T2：无规律对照（同一调度）
        X_null_tr = R.sample_null(rng, args.n_train)
        X_null_te = R.sample_null(rng, R.N_TEST)
        U_nL, _, _ = oja_sched(R.make_A(np.random.default_rng(7000 + seed)), X_null_tr, args.sched)
        U_n0 = R.make_A(np.random.default_rng(7000 + seed))
        ang = {n: R.max_angle_cos(snaps[n], A) for n in snaps}
        rows.append({"seed": seed, "D_T1": e_rand - e_learn, "gap_oracle": e_rand - e_oracle,
                     "drift_angle": R.max_angle_cos(U_L, U_d),
                     "D_null": R.recon_err(U_n0, X_null_te) - R.recon_err(U_nL, X_null_te),
                     "angle_curve": ang})
        print(f"seed {seed}: D_T1={rows[-1]['D_T1']:.4f} angle@final={ang[max(ang)]:.4f} "
              f"drift={rows[-1]['drift_angle']:.4f} D_null={rows[-1]['D_null']:.5f}", flush=True)

    D_T1 = [r["D_T1"] for r in rows]; gap = [r["gap_oracle"] for r in rows]
    D_null = [r["D_null"] for r in rows]
    m1, lo1, hi1 = R.bci(D_T1)
    frac = m1 / float(np.mean(gap))
    mn, lon, hin = R.bci(D_null)
    drift = float(np.mean([r["drift_angle"] for r in rows]))
    curve = {n: float(np.mean([r["angle_curve"][n] for r in rows])) for n in CKPTS if n in rows[0]["angle_curve"]}
    cvals = [curve[n] for n in sorted(curve)]
    max_drop = max([cvals[i] - cvals[i + 1] for i in range(len(cvals) - 1)] + [0.0])
    T1 = bool(lo1 > 0 and frac >= 0.5)
    T2 = bool(lon <= 0 <= hin and abs(mn) < 0.25 * abs(m1))
    T3 = bool(drift >= 0.95)
    T4 = bool(max_drop <= T4_TOL and curve[max(curve)] >= 0.95)
    summary = {"n_train": args.n_train, "sched": args.sched,
               "T1": {"D_mean": m1, "ci95": [lo1, hi1], "frac_oracle": frac, "passed": T1},
               "T2": {"D_null": mn, "ci95": [lon, hin], "passed": T2},
               "T3": {"drift_angle_mean": drift, "passed": T3},
               "T4": {"curve_mean": curve, "max_drop": max_drop, "tol": T4_TOL, "passed": T4},
               "all_passed": bool(T1 and T2 and T3 and T4)}
    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=float))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
