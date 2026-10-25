"""R1 正式实验（oracle 参数版）：等比特预算下，预测 + 量化创新（FS）是否优于直接量化原始观测（F）。

参数（来自预试验 pilot.py，种子 900–902，见 docs/design_R1_division.md §9）：d=8，R=0.5，量化范围 ±3·RMS。
本步使用真实 A、C、稳态卡尔曼增益 K（oracle）。慢层的在线参数学习留作第二步（R1b），本文件不包含。
判据（设计 §5，R1-a/b/c 本步可检验；R1-d 需在线学习，本步不判定）。
用法：python experiments/R1_division/run.py --out experiments/R1_division/results
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot", ROOT / "pilot.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
D, R_NOISE, K_RANGE = 8, 0.5, 3.0
SEEDS = list(range(20))
N = 4000


def run_seed(seed):
    rng = np.random.default_rng(seed)
    C = P.make_C(rng, D)
    K, S = P.kalman_steady(C, R_NOISE)
    X, _ = P.simulate(rng, C, R_NOISE, N)
    sh = np.zeros(P.M); xpred = np.zeros((N, D)); innov = np.zeros((N, D))
    for t in range(N):
        xp = C @ sh
        xpred[t] = xp
        innov[t] = X[t] - xp
        sh = P.A @ (sh + K @ innov[t])
    raw_scale = K_RANGE * np.sqrt(np.mean(X ** 2))
    inn_scale = K_RANGE * np.sqrt(np.mean(innov ** 2))
    out = {"seed": seed}
    denom = np.mean(X ** 2)
    out["pred_nmse"] = float(np.mean((X - xpred) ** 2) / denom)          # 慢层预测（与比特无关）
    out["zero_nmse"] = 1.0                                                # x̂ = 0 基线
    for b in P.BITS:
        qX, _ = P.quantize(X, b, raw_scale)
        qI, _ = P.quantize(innov, b, inn_scale)
        out[f"F_{b}"] = float(np.mean((X - qX) ** 2) / denom)
        out[f"FS_{b}"] = float(np.mean((X - (xpred + qI)) ** 2) / denom)
    return out


def bci(x, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(5000)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = [run_seed(s) for s in SEEDS]
    summ = {"params": {"d": D, "R": R_NOISE, "k": K_RANGE, "N": N, "bits": P.BITS, "n_seeds": len(SEEDS)},
            "per_bits": {}}
    ratio_ok, gain_ok = True, True
    for b in P.BITS:
        diff = [r[f"F_{b}"] - r[f"FS_{b}"] for r in rows]     # >0 表示 FS 更好
        ratio = float(np.mean([r[f"FS_{b}"] / r[f"F_{b}"] for r in rows]))
        m, lo, hi = bci(diff)
        summ["per_bits"][str(b)] = {"F": float(np.mean([r[f"F_{b}"] for r in rows])),
                                    "FS": float(np.mean([r[f"FS_{b}"] for r in rows])),
                                    "diff_F_minus_FS": {"mean": m, "ci95": [lo, hi]}, "ratio_FS_over_F": ratio}
        gain_ok &= lo > 0
        ratio_ok &= ratio < 1
    pm = float(np.mean([r["pred_nmse"] for r in rows]))
    summ["R1_a_passed"] = bool(gain_ok)
    summ["R1_b_passed"] = bool(ratio_ok)
    summ["R1_c_passed"] = bool(pm < 1.0)
    summ["R1_c_pred_nmse"] = pm
    summ["R1_d"] = "not evaluated in this step (requires online learning of θ)"
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
