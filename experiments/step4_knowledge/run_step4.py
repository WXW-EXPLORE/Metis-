"""第 4 步正式实验（docs/design_step4_knowledge.md §9，预先登记）：种子 0–19。"""
import importlib.util, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
s = importlib.util.spec_from_file_location("pilot4", ROOT / "experiments/step4_knowledge/pilot.py"); Q4 = importlib.util.module_from_spec(s); s.loader.exec_module(Q4)
P, D, R0, NTR, NTE, A = Q4.P, Q4.D, Q4.R0, Q4.NTR, Q4.NTE, Q4.P.A
SEEDS = list(range(20))
def bci(x, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = np.array([r.choice(x, size=x.size, replace=True).mean() for _ in range(5000)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
def run(seed):
    rng = np.random.default_rng(seed)
    C = P.make_C(rng, D)
    Xtr, _ = Q4.gen(rng, C, R0, A, NTR); M = Q4.B.lms(Xtr, Q4.G0, Q4.TAU)
    Xte, Ste = Q4.gen(rng, C, R0, A, NTE)
    O = np.vstack([np.zeros((1, D)), Xte[:-1] @ M.T]); S = Q4.kalman_pred(C, R0, A, Xte)
    rec = {"K_A_O": Q4.nmse(O, Xte), "K_A_S": Q4.nmse(S, Xte)}
    rec["K_A_gap"] = rec["K_A_O"] / rec["K_A_S"] - 1
    # K-B：旧模型，噪声 ×2
    Xn, _ = Q4.gen(rng, C, 2 * R0, A, NTE)
    dS = Q4.nmse(Q4.kalman_pred(C, R0, A, Xn), Xn) - rec["K_A_S"]
    dO = Q4.nmse(np.vstack([np.zeros((1, D)), Xn[:-1] @ M.T]), Xn) - rec["K_A_O"]
    rec["K_B_diff"] = dO - dS
    # K-C：do(x_{t-1})，潜在状态不变；Δ_S 由结构预测器（忽略被干预观测）给出，定义为 0
    Xint = Xte.copy(); sd = np.sqrt(np.mean(Xte ** 2)); Xint[:-1] = rng.standard_normal(Xte[:-1].shape) * sd
    dO_int = Q4.nmse(np.vstack([np.zeros((1, D)), Xint[:-1] @ M.T]), Xte) - rec["K_A_O"]
    rec["K_C_dO"] = dO_int; rec["K_C_dS"] = 0.0
    # do(s)：仅报告
    Sint = Ste.copy(); Sint[:-1] = rng.standard_normal(Ste[:-1].shape)
    Xs = np.vstack([Xte[:1], (Sint @ C.T)[:-1] + np.sqrt(R0) * rng.standard_normal((NTE - 1, D))])
    rec["do_s_report_dO"] = Q4.nmse(np.vstack([np.zeros((1, D)), Xs[:-1] @ M.T]), Xte) - rec["K_A_O"]
    return rec
if __name__ == "__main__":
    rows = [run(s) for s in SEEDS]
    summ = {"K_A": {"gap_mean": float(np.mean([r["K_A_gap"] for r in rows])), "threshold": 0.118,
                    "passed": bool(np.mean([r["K_A_gap"] for r in rows]) <= 0.118)},
            "K_B": {"diff": bci([r["K_B_diff"] for r in rows]), "threshold": 0.02,
                    "passed": bool(bci([r["K_B_diff"] for r in rows])[0] <= 0.02)},
            "K_C": {"dO_minus_dS": bci([r["K_C_dO"] - r["K_C_dS"] for r in rows])},
            "do_s_report": bci([r["do_s_report_dO"] for r in rows]), "n_seeds": len(SEEDS)}
    summ["K_C"]["passed"] = bool(summ["K_C"]["dO_minus_dS"][1] > 0)
    summ["all_passed"] = bool(summ["K_A"]["passed"] and summ["K_B"]["passed"] and summ["K_C"]["passed"])
    out = ROOT / "experiments/step4_knowledge/results"; out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); (out / "rows.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(summ, indent=1), flush=True)
