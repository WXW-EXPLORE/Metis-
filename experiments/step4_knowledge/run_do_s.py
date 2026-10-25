"""第 4 步修正：do(s) 报告（只报告，不判定）。
干预：每个时刻把潜在状态 s_{t-1} 直接设为新值 s'，观测 x_{t-1}' = C s' + v，真实目标 x_t' 由 s' 按动态与相同噪声重新生成。
比较：观测预测器 O（M x_{t-1}'）与真实因果预测器（已知 s'，预测 C A s'）相对于各自未干预基线的增量。"""
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
    base_O = Q4.nmse(np.vstack([np.zeros((1, D)), Xte[:-1] @ M.T]), Xte)
    base_or = Q4.nmse(np.vstack([np.zeros((1, D)), Ste[:-1] @ A.T @ C.T]), Xte)   # 因果预测器（已知真实 s_{t-1}）
    # 干预序列
    T = NTE
    s_prev = rng.standard_normal((T, 4))            # s'_{t-1}（每个 t 独立干预）
    Lq = np.linalg.cholesky(np.eye(4) - A @ A.T + 1e-12 * np.eye(4))
    w = rng.standard_normal((T, 4)) @ Lq.T              # 状态噪声 w ~ N(0, Q)
    v_t = np.sqrt(R0) * rng.standard_normal((T, D)); v_prev = np.sqrt(R0) * rng.standard_normal((T, D))
    x_prev = s_prev @ C.T + v_prev                  # 干预后的观测 x_{t-1}'
    s_next = s_prev @ A.T + w                       # s_t = A s'_{t-1} + w
    x_next = s_next @ C.T + v_t                     # 真实目标 x_t'
    ser = np.zeros((T, 1))
    O_pred = x_prev @ M.T
    dO = float(np.mean(np.sum((x_next - O_pred) ** 2, 1)) / np.mean(np.sum(x_next ** 2, 1)))
    oracle_pred = (A @ s_prev.T).T @ C.T            # 因果预测器（已知干预后的状态）
    dor = float(np.mean(np.sum((x_next - oracle_pred) ** 2, 1)) / np.mean(np.sum(x_next ** 2, 1)))
    return {"O_do_s": dO, "O_base": base_O, "oracle_do_s": dor, "oracle_base": base_or}
if __name__ == "__main__":
    rows = [run(s) for s in SEEDS]
    dO = [r["O_do_s"] - r["O_base"] for r in rows]; dR = [r["oracle_do_s"] - r["oracle_base"] for r in rows]
    summ = {"note": "报告，不判定（修正后的 do(s)）", "delta_O": bci(dO), "delta_oracle": bci(dR),
            "O_do_s_mean": float(np.mean([r["O_do_s"] for r in rows])), "oracle_do_s_mean": float(np.mean([r["oracle_do_s"] for r in rows]))}
    out = ROOT / "experiments/step4_knowledge/results_do_s"; out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1), flush=True)
