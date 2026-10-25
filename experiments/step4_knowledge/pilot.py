"""第 4 步预试验（种子 900–902，与正式实验分开）：校准 K-A 阈值、K-B 扰动与倍数。docs/design_step4_knowledge.md §8。"""
import importlib.util, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
P = _load("p1", "experiments/R1_division/pilot.py")
B = _load("b1", "experiments/R1_division/pilot_r1b.py")
D, R0, G0, TAU, NTR, NTE = 8, 0.5, 0.1, 100.0, 20000, 4000
SEEDS = [900, 901, 902]

def gen(rng, C, R, A, n):
    Q = np.eye(4) - A @ A.T; L = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
    s = rng.standard_normal(4); X = np.zeros((n, D)); S = np.zeros((n, 4))
    for t in range(n):
        s = A @ s + L @ rng.standard_normal(4); S[t] = s
        X[t] = C @ s + np.sqrt(R) * rng.standard_normal(D)
    return X, S

def kalman_pred(C, R, A, X):
    Kg, _ = P.kalman_steady(C, R)
    sh = np.zeros(4); pred = np.zeros_like(X)
    for t in range(len(X)):
        pred[t] = C @ sh; innov = X[t] - pred[t]; sh = A @ (sh + Kg @ innov)
    return pred

def nmse(pred, X):
    return float(np.mean((X[1:] - pred[1:]) ** 2) / np.mean(X[1:] ** 2))

def run(seed):
    rng = np.random.default_rng(seed)
    C = P.make_C(rng, D); A = P.A
    Xtr, _ = gen(rng, C, R0, A, NTR)
    M = B.lms(Xtr, G0, TAU)                                  # 观测预测器（学习）
    Xte, _ = gen(rng, C, R0, A, NTE)
    res = {}
    Spred = kalman_pred(C, R0, A, Xte)
    Opred = np.vstack([np.zeros((1, D)), Xte[:-1] @ M.T])
    res["K_A"] = {"O": nmse(Opred, Xte), "S": nmse(Spred, Xte), "bad": nmse(Xte[:-1] @ (0.5 * M).T, Xte[1:]) if False else None}
    Bad = np.vstack([np.zeros((1, D)), Xte[:-1] @ (0.5 * M).T])
    res["K_A"]["bad"] = nmse(Bad, Xte)
    # K-B（修正）：旧模型（M、A、R0）固定不重新适应，只改变数据；比较各预测器的 NMSE 增量
    base_S = res["K_A"]["S"]; base_O = res["K_A"]["O"]
    Xa, _ = gen(rng, C, 2 * R0, A, NTE)                      # 扰动 (a)：观测噪声 ×2
    res["pert_noise"] = {"dS": nmse(kalman_pred(C, R0, A, Xa), Xa) - base_S,
                         "dO": nmse(np.vstack([np.zeros((1, D)), Xa[:-1] @ M.T]), Xa) - base_O}
    Xb, _ = gen(rng, C, R0, 0.95 * A, NTE)                   # 扰动 (b)：结构正对照（转移 ×0.95）
    res["pert_struct"] = {"dS": nmse(kalman_pred(C, R0, A, Xb), Xb) - base_S,
                          "dO": nmse(np.vstack([np.zeros((1, D)), Xb[:-1] @ M.T]), Xb) - base_O}
    return res

if __name__ == "__main__":
    rows = [run(s) for s in SEEDS]
    summ = {}
    summ["K_A_O"] = float(np.mean([r["K_A"]["O"] for r in rows]))
    summ["K_A_S"] = float(np.mean([r["K_A"]["S"] for r in rows]))
    summ["K_A_bad"] = float(np.mean([r["K_A"]["bad"] for r in rows]))
    summ["K_A_threshold_geo_mid"] = float(np.sqrt(summ["K_A_O"] * summ["K_A_bad"]))
    for k in ["pert_noise", "pert_struct"]:
        summ[k] = {"dS_mean": float(np.mean([r[k]["dS"] for r in rows])), "dO_mean": float(np.mean([r[k]["dO"] for r in rows]))}
    out = ROOT / "experiments/step4_knowledge/pilot.json"; out.write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)
