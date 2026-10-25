"""第 4 步预试验：K-C（观测干预）是否有区分力。仅用于校准/检查，种子 900–902。"""
import importlib.util, json
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
s = importlib.util.spec_from_file_location("pilot4", ROOT / "experiments/step4_knowledge/pilot.py"); Q4 = importlib.util.module_from_spec(s); s.loader.exec_module(Q4)
P, D, R0, NTR, NTE, SEEDS = Q4.P, Q4.D, Q4.R0, Q4.NTR, Q4.NTE, Q4.SEEDS
def run(seed):
    rng = np.random.default_rng(seed)
    C = P.make_C(rng, D); A = P.A
    Xtr, _ = Q4.gen(rng, C, R0, A, NTR); M = Q4.B.lms(Xtr, Q4.G0, Q4.TAU)
    Xte, _ = Q4.gen(rng, C, R0, A, NTE)
    Opred = np.vstack([np.zeros((1, D)), Xte[:-1] @ M.T])
    base = Q4.nmse(Opred, Xte)
    # do(x_{t-1})：把观测替换为独立边际抽样，真实 x_t 不变（潜在状态未改变）
    Xint = Xte.copy(); sd = np.sqrt(np.mean(Xte ** 2))
    Xint[:-1] = rng.standard_normal(Xte[:-1].shape) * sd
    Oint = np.vstack([np.zeros((1, D)), Xint[:-1] @ M.T])
    dO = Q4.nmse(Oint, Xte) - base
    # 结构预测器（忽略被干预的观测，使用未干预的滤波）：Δ_S = 0 by construction
    return dO
if __name__ == "__main__":
    d = [run(s) for s in SEEDS]
    print(json.dumps({"dO_mean": float(np.mean(d)), "dO_sd": float(np.std(d)), "per_seed": d}, indent=1))
