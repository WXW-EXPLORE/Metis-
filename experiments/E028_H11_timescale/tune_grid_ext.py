"""H11b 方案 A：只在调参集上扩展 λ 网格，检查最优是否仍在边界（不接触测试集，不作为 G4 判定）。
设计：docs/design_H11b_lambda_boundary.md §3。

网格：λ_f ∈ {0.05, 0.1, 0.2, 0.3}，λ_s ∈ {0.8, 0.9, 0.95}（12 组）。
用法（仓库根目录）：
    PYTHONPATH=.:src python experiments/E028_H11_timescale/tune_grid_ext.py --keys /hy-tmp/e028_keys.pt --out experiments/E028_H11_timescale/results_tune_ext
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


E = _load("e28eval", "experiments/E028_H11_timescale/eval.py")


def bci(x, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(5000)])
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--keys", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    d = torch.load(args.keys, weights_only=False, mmap=True)
    fit, tune = d["data"]["fit"], d["data"]["tune"]
    grid = [(lf, ls) for lf in (0.05, 0.1, 0.2, 0.3) for ls in (0.8, 0.9, 0.95)]
    cfg = {li: E.fit_basis_and_protos(fit, li) for li in E.LAYERS}
    res = {}
    for li in E.LAYERS:
        Q, C_cl = cfg[li]
        s1, _ = E.evaluate(tune, li, {"arm": "S1"})
        for lf, ls in grid:
            t2, _ = E.evaluate(tune, li, {"arm": "T2L", "lams": [lf, ls], "Q": Q, "centers": C_cl})
            m, lo, hi = bci(t2 - s1)
            res.setdefault(f"{lf},{ls}", {})[str(li)] = {"mean": m, "ci95": [lo, hi]}
            print("tune layer", li, lf, ls, round(m, 5), flush=True)
    score = {k: np.mean([v[str(li)]["mean"] for li in E.LAYERS]) for k, v in res.items()}
    best = max(score, key=score.get)
    on_edge = best.split(",")[0] == "0.05" or best.split(",")[1] == "0.8"   # 扩展方向为向下，只检查下端点
    summary = {"best": best, "best_mean": float(score[best]), "best_on_grid_edge": bool(on_edge),
               "note": "只在调参集上计算；不接触测试集；不作为 G4 判定（设计 H11b 方案 A）",
               "grid": res}
    (out / "tune_grid_ext.json").write_text(json.dumps(summary, indent=1))
    print("best", best, round(float(score[best]), 5), "on_edge", on_edge, flush=True)


if __name__ == "__main__":
    main()
