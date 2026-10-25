"""E028 补充对照（PCA-512 投影与时间尺度分离的分解）。

在测试集上（λ 使用 result.json 中已锁定的值，不再调参），比较：
  S1       ：单库，1024 行，全键（主判据中的基线，直接读取 result.json）
  S1p      ：单库，512 行，PCA-512 键，λ=0.9（只含投影成本）
  T2same   ：两组各 512 行，PCA-512 键，竞争学习路由，两组 λ 都为 0.9（同样的结构，没有时间尺度分离）
  T2L      ：两组各 512 行，PCA-512 键，竞争学习路由，λ 为锁定值（主判据中的 T2L，直接读取）

分解：
  T2L − T2same ：时间尺度分离本身的效应（结构、维度、路由都相同）。
  T2same − S1p ：两组结构（分配到两组）相对于单组的效应（维度与投影相同）。
  S1p − S1     ：PCA-512 投影的成本。

注意：这些对照是在主判定之后补充的，属于探索性分析，不改变 G4 的判定结论。

用法（仓库根目录）：
    PYTHONPATH=.:src python experiments/E028_H11_timescale/eval_controls.py --keys /hy-tmp/e028_keys.pt --result experiments/E028_H11_timescale/results/result.json --out experiments/E028_H11_timescale/results
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    main_res = json.loads(Path(args.result).read_text())
    lf, ls = main_res["lambda_selected"]

    d = torch.load(args.keys, weights_only=False, mmap=True)
    fit = d["data"]["fit"]; test = d["data"]["test"]

    res = {"lambda_selected": [lf, ls], "arms": {}, "comparisons": {}}
    for li in E.LAYERS:
        Q, C_cl = E.fit_basis_and_protos(fit, li)
        cfg = {"S1p": {"arm": "S1p", "Q": Q},
               "T2same": {"arm": "T2L", "lams": [0.9, 0.9], "Q": Q, "centers": C_cl},
               "T2L": {"arm": "T2L", "lams": [lf, ls], "Q": Q, "centers": C_cl}}
        for name, c in cfg.items():
            r, _ = E.evaluate(test, li, c)
            res["arms"].setdefault(name, {})[str(li)] = r.tolist()
            print(name, "layer", li, round(float(np.nanmean(r)), 5), flush=True)
    # 主判据中的 S1 与 T2L 直接读取 result.json（同样的测试集、同样的键）
    for li in E.LAYERS:
        res["arms"].setdefault("S1", {})[str(li)] = main_res["test"]["S1"][str(li)]
        res["arms"].setdefault("T2L_main", {})[str(li)] = main_res["test"]["T2L"][str(li)]

    pairs = {
        "T2L_minus_T2same (timescale separation)": ("T2L", "T2same"),
        "T2same_minus_S1p (two groups vs one, same dims)": ("T2same", "S1p"),
        "S1p_minus_S1 (PCA-512 projection cost)": ("S1p", "S1"),
    }
    for label, (a, b) in pairs.items():
        res["comparisons"][label] = {}
        for li in E.LAYERS:
            x = np.array(res["arms"][a][str(li)]); y = np.array(res["arms"][b][str(li)])
            m, lo, hi = bci(x - y)
            res["comparisons"][label][str(li)] = {"mean": m, "ci95": [lo, hi]}
            print(label, "layer", li, round(m, 5), [round(lo, 5), round(hi, 5)], flush=True)
    (out / "controls.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
