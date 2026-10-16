"""运行 Phase 0 合成诊断，输出原始结果与汇总。

用法（在仓库根目录）：
    PYTHONPATH=src python -m diagnostics.run_diagnostics \
        --config experiments/E001_diagnostics/config.json \
        --out experiments/E001_diagnostics
"""
import argparse
import csv
import json
import os

import numpy as np

from diagnostics.synthetic_memory import (
    crosstalk_probe,
    decay_probe,
    deletion_probe,
    make_target_and_distractors,
)


def run_decay(cfg, seeds, rows):
    for lam in cfg["lam"]:
        for age in cfg["ages"]:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                k_c, v_c, _, _ = make_target_and_distractors(rng, cfg["d"], cfg["dv"], 0, 0.0)
                rows.append({
                    "experiment": "decay", "lam": lam, "age": age, "seed": seed,
                    "rho": "", "n": 0,
                    "amplitude": decay_probe(k_c, v_c, lam, age),
                })


def run_crosstalk(cfg, seeds, rows):
    for lam in cfg["lam"]:
        for n in cfg["n"]:
            for rho in cfg["rho"]:
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    out = crosstalk_probe(rng, cfg["d"], cfg["dv"], lam, cfg["age"], n, rho)
                    rows.append({
                        "experiment": "crosstalk", "lam": lam, "age": cfg["age"],
                        "seed": seed, "rho": rho, "n": n, **out,
                    })


def run_deletion(cfg, seeds, rows):
    for lam in cfg["lam"]:
        for method in cfg["methods"]:
            for n in cfg["n"]:
                for rho in cfg["rho"]:
                    for seed in seeds:
                        rng = np.random.default_rng(seed)
                        gamma = method["gamma"] if method["gamma"] is not None else 0.0
                        out = deletion_probe(rng, cfg["d"], cfg["dv"], lam, cfg["age"], n, rho,
                                             method["name"], gamma=gamma)
                        label = method["name"] if method["name"] == "projection" else f"global_g{gamma}"
                        rows.append({
                            "experiment": "deletion", "method": label, "lam": lam,
                            "age": cfg["age"], "seed": seed, "rho": rho, "n": n, **out,
                        })


def decay_tau_rows(rows, cfg):
    """对每个 lam，用 log 振幅对年龄做线性拟合，得到 tau_eff，并与设计值比较。"""
    out = []
    for lam in cfg["decay"]["lam"]:
        taus = []
        for seed in cfg["seeds"]:
            sel = [r for r in rows if r["experiment"] == "decay"
                   and r["lam"] == lam and r["seed"] == seed]
            ages = np.array([r["age"] for r in sel], dtype=float)
            amps = np.array([r["amplitude"] for r in sel], dtype=float)
            slope = np.polyfit(ages, np.log(np.abs(amps)), 1)[0]
            taus.append(-1.0 / slope)
        design = -1.0 / np.log(lam)
        out.append({"lam": lam, "tau_eff_mean": float(np.mean(taus)),
                    "tau_eff_std": float(np.std(taus)), "tau_design": design})
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, group_keys, value_keys):
    """按 group_keys 分组，对 value_keys 求均值与标准差（跨种子）。"""
    groups = {}
    for r in rows:
        g = tuple(r.get(k, "") for k in group_keys)
        groups.setdefault(g, []).append(r)
    out = []
    for g, rs in groups.items():
        row = dict(zip(group_keys, g))
        for v in value_keys:
            vals = np.array([float(r[v]) for r in rs])
            row[f"{v}_mean"] = float(np.mean(vals))
            row[f"{v}_std"] = float(np.std(vals))
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    seeds = cfg["seeds"]

    rows = []
    run_decay({**cfg["decay"], "d": cfg["d"], "dv": cfg["dv"]}, seeds, rows)
    run_crosstalk({**cfg["crosstalk"], "d": cfg["d"], "dv": cfg["dv"]}, seeds, rows)
    run_deletion({**cfg["deletion"], "d": cfg["d"], "dv": cfg["dv"]}, seeds, rows)
    write_csv(os.path.join(args.out, "raw.csv"), rows)

    tau_rows = decay_tau_rows(rows, cfg)
    write_csv(os.path.join(args.out, "decay_tau.csv"), tau_rows)

    xt = [r for r in rows if r["experiment"] == "crosstalk"]
    write_csv(os.path.join(args.out, "crosstalk_summary.csv"),
              summarize(xt, ["lam", "n", "rho"],
                        ["snr", "target_norm", "crosstalk_norm", "crosstalk_over_fresh"]))

    dl = [r for r in rows if r["experiment"] == "deletion"]
    write_csv(os.path.join(args.out, "deletion_summary.csv"),
              summarize(dl, ["method", "lam", "n", "rho"], ["target_residual", "collateral"]))

    print(f"wrote {len(rows)} raw rows to {args.out}")


if __name__ == "__main__":
    main()
