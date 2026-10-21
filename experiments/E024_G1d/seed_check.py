"""E024 种子核实（只读检查，不改判据）。

问题：E024 中 K4_CL 与 K4_CL_cons 在 3 个种子间的最大库占比逐位相同。
检查：(1) 不同种子的初始化原型是否不同；(2) 训练后的原型是否不同；
      (3) 每个种子的完整负载向量（不取平均、不取最大值）；(4) 把初始化与学习顺序分开看。
同时把 E023/E024 所用的 train_prototypes 与一个独立实现对照，排除实现错误。

用法（仓库根目录）：PYTHONPATH=.:src python experiments/E024_G1d/seed_check.py --out experiments/E024_G1d/seed_check
"""
import argparse, importlib.util, json, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri
import holistic.rls_metis as R
from train.train_utils import freeze_backbone
_s18 = importlib.util.spec_from_file_location("e18", ROOT / "experiments/E018_G1_equal_state/run.py")
E18 = importlib.util.module_from_spec(_s18); _s18.loader.exec_module(E18)
_s23 = importlib.util.spec_from_file_location("e23", ROOT / "experiments/E023_G1c/run.py")
E23 = importlib.util.module_from_spec(_s23); _s23.loader.exec_module(E23)
LOCOMO = ROOT / "eval/data/memqa/locomo_evidence_sessions_tps16_20260626.jsonl"
K = 4


def load_locomo():
    groups = OrderedDict()
    with open(LOCOMO, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            key = row["instance_id"].split("__qa_")[0]
            if key not in groups:
                groups[key] = {"episode_id": key, "conv": row["source_sample_id"],
                               "memory_chunks": [st["content"] for st in row["memory_steps"]]}
    return list(groups.values())


def independent_cl(keys, beta, seed, eta=0.05, rho=0.01):
    """独立实现（与 E023 的 train_prototypes 逻辑相同，但单独编写，用于交叉核对）。"""
    gen = torch.Generator().manual_seed(seed)
    C = torch.nn.functional.normalize(torch.randn(K, keys.shape[1], generator=gen), dim=-1)
    C0 = C.clone()
    u = torch.full((K,), 1.0 / K)
    for k in keys:
        score = (C @ k) - beta * (u - 1.0 / K)
        b = int(torch.argmax(score))
        newc = C[b] + eta * (k - C[b])
        C[b] = newc / newc.norm()
        oh = torch.zeros(K); oh[b] = 1.0
        u = (1 - rho) * u + rho * oh
    return C0, C, u


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    eps = load_locomo()
    convs = list(OrderedDict.fromkeys(e["conv"] for e in eps))
    fit = [e for e in eps if e["conv"] in set(convs[:5])]
    test = [e for e in eps if e["conv"] in set(convs[5:])][:30]
    load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load); model.eval(); freeze_backbone(model)
    R.install_rls(model, lam=0.9)
    fit_d = E18.record(model, tok, dev, fit, [3]); test_d = E18.record(model, tok, dev, test, [3])
    R.uninstall_rls(model)
    seqs_f, seqs_t = fit_d[3], test_d[3]
    Xf = torch.cat([F.normalize(k, dim=-1) for seq in seqs_f for (k, _, _) in seq], 0)
    Xt = torch.cat([F.normalize(k, dim=-1) for seq in seqs_t for (k, _, _) in seq], 0)
    print("fit keys", Xf.shape, "test keys", Xt.shape, flush=True)

    report = {"n_fit_keys": int(Xf.shape[0]), "n_test_keys": int(Xt.shape[0]), "runs": []}
    seeds = [2000, 2001, 2002]
    for beta in [0.0, 1.0]:
        finals, inits, loads = [], [], []
        for s in seeds:
            C, u = E23.train_prototypes(Xf, beta, seed=s)
            C0, C_ind, u_ind = independent_cl(Xf, beta, s)
            agree = float((C - C_ind).abs().max())   # 与独立实现的一致性
            b = E23.route_scores(Xt, C, u, beta)
            load = torch.bincount(b, minlength=K).float().div(b.numel()).tolist()
            finals.append(C); inits.append(C0); loads.append(load)
            report["runs"].append({"beta": beta, "seed": s, "load_vector": load,
                                   "max_share": max(load), "impl_agreement_maxabs": agree,
                                   "usage_final": u.tolist()})
            print(f"beta={beta} seed={s} load={[round(x,3) for x in load]} impl_diff={agree:.2e}", flush=True)
        init_gap = [float((inits[i] - inits[j]).abs().max()) for i in range(3) for j in range(i + 1, 3)]
        final_gap = [float((finals[i] - finals[j]).abs().max()) for i in range(3) for j in range(i + 1, 3)]
        report["runs"].append({"beta": beta, "summary": {"init_pairwise_maxabs_diff": init_gap,
                                                         "final_pairwise_maxabs_diff": final_gap,
                                                         "load_vectors_identical": bool(all(
                                                             np.allclose(loads[0], l) for l in loads[1:]))}})
        print(f"beta={beta} init_gap={[round(x,4) for x in init_gap]} final_gap={[round(x,4) for x in final_gap]}", flush=True)

    # 对照：完全随机的分配（不依赖原型）是否按种子变化
    _, _, Vh = torch.linalg.svd(Xf, full_matrices=False)
    Q = Vh[:256].T
    rnd = []
    for s in seeds:
        g = torch.Generator().manual_seed(1000 + s)
        Cr = F.normalize(torch.randn(K, 256, generator=g) @ Q.T, dim=-1)
        rnd.append(torch.bincount((Xt @ Cr.T).argmax(-1), minlength=K).float().div(Xt.shape[0]).tolist())
    report["random_centers_loads"] = rnd
    # 排序后的负载"形状"（去掉库编号）：用于判断种子之间是否只是库的排列不同
    report["sorted_load_profiles"] = {f"beta={r['beta']} seed={r['seed']}": sorted(r["load_vector"], reverse=True)
                                      for r in report["runs"] if "load_vector" in r}
    (out / "seed_check.json").write_text(json.dumps(report, indent=1))
    print("random centers loads:", [[round(x, 3) for x in r] for r in rnd], flush=True)


if __name__ == "__main__":
    main()
