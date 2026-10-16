"""E024（门控 G1d）：数据自适应路由在真实对话（LoCoMo 记忆步骤）上的检验。设计：docs/design_G1d.md（预先登记）。

复用 E023 的原型学习、路由与回放函数（同一口径），只替换数据：
  - 数据：LoCoMo memqa（诊断用，不训练）。按会话去重；每个会话的 memory_steps 为一个 episode 的记忆 chunk。
  - 拆分：按对话（sample_id）。前 5 个对话拟合（原型与 PCA 基），后 5 个对话测试（至多 30 个会话）。
  - 没有查询问题参与路由与指标（指标为自检索）。

用法（仓库根目录）：PYTHONPATH=.:src python experiments/E024_G1d/run.py --out experiments/E024_G1d
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

K, LAM = 4, 0.9
LOCOMO = ROOT / "eval/data/memqa/locomo_evidence_sessions_tps16_20260626.jsonl"


def load_locomo_episodes():
    groups = OrderedDict()
    with open(LOCOMO, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            key = row["instance_id"].split("__qa_")[0]
            if key not in groups:
                groups[key] = {"episode_id": key, "conv": row["source_sample_id"],
                               "memory_chunks": [st["content"] for st in row["memory_steps"]]}
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--n_test", type=int, default=30)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    eps = load_locomo_episodes()
    convs = list(OrderedDict.fromkeys(e["conv"] for e in eps))
    fit_convs, test_convs = set(convs[:5]), set(convs[5:])
    fit = [e for e in eps if e["conv"] in fit_convs]
    test = [e for e in eps if e["conv"] in test_convs][: args.n_test]
    print(f"conversations={len(convs)} fit_sessions={len(fit)} test_sessions={len(test)}", flush=True)

    load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load); model.eval(); freeze_backbone(model)
    R.install_rls(model, lam=LAM)
    layers = [3, 7]
    fit_d = E18.record(model, tok, dev, fit, layers); test_d = E18.record(model, tok, dev, test, layers)
    R.uninstall_rls(model)

    result = {"n_fit": len(fit), "n_test": len(test), "layers": {}}
    for layer in layers:
        seqs_f, seqs_t = fit_d[layer], test_d[layer]
        Xf = torch.cat([F.normalize(k, dim=-1) for seq in seqs_f for (k, _, _) in seq], 0)
        _, _, Vh = torch.linalg.svd(Xf, full_matrices=False); Q = Vh[:256].T.contiguous()
        Xt = torch.cat([F.normalize(k, dim=-1) for seq in seqs_t for (k, _, _) in seq], 0)
        lens = [sum(k.shape[0] for (k, _, _) in seq) for seq in seqs_t]
        per_arm, loads = {}, {}

        def ep_split(b):
            outl, off = [], 0
            for L in lens:
                outl.append(b[off:off + L]); off += L
            return outl

        per_arm["K1_full"] = E23.simulate(seqs_t, torch.eye(1024), [torch.zeros(L, dtype=torch.long) for L in lens])
        per_arm["K1_pca256"] = E23.simulate(seqs_t, Q, [torch.zeros(L, dtype=torch.long) for L in lens])
        runs = []
        for s in range(args.seeds):
            g = torch.Generator().manual_seed(1000 + s)
            Cr = F.normalize(torch.randn(K, 256, generator=g) @ Q.T, dim=-1)
            b = (Xt @ Cr.T).argmax(-1)
            runs.append(E23.simulate(seqs_t, Q, ep_split(b)))
            loads.setdefault("K4_random", []).append(torch.bincount(b, minlength=K).float().div(b.numel()).tolist())
        per_arm["K4_random"] = np.mean(np.array(runs), axis=0).tolist()
        for name, beta in [("K4_CL", 0.0), ("K4_CL_cons", 1.0)]:
            runs = []
            for s in range(args.seeds):
                C, u = E23.train_prototypes(Xf, beta, seed=2000 + s)
                b = E23.route_scores(Xt, C, u, beta)
                runs.append(E23.simulate(seqs_t, Q, ep_split(b)))
                loads.setdefault(name, []).append(torch.bincount(b, minlength=K).float().div(b.numel()).tolist())
            per_arm[name] = np.mean(np.array(runs), axis=0).tolist()
        bal = []
        for s in range(args.seeds):
            g = torch.Generator().manual_seed(3000 + s)
            re, allb = [], []
            for L in lens:
                p = (torch.arange(L) % K)[torch.randperm(L, generator=g)]
                re.append(p); allb.append(p)
            bal.append(E23.simulate(seqs_t, Q, re))
            loads.setdefault("K4_balanced_free", []).append(torch.bincount(torch.cat(allb), minlength=K).float().div(sum(lens)).tolist())
        per_arm["K4_balanced_free"] = np.mean(np.array(bal), axis=0).tolist()

        def paired(a, b):
            d = np.array(per_arm[a]) - np.array(per_arm[b]); rng = np.random.default_rng(0)
            means = np.array([rng.choice(d, size=d.size, replace=True).mean() for _ in range(5000)])
            return {"mean_diff": float(d.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}

        maxshare = {n: [float(max(v)) for v in runs_] for n, runs_ in loads.items()}
        lr = {"means": {k: float(np.mean(v)) for k, v in per_arm.items()}, "per_seed_max_share": maxshare,
              "median_max_share_CL_cons": float(np.median(maxshare["K4_CL_cons"])),
              "K4_CL_cons_minus_K1_full": paired("K4_CL_cons", "K1_full"),
              "K4_CL_cons_minus_balanced_free": paired("K4_CL_cons", "K4_balanced_free"),
              "K4_CL_cons_minus_random": paired("K4_CL_cons", "K4_random"),
              "K4_CL_cons_minus_CL": paired("K4_CL_cons", "K4_CL"), "per_episode": per_arm}
        lr["G1d_layer_passed"] = bool(lr["median_max_share_CL_cons"] < 0.40
                                      and lr["K4_CL_cons_minus_K1_full"]["ci95"][0] > 0
                                      and lr["K4_CL_cons_minus_balanced_free"]["ci95"][0] > 0)
        result["layers"][f"layer{layer}"] = lr
        print(layer, {k: round(v, 4) for k, v in lr["means"].items()}, flush=True)
    result["G1d_passed"] = bool(all(v["G1d_layer_passed"] for v in result["layers"].values()))
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print("G1d_passed =", result["G1d_passed"], flush=True)


if __name__ == "__main__":
    main()
