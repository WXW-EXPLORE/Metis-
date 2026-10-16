"""E022（N13 的检验）：未中心化路由的负载对随机中心抽取的稳健性。
真实键（预训练 Metis-4B，第 3、7 层，seed 9400 的 20 条 episode），PCA 基与 E021 相同（seed 9000 前 10 条）。
对 10 个中心种子，统计各库负载的最大占比与归一化熵（按 token，每层、每种子）。
用法（仓库根目录）：PYTHONPATH=.:src python experiments/E022_load_robustness/run.py --out experiments/E022_load_robustness
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri
import holistic.rls_metis as R
from holistic.synthetic_memory_data import make_split
from train.train_utils import freeze_backbone
_s = importlib.util.spec_from_file_location("e18", ROOT / "experiments/E018_G1_equal_state/run.py")
E18 = importlib.util.module_from_spec(_s); _s.loader.exec_module(E18)

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, default=10)
args = ap.parse_args()
load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                          backbone_type="qwen3_5", device="cuda", dtype="bf16")
model, tok, dev = ri.load_model_and_tokenizer(load); model.eval(); freeze_backbone(model)
R.install_rls(model, lam=0.9)
fit = make_split("val", 10, seed=9000); test = make_split("val", 20, seed=9400)
fit_d = E18.record(model, tok, dev, fit, [3, 7]); test_d = E18.record(model, tok, dev, test, [3, 7])
R.uninstall_rls(model)
K = 4; res = {}
for layer in [3, 7]:
    Xf = torch.cat([F.normalize(k, dim=-1) for seq in fit_d[layer] for (k, _, _) in seq], 0)
    _, _, Vh = torch.linalg.svd(Xf, full_matrices=False); Q = Vh[:256].T
    keys = torch.cat([k for seq in test_d[layer] for (k, _, _) in seq], 0)
    kn = F.normalize(keys, dim=-1)
    rows = []
    for s in range(args.seeds):
        g = torch.Generator().manual_seed(1000 + s)
        C = F.normalize(torch.randn(K, 256, generator=g) @ Q.T, dim=-1)
        idx = (kn @ C.T).argmax(-1)
        p = torch.bincount(idx, minlength=K).float() / idx.numel()
        ent = float(-(p.clamp_min(1e-12).log() * p).sum() / np.log(K))
        rows.append({"seed": s, "max_share": float(p.max()), "norm_entropy": ent, "shares": p.tolist()})
    res[f"layer{layer}"] = {"per_seed": rows,
                            "max_share_min": min(r["max_share"] for r in rows),
                            "max_share_median": float(np.median([r["max_share"] for r in rows])),
                            "max_share_max": max(r["max_share"] for r in rows),
                            "n_seeds_max_share_gt_0.5": sum(r["max_share"] > 0.5 for r in rows)}
    print(layer, {k: v for k, v in res[f'layer{layer}'].items() if k != 'per_seed'}, flush=True)
Path(args.out).mkdir(parents=True, exist_ok=True)
(Path(args.out) / "result.json").write_text(json.dumps(res, indent=1))
print("wrote", flush=True)
