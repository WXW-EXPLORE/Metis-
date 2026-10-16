"""E016b：真实键上的硬路由（top-1，随机固定中心）是否均衡，中心化前后对比。
用法（仓库根目录）：PYTHONPATH=.:src python experiments/E016_real_keys/route_test.py --out experiments/E016_real_keys
"""
import argparse, json, sys
from pathlib import Path
import torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri
import holistic.rls_metis as R
from holistic.synthetic_memory_data import make_split
from train.train_utils import freeze_backbone
from importlib import util as iu
spec = iu.spec_from_file_location("e16", ROOT / "experiments/E016_real_keys/run.py")
E = iu.module_from_spec(spec); spec.loader.exec_module(E)

def util_stats(X, C, K):
    idx = (X @ C.T).argmax(-1)
    p = torch.bincount(idx, minlength=K).float() / idx.numel()
    ent = -(p.clamp_min(1e-12).log() * p).sum().item() / torch.log(torch.tensor(float(K))).item()
    return {"max_util": p.max().item(), "norm_entropy": ent, "used_banks": int((p > 0).sum())}

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--K", type=int, default=4)
ap.add_argument("--n", type=int, default=3); args = ap.parse_args()
load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                          backbone_type="qwen3_5", device="cuda", dtype="bf16")
model, tok, dev = ri.load_model_and_tokenizer(load); model.eval(); freeze_backbone(model)
R.install_rls(model, lam=0.9)
eps = make_split("val", args.n, seed=9000)
keys = {}
with torch.no_grad():
    for ep in eps:
        model.reset(); E.KEYS["list"] = []
        for chunk in ep["memory_chunks"]:
            ids, mask = ri.encode_messages(tok, [{"role": "user", "content": chunk}], dev, add_generation_prompt=False)
            model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)
        cap = E.KEYS["list"]; E.KEYS["list"] = None
        nl = len(cap) // len(ep["memory_chunks"])
        for i, k in enumerate(cap):
            keys.setdefault(i % nl, []).append(F.normalize(k, dim=-1))
R.uninstall_rls(model)
torch.manual_seed(0)
out = {}
for layer in [0, 3, 7]:
    X = torch.cat(keys[layer], 0)
    C = F.normalize(torch.randn(args.K, X.shape[1]), dim=-1)  # 随机固定中心（未训练）
    mu = X.mean(0, keepdim=True)
    out[f"layer_index{layer}"] = {
        "uncentered": util_stats(X, C, args.K),
        "centered": util_stats(F.normalize(X - mu, dim=-1), C, args.K),
    }
(Path(args.out) / "route_test.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1), flush=True)
