"""E016（原计划第二步的前置诊断）：真实 Metis 的写入键分布是否有路由可用的结构。

从预训练 Metis-4B 取各记忆层的写入键（与 RLS / GDN 写入使用的 W_k 相同），统计：
    - 有效秩（参与比，participation ratio of singular values）
    - 前 16 / 64 个主成分的方差占比
    - 两两余弦的分布（均值、|cos|>0.5 的比例）
并与各向同性随机单位键对比。

用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E016_real_keys/run.py --out experiments/E016_real_keys --n 10
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri  # noqa: E402
import holistic.rls_metis as R  # noqa: E402
from holistic.synthetic_memory_data import make_split  # noqa: E402
from train.train_utils import freeze_backbone  # noqa: E402

KEYS = {"list": None}
_orig = R.rls_commit_update


def _capture(M, P, z, kappa, v, w, lam, p_max=None):
    if KEYS["list"] is not None:
        KEYS["list"].append(kappa.detach().float().reshape(-1, kappa.shape[-1]).cpu())
    return _orig(M, P, z, kappa, v, w, lam, p_max=p_max)


R.rls_commit_update = _capture


def stats(X):
    """X: (N, d)，行为单位键（先乘回 sqrt(kv) 得到单位向量）。"""
    X = F.normalize(X, dim=-1)
    s = torch.linalg.svdvals(X)
    s2 = s ** 2
    pr = (s2.sum() ** 2 / (s2 ** 2).sum()).item()
    frac = s2.cumsum(0) / s2.sum()
    cos = X @ X.T
    n = X.shape[0]
    off = cos[~torch.eye(n, dtype=torch.bool)]
    return {"n_keys": n, "participation_ratio": pr,
            "var_top16": frac[15].item(), "var_top64": frac[63].item(),
            "cos_mean": off.mean().item(), "frac_abs_cos_gt_0.5": (off.abs() > 0.5).float().mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", default="/hy-tmp/metis-models/Metis-4B")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    load = argparse.Namespace(checkpoint_path=args.checkpoint_path, model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load)
    model.eval()
    freeze_backbone(model)
    R.install_rls(model, lam=0.9)
    eps = make_split("val", args.n, seed=9000)
    per_layer = {}
    with torch.no_grad():
        for ep in eps:
            model.reset()
            KEYS["list"] = []
            for chunk in ep["memory_chunks"]:
                ids, mask = ri.encode_messages(tok, [{"role": "user", "content": chunk}], dev,
                                               add_generation_prompt=False)
                model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)
            captured = KEYS["list"]
            KEYS["list"] = None
            n_layers = len(captured) // len(ep["memory_chunks"])
            for idx, k in enumerate(captured):
                per_layer.setdefault(idx % n_layers, []).append(k)
    R.uninstall_rls(model)

    result = {"real": {}, "isotropic_reference": {}}
    for layer, parts in sorted(per_layer.items()):
        X = torch.cat(parts, 0)
        result["real"][f"layer_index{layer}"] = stats(X)
    torch.manual_seed(0)
    result["isotropic_reference"] = stats(torch.randn(2000, 1024))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
