"""E018（门控 G1）：等总状态下，内容路由的分库是否优于单库（真实键，离线回放）。

设计：docs/design_G1.md。
  - 键：预训练 Metis-4B 的真实写入键（单位化）。
  - PCA 基：前 10 条 episode 拟合（每层），后 10 条 episode 测试（不在测试键上拟合）。
  - 各臂的状态（kd × dv）：K1_full = 1024×1024（基线）；K4 各库 256×1024，合计 1024×1024（等状态）。
  - RLS 块更新（rls_commit_update，λ=0.9，P0=I，单位键）。
  - 查询：每个已写入 token 的键，经同一路由读出，计算读出与其值的余弦（自检索）。
  - 配对统计：按 episode 配对（先对 3 个随机中心种子取平均），bootstrap 95% 区间。

用法（仓库根目录）：
    PYTHONPATH=.:src python experiments/E018_G1_equal_state/run.py --out experiments/E018_G1_equal_state --n 20
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri  # noqa: E402
import holistic.rls_metis as R  # noqa: E402
from holistic.rls_metis import rls_commit_update  # noqa: E402
from holistic.synthetic_memory_data import make_split  # noqa: E402
from train.train_utils import freeze_backbone  # noqa: E402

CAP = {"list": None}
_orig = R.rls_commit_update


def _capture(M, P, z, kappa, v, w, lam, p_max=None):
    if CAP["list"] is not None:
        CAP["list"].append((kappa.detach().float()[0].cpu(), v.detach().float()[0].cpu(), w.detach().float()[0].cpu()))
    return _orig(M, P, z, kappa, v, w, lam, p_max=p_max)


R.rls_commit_update = _capture


def record(model, tok, dev, eps, layers):
    data = {l: [] for l in layers}
    mem_blocks = [i for i, b in enumerate(model.model.metis_blocks) if b.local_memory is not None]
    with torch.no_grad():
        for ep in eps:
            model.reset()
            CAP["list"] = []
            for chunk in ep["memory_chunks"]:
                ids, mask = ri.encode_messages(tok, [{"role": "user", "content": chunk}], dev,
                                               add_generation_prompt=False)
                model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)
            cap = CAP["list"]
            CAP["list"] = None
            nl = len(cap) // len(ep["memory_chunks"])
            for l in layers:
                idx = mem_blocks.index(l)
                data[l].append([cap[c * nl + idx] for c in range(len(ep["memory_chunks"]))])
    return data


def pca_basis(seqs, kd):
    """前 kd 个右奇异向量（单位化键，不中心化：保留共同方向）。"""
    X = torch.cat([F.normalize(k, dim=-1) for seq in seqs for (k, _, _) in seq], 0)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return Vh[:kd].T.contiguous()  # (1024, kd)


def route_fn(mode, centers, mu):
    if mode == "random":
        g = torch.Generator().manual_seed(77)
        return lambda k: torch.randint(0, centers.shape[0], (k.shape[0],), generator=g)
    if mode == "center":
        return lambda k: (F.normalize(k - mu, dim=-1) @ centers.T).argmax(-1)
    if mode == "raw":
        return lambda k: (F.normalize(k, dim=-1) @ centers.T).argmax(-1)
    raise ValueError(mode)


def simulate(seqs, K, kd, Q, mode, centers, mu, lam=0.9):
    """回放每条 episode，返回逐 episode 的自检索余弦均值（list，长度 = episode 数）。"""
    route = route_fn(mode, centers, mu) if K > 1 else (lambda k: torch.zeros(k.shape[0], dtype=torch.long))
    per_ep = []
    dv = seqs[0][0][1].shape[-1]
    for seq in seqs:
        Ms = [torch.zeros(1, kd, dv) for _ in range(K)]
        Ps = [torch.eye(kd).expand(1, kd, kd).clone() for _ in range(K)]
        zs = [torch.zeros(1, kd, 1) for _ in range(K)]
        items = []
        for k, v, w in seq:
            kp = F.normalize(k @ Q, dim=-1) if Q is not None else F.normalize(k, dim=-1)
            b_idx = route(k)
            for b in range(K):
                mask = (b_idx == b).float() * w
                Ms[b], Ps[b], zs[b] = rls_commit_update(Ms[b], Ps[b], zs[b], kp[None], v[None], mask[None], lam)
            items.append((k, v, w, kp, b_idx))
        cos = []
        for k, v, w, kp, b_idx in items:
            for t in range(k.shape[0]):
                if w[t] == 0:
                    continue
                b = int(b_idx[t])
                q = kp[t]
                num = (q[None] @ Ms[b][0]).squeeze(0)
                den = (q[None] @ zs[b][0]).squeeze() + 1.0
                cos.append(F.cosine_similarity(num / den, v[t], dim=0).item())
        per_ep.append(float(np.mean(cos)))
    return per_ep


def bootstrap(diff, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff)
    means = np.array([rng.choice(d, size=d.size, replace=True).mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", default="/hy-tmp/metis-models/Metis-4B")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--layers", type=str, default="3,7")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    load = argparse.Namespace(checkpoint_path=args.checkpoint_path, model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load)
    model.eval()
    freeze_backbone(model)
    R.install_rls(model, lam=0.9)
    eps = make_split("val", args.n, seed=9000)
    data = record(model, tok, dev, eps, layers)
    R.uninstall_rls(model)

    half = args.n // 2
    arms = [("K1_full", 1, 1024, "none"), ("K1_pca256", 1, 256, "none"),
            ("K4_pca_center", 4, 256, "center"), ("K4_pca_raw", 4, 256, "raw"),
            ("K4_pca_random", 4, 256, "random")]
    result = {"layers": layers, "n_episodes": args.n, "n_test": args.n - half, "arms": {}}
    for l in layers:
        seqs = data[l]
        fit, test = seqs[:half], seqs[half:]
        Q = pca_basis(fit, 256)
        mu = torch.cat([F.normalize(k, dim=-1) for seq in fit for (k, _, _) in seq], 0).mean(0, keepdim=True)
        per_arm = {}
        for name, K, kd, mode in arms:
            proj = Q if kd == 256 else None
            seed_means = []
            for s in range(args.seeds):
                g = torch.Generator().manual_seed(1000 + s)
                if K > 1:
                    # 中心位于 PCA 子空间内，并映回 1024 维（路由作用于完整键）
                    centers_proj = F.normalize(torch.randn(K, kd, generator=g) @ Q.T, dim=-1)
                else:
                    centers_proj = None
                pe = simulate(test, K, kd, proj, mode if K > 1 else "none", centers_proj, mu.squeeze(0))
                seed_means.append(pe)
            per_arm[name] = np.mean(np.array(seed_means), axis=0).tolist()
            print(f"layer{l} {name}: mean={np.mean(per_arm[name]):.4f}", flush=True)
        base = per_arm["K1_full"]
        result.setdefault("paired_vs_K1_full", {})[f"layer{l}"] = {}
        for name in per_arm:
            if name == "K1_full":
                continue
            diff = np.array(per_arm[name]) - np.array(base)
            m, lo, hi = bootstrap(diff)
            result["paired_vs_K1_full"][f"layer{l}"][name] = {"mean_diff": m, "ci95": [lo, hi]}
        for name, pe in per_arm.items():
            result["arms"].setdefault(f"layer{l}", {})[name] = {"mean": float(np.mean(pe)), "per_episode": pe}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["paired_vs_K1_full"], indent=1), flush=True)


if __name__ == "__main__":
    main()
