"""E017：真实键上，路由是否减少 RLS 记忆的串扰（离线回放，真实模型的键/值/权重）。

步骤：
  1. 从预训练 Metis-4B 记录第 3、7 层（以及其它层，见 --layers）每次 commit 的键、值、权重（真实输入）。
  2. 对每种方式，用同样的 RLS 块更新（rls_commit_update，lam=0.9，P0=kv·I，与模型一致）回放：
       K1          单库，全部 token 写入（基线）
       K4_center   4 库，路由 = argmax_b (中心化后的键 · c_b)，c_b 为随机固定中心
       K4_raw      4 库，路由 = argmax_b (未中心化的键 · c_b)
       K4_random   4 库，路由随机（对照：排除"只是分了库"的收益）
     每个库只接收路由到它的 token（权重 w=1），其余为 0。
  3. 读出：用每个已写入 token 的键作为查询（查询与写入使用同一路由），计算读出与该 token 的值的余弦。
     主指标 = 该余弦的均值（越高串扰越少）。

注意：
  - 中心 mu 使用同一批数据的均值（诊断用，不是学习）。
  - 随机中心与种子固定；每个设置多个种子。
  - 等状态：K4 每库仍为完整 kv 维（总状态为 K1 的 4 倍）。这是一个容量混杂的参考，不是等参数比较。

用法（仓库根目录）：
    PYTHONPATH=.:src python experiments/E017_routed_recall/run.py --out experiments/E017_routed_recall --n 10
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
    """返回 {layer: [episode: [commit: (k, v, w)]]}。"""
    data = {l: [] for l in layers}
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
                # 第 l 个记忆层的 commit 序列（按 chunk 顺序）
                seq = [cap[c * nl + layers_index(l, model)] for c in range(len(ep["memory_chunks"]))]
                data[l].append(seq)
    return data


def layers_index(layer, model):
    """记忆层的序号（0..7）对应模型中的第几个 block。"""
    mem_blocks = [i for i, b in enumerate(model.model.metis_blocks) if b.local_memory is not None]
    return mem_blocks.index(layer)


def route_fn(mode, keys, centers, mu):
    if mode == "K1":
        return lambda k: torch.zeros(k.shape[0], dtype=torch.long)
    if mode == "K4_random":
        g = torch.Generator().manual_seed(123)
        return lambda k: torch.randint(0, centers.shape[0], (k.shape[0],), generator=g)
    if mode == "K4_center":
        def f(k):
            kc = F.normalize(k - mu, dim=-1)
            return (kc @ centers.T).argmax(-1)
        return f
    if mode == "K4_raw":
        return lambda k: (F.normalize(k, dim=-1) @ centers.T).argmax(-1)
    raise ValueError(mode)


def simulate(seqs, mode, K, centers, mu, kv, lam=0.9):
    """对每个 episode 回放，返回所有查询的余弦列表。"""
    route = route_fn(mode, None, centers, mu)
    cos_all = []
    for seq in seqs:
        Ms = [torch.zeros(1, kv, seq[0][1].shape[-1]) for _ in range(K)]
        Ps = [(kv * torch.eye(kv)).expand(1, kv, kv).clone() for _ in range(K)]
        zs = [torch.zeros(1, kv, 1) for _ in range(K)]
        items = []
        for k, v, w in seq:
            b_idx = route(k)
            for b in range(K):
                mask = (b_idx == b).float() * w
                if mask.sum() == 0 and K > 1:
                    # 无 token 进入该库：仍需衰减（时间推进）
                    pass
                Ms[b], Ps[b], zs[b] = rls_commit_update(Ms[b], Ps[b], zs[b], k[None], v[None], mask[None], lam)
            items.append((k, v, w))
        # 查询：每个已写入 token 的键，经同一路由读出
        for k, v, w in items:
            b_idx = route(k)
            for t in range(k.shape[0]):
                if w[t] == 0:
                    continue
                b = int(b_idx[t])
                q = F.normalize(k[t], dim=-1)[None, None]
                num = (q[0] @ Ms[b][0]).squeeze(0)
                den = (q[0] @ zs[b][0]).squeeze() + 1.0
                y = num / den
                cos_all.append(F.cosine_similarity(y, v[t], dim=0).item())
    return cos_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", default="/hy-tmp/metis-models/Metis-4B")
    ap.add_argument("--n", type=int, default=10)
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

    kv = 1024
    results = {}
    for l in layers:
        seqs = data[l]
        allk = torch.cat([k for seq in seqs for (k, _, _) in seq], 0)
        mu = allk.mean(0, keepdim=True)
        results[f"layer{l}"] = {}
        for mode, K in [("K1", 1), ("K4_center", 4), ("K4_raw", 4), ("K4_random", 4)]:
            vals = []
            for s in range(args.seeds):
                g = torch.Generator().manual_seed(1000 + s)
                centers = F.normalize(torch.randn(K, kv, generator=g), dim=-1)
                c = simulate(seqs, mode, K, centers, mu, kv)
                vals.append(sum(c) / len(c))
            results[f"layer{l}"][mode] = {"recall_cos_mean": sum(vals) / len(vals),
                                          "recall_cos_per_seed": vals}
            print(f"layer{l} {mode}: {results[f'layer{l}'][mode]['recall_cos_mean']:.4f}", flush=True)
    (out / "result.json").write_text(json.dumps(results, indent=2))
    print("wrote", out / "result.json", flush=True)


if __name__ == "__main__":
    main()
