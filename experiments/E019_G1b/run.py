"""E019（门控 G1b）：未中心化路由，等总状态，新测试数据。设计：docs/design_G1b.md（预先登记）。

与 E018 的区别：
  - 路由不中心化（argmax_b k̂·c_b，k̂ 为单位化的原始键，c_b 为随机固定中心，不使用数据统计量）；
  - 测试集为新的 20 条 episode（seed 9100），PCA 基仍由 seed 9000 的前 10 条拟合；
  - 报告各库负载（最大占比、归一化熵）。
判据见设计文档 §4，运行前已固定。

用法（仓库根目录）：
    PYTHONPATH=.:src python experiments/E019_G1b/run.py --out experiments/E019_G1b
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
from holistic.synthetic_memory_data import make_split  # noqa: E402
from train.train_utils import freeze_backbone  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("e18", ROOT / "experiments/E018_G1_equal_state/run.py")
E18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E18)
from holistic.banked_memory import BankedLinearMemory  # noqa: E402,F401


def simulate_raw(seqs, K, kd, Q, centers, lam=0.9):
    """未中心化路由的回放。返回逐 episode 的自检索余弦均值，以及按 token 的路由计数。"""
    dv = seqs[0][0][1].shape[-1]
    per_ep = []
    counts = torch.zeros(K)
    for seq in seqs:
        Ms = [torch.zeros(1, kd, dv) for _ in range(K)]
        Ps = [torch.eye(kd).expand(1, kd, kd).clone() for _ in range(K)]
        zs = [torch.zeros(1, kd, 1) for _ in range(K)]
        items = []
        for k, v, w in seq:
            kp = F.normalize(k @ Q, dim=-1) if Q is not None else F.normalize(k, dim=-1)
            b_idx = (F.normalize(k, dim=-1) @ centers.T).argmax(-1) if K > 1 else torch.zeros(k.shape[0], dtype=torch.long)
            counts += torch.bincount(b_idx[w > 0], minlength=K).float()
            for b in range(K):
                mask = (b_idx == b).float() * w
                Ms[b], Ps[b], zs[b] = E18.rls_commit_update(Ms[b], Ps[b], zs[b], kp[None], v[None], mask[None], lam)
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
    return per_ep, counts


def random_centers_in_subspace(K, Q, seed):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(K, Q.shape[1], generator=g) @ Q.T, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", default="/hy-tmp/metis-models/Metis-4B")
    ap.add_argument("--n_test", type=int, default=20)
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
    fit_eps = make_split("val", 10, seed=9000)     # 与 E018 相同：仅用于 PCA 基
    test_eps = make_split("val", args.n_test, seed=9100)  # 新测试集
    fit_data = E18.record(model, tok, dev, fit_eps, layers)
    test_data = E18.record(model, tok, dev, test_eps, layers)
    R.uninstall_rls(model)

    result = {"layers": layers, "n_test": args.n_test, "test_seed": 9100, "fit_seed": 9000, "layers_result": {}}
    for l in layers:
        Q = E18.pca_basis(fit_data[l], 256)
        test = test_data[l]
        per_arm = {}
        loads = {}
        arms = {"K1_full": (1, 1024, None), "K1_pca256": (1, 256, Q), "K4_random": (4, 256, Q), "K4_raw": (4, 256, Q)}
        for name, (K, kd, proj) in arms.items():
            seed_pe, seed_counts = [], []
            for s in range(args.seeds):
                if name == "K4_random":
                    # 随机路由：与 raw 同样的内容无关分配（每个 token 独立随机）
                    g = torch.Generator().manual_seed(77 + s)
                    pe = []
                    for seq in test:
                        Ms = [torch.zeros(1, kd, 1024) for _ in range(K)]
                        Ps = [torch.eye(kd).expand(1, kd, kd).clone() for _ in range(K)]
                        zs = [torch.zeros(1, kd, 1) for _ in range(K)]
                        items = []
                        for k, v, w in seq:
                            kp = F.normalize(k @ proj, dim=-1)
                            b_idx = torch.randint(0, K, (k.shape[0],), generator=g)
                            for b in range(K):
                                mask = (b_idx == b).float() * w
                                Ms[b], Ps[b], zs[b] = E18.rls_commit_update(Ms[b], Ps[b], zs[b], kp[None], v[None], mask[None], 0.9)
                            items.append((k, v, w, kp, b_idx))
                        cos = []
                        for k, v, w, kp, b_idx in items:
                            for t in range(k.shape[0]):
                                if w[t] == 0:
                                    continue
                                b = int(b_idx[t]); q = kp[t]
                                num = (q[None] @ Ms[b][0]).squeeze(0)
                                den = (q[None] @ zs[b][0]).squeeze() + 1.0
                                cos.append(F.cosine_similarity(num / den, v[t], dim=0).item())
                        pe.append(float(np.mean(cos)))
                    seed_pe.append(pe)
                else:
                    centers = random_centers_in_subspace(K, proj, 1000 + s) if K > 1 and proj is not None else None
                    if K == 1:
                        pe, cnt = simulate_raw(test, 1, kd, proj, None)
                    else:
                        pe, cnt = simulate_raw(test, K, kd, proj, centers)
                    seed_pe.append(pe)
                    seed_counts.append(cnt)
            per_arm[name] = np.mean(np.array(seed_pe), axis=0).tolist()
            if seed_counts:
                c = torch.stack(seed_counts).mean(0)
                p = c / c.sum()
                ent = float(-(p.clamp_min(1e-12).log() * p).sum() / np.log(K))
                loads[name] = {"max_share": float(p.max()), "norm_entropy": ent}
            print(f"layer{l} {name}: mean={np.mean(per_arm[name]):.4f}", flush=True)

        def paired(a, b):
            d = np.array(per_arm[a]) - np.array(per_arm[b])
            rng = np.random.default_rng(0)
            means = np.array([rng.choice(d, size=d.size, replace=True).mean() for _ in range(5000)])
            return {"mean_diff": float(d.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}

        result["layers_result"][f"layer{l}"] = {
            "means": {k: float(np.mean(v)) for k, v in per_arm.items()},
            "K4_raw_minus_K1_full": paired("K4_raw", "K1_full"),
            "K4_raw_minus_K4_random": paired("K4_raw", "K4_random"),
            "K1_pca256_minus_K1_full": paired("K1_pca256", "K1_full"),
            "loads": loads,
            "per_episode": per_arm,
        }
    passed = all(
        result["layers_result"][f"layer{l}"]["K4_raw_minus_K1_full"]["ci95"][0] > 0
        and result["layers_result"][f"layer{l}"]["K4_raw_minus_K4_random"]["ci95"][0] > 0
        for l in layers)
    result["G1b_passed"] = bool(passed)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print("G1b_passed =", passed, flush=True)


if __name__ == "__main__":
    main()
