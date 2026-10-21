"""E025 诊断：梯度范数与 episode 长度的关系（只前向、反向，不更新参数）。

动机：E025 K1_full 训练在第 1 步起梯度范数即为 10^3–10^5（见 runs/K1_full_s0.log），且第 12 步显存不足。
本脚本在初始化参数（不更新）下，对训练集前 n 个 QA episode 逐个计算梯度，记录：
  - chunk 数、token 数、答案损失；
  - 总梯度范数与各参数组的梯度范数；
  - 该 episode 的显存峰值；若因显存不足失败，记录为 oom 并继续。
判读：若梯度范数随 chunk 数增长（BPTT 长度），则问题与长序列相关；若与 chunk 数无关，则与具体数据或参数初始化相关。
不修改 train.py、不修改 metis/ 与 src/ 下的代码。

用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E025_G1e_train/diag_grad.py --arm K1_full --seed 0 --n 40 --out experiments/E025_G1e_train/diag_grad
"""
import argparse, json, random, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import importlib.util
import run_inference as ri
from holistic.banked_metis import install_banked
from train.train_utils import freeze_backbone


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


T = _load("e25train", "experiments/E025_G1e_train/train.py")  # 复用同一数据划分、臂与参数选择
E13 = T.E13
GROUPS = ["W_k", "W_v", "pool_score", "query_proj", "mem_norm", "query_norm"]


def group_of(name):
    for g in GROUPS:
        if g in name:
            return g
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="K1_full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    fit, _ = T.load_qa()
    rng = random.Random(1000 + args.seed)
    perm = list(fit); rng.shuffle(perm)
    eps = perm[: args.n]

    load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load)
    freeze_backbone(model)
    install_banked(model, lam=0.9)  # K1_full：与 train.py 相同
    E13.reinit_memory(model)
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(g in name for g in E13.GATE_NAMES):
            p.requires_grad = False
            continue
        params.append((name, p))
    print(f"trainable tensors={len(params)}", flush=True)

    model.train()
    f = open(out / "episodes.jsonl", "w")
    for i, ep in enumerate(eps):
        rec = {"i": i, "episode_id": ep["episode_id"], "n_chunks": len(ep["memory_chunks"]),
               "n_tokens": sum(len(tok.encode(c, add_special_tokens=False)) for c in ep["memory_chunks"])}
        for _, p in params:
            p.grad = None
        torch.cuda.reset_peak_memory_stats()
        try:
            loss = E13.episode_loss(model, tok, dev, ep)
            rec["loss"] = float(loss.item())
            loss.backward()
            per = {g: 0.0 for g in GROUPS + ["other"]}
            for name, p in params:
                if p.grad is not None:
                    per[group_of(name)] += float(p.grad.double().pow(2).sum())
            rec["grad_norm"] = sum(per.values()) ** 0.5
            rec["grad_norm_by_group"] = {k: v ** 0.5 for k, v in per.items()}
            rec["finite"] = bool(torch.isfinite(torch.tensor(rec["grad_norm"])))
        except torch.OutOfMemoryError:
            rec["oom"] = True
            torch.cuda.empty_cache()
        rec["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(f"ep {i} chunks={rec['n_chunks']} tokens={rec['n_tokens']} "
              f"gnorm={rec.get('grad_norm', 'OOM')} peak={rec['peak_gb']:.1f}GB", flush=True)
        model.reset()
    f.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
