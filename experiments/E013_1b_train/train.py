"""E013（阶段 1b）：原版 Metis 记忆与 RLS 记忆在合成记忆任务上的对照训练。

变体：
    baseline  原版 Metis（GDN 形式，门控 alpha/beta 可学习），训练 freeze_backbone 指定的全部记忆参数
    rls       RLS 记忆（src/holistic/rls_metis.py，lam 固定），去掉 alpha/beta 门控参数（不参与前向）
两者使用相同的数据、步数、学习率、种子与验证集。骨干冻结。

每个 episode：N_MEM 个记忆 chunk 依次前向（commit_memory=True，梯度经记忆写入回传），
然后对查询的答案 token 计算 NLL。梯度累积 ACCUM 个 episode 后更新一次。

用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E013_1b_train/train.py --variant rls --seed 0 \
        --steps 300 --out experiments/E013_1b_train/rls_s0
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri  # noqa: E402
from holistic.rls_metis import install_rls  # noqa: E402
from holistic.synthetic_memory_data import data_hash, make_split  # noqa: E402
from train.train_utils import MasterWeightAdamW, freeze_backbone  # noqa: E402

GATE_NAMES = ("gated_delta_alpha", "gated_delta_beta")


def reinit_memory(model):
    """把记忆相关参数重新初始化（从头训练，设计 1b §5）。骨干不动。

    线性层用 PyTorch 默认初始化；query_proj 与原版一致（normal, std=hidden^-0.5，bias=0）；
    门控 alpha/beta 与原版一致（权重为 0，偏置使 alpha 与 beta 的初值为 1，即无遗忘、全写入）；
    归一化层权重为 1（mem_norm_init 默认值）。
    """
    import math
    for block in model.model.metis_blocks:
        if block.local_memory is None:
            continue
        hm = block.hyper_memory
        for name in ("W_k", "W_v", "pool_score"):
            mod = getattr(hm, name, None)
            if mod is not None:
                mod.reset_parameters()
        for name in ("gated_delta_alpha", "gated_delta_beta"):
            mod = getattr(hm, name, None)
            if mod is not None:
                torch.nn.init.zeros_(mod.weight)
                torch.nn.init.constant_(mod.bias, math.log(0.9999 / 0.0001))
        qp = getattr(block, "query_proj", None)
        if qp is not None:
            torch.nn.init.normal_(qp.weight, std=model.config.hidden_size ** -0.5)
            if qp.bias is not None:
                torch.nn.init.zeros_(qp.bias)
        for name in ("mem_norm", "query_norm"):
            mod = getattr(block, name, None)
            if mod is not None and hasattr(mod, "weight"):
                mod.weight.data.fill_(1.0)


def episode_loss(model, tokenizer, device, ep):
    """一个 episode 的答案 NLL（会对记忆写入回传梯度）。"""
    model.reset()
    for chunk in ep["memory_chunks"]:
        ids, mask = ri.encode_messages(tokenizer, [{"role": "user", "content": chunk}], device,
                                       add_generation_prompt=False)
        model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)
    prompt_ids, _ = ri.encode_messages(tokenizer, [{"role": "user", "content": ep["question"]}], device,
                                       add_generation_prompt=True)
    ans_ids = tokenizer.encode(ep["answer"], add_special_tokens=False)
    ans = torch.tensor([ans_ids], device=device, dtype=prompt_ids.dtype)
    input_ids = torch.cat([prompt_ids, ans], dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, prompt_ids.size(1):] = ans
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels,
                use_cache=False, logits_to_keep=0)
    return out.loss


@torch.no_grad()
def evaluate(model, tokenizer, device, episodes, per_episode=None):
    """返回均值；若给出 per_episode（列表），则逐条追加 (episode_id, nll)，供配对分析。"""
    model.eval()
    nlls = []
    for ep in episodes:
        x = float(episode_loss(model, tokenizer, device, ep).item())
        nlls.append(x)
        if per_episode is not None:
            per_episode.append({"episode_id": ep["episode_id"], "nll": x})
    model.train()
    finite = [x for x in nlls if math.isfinite(x)]
    return {"nll_mean": sum(finite) / len(finite), "n": len(finite), "n_nonfinite": len(nlls) - len(finite)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", default="/hy-tmp/metis-models/Metis-4B")
    ap.add_argument("--variant", choices=["baseline", "rls"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_eps = make_split("train", args.steps * args.accum, seed=1000 + args.seed)
    val_eps = make_split("val", args.n_val, seed=9000)          # 主指标：词表与训练完全不重叠
    seen_eps = make_split("train", args.n_val, seed=9500)       # 诊断：训练词表、未见过的 episode
    (out / "data_hash.json").write_text(json.dumps({
        "train_hash": data_hash(train_eps), "val_hash": data_hash(val_eps), "val_seen_hash": data_hash(seen_eps),
        "n_train": len(train_eps), "n_val": len(val_eps), "seed": args.seed}, indent=2))

    load = argparse.Namespace(checkpoint_path=args.checkpoint_path, model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tokenizer, device = ri.load_model_and_tokenizer(load)
    freeze_backbone(model)
    if args.variant == "rls":
        n = install_rls(model, lam=args.lam)
        print(f"installed RLS in {n} layers (lam={args.lam})", flush=True)
    reinit_memory(model)
    print("memory parameters re-initialized (from scratch)", flush=True)
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if args.variant == "rls" and any(g in name for g in GATE_NAMES):
            p.requires_grad = False
            continue
        params.append(p)
    n_params = sum(p.numel() for p in params)
    print(f"trainable tensors={len(params)} params={n_params:,}", flush=True)
    opt = MasterWeightAdamW(params, lr=args.lr, weight_decay=0.0)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    log = []
    t0 = time.time()
    init_eval = evaluate(model, tokenizer, device, val_eps)
    init_seen = evaluate(model, tokenizer, device, seen_eps)
    log.append({"step": 0, "val_unseen_nll": init_eval["nll_mean"], "val_seen_nll": init_seen["nll_mean"]})
    print(f"step 0 val_unseen nll={init_eval['nll_mean']:.4f} val_seen nll={init_seen['nll_mean']:.4f}", flush=True)

    ep_idx = 0
    model.train()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1)
        opt.zero_grad(set_to_none=True)
        losses = []
        for _ in range(args.accum):
            loss = episode_loss(model, tokenizer, device, train_eps[ep_idx])
            ep_idx += 1
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at step {step}; stopping (no tuning to hide it)")
            (loss / args.accum).backward()
            losses.append(float(loss.item()))
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gnorm):
            raise RuntimeError(f"non-finite grad norm at step {step}; stopping")
        opt.step()
        rec = {"step": step, "train_nll": sum(losses) / len(losses), "grad_norm": float(gnorm),
               "lr": lr_at(step - 1), "elapsed_s": time.time() - t0}
        if step % args.eval_every == 0 or step == args.steps:
            pe_u, pe_s = [], []
            ev = evaluate(model, tokenizer, device, val_eps, per_episode=pe_u)
            ev_seen = evaluate(model, tokenizer, device, seen_eps, per_episode=pe_s)
            (out / f"val_episode_nll_step{step}.json").write_text(
                json.dumps({"unseen": pe_u, "seen": pe_s}, indent=1))
            rec.update({"val_unseen_nll": ev["nll_mean"], "val_seen_nll": ev_seen["nll_mean"],
                        "val_n_nonfinite": ev["n_nonfinite"]})
            print(f"step {step} val_unseen nll={ev['nll_mean']:.4f} val_seen nll={ev_seen['nll_mean']:.4f}", flush=True)
        log.append(rec)
        print(f"step {step} train nll={rec['train_nll']:.4f} gnorm={rec['grad_norm']:.3f} "
              f"t={rec['elapsed_s']:.0f}s", flush=True)
        (out / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")

    pe_u, pe_s = [], []
    final = evaluate(model, tokenizer, device, val_eps, per_episode=pe_u)
    final_seen = evaluate(model, tokenizer, device, seen_eps, per_episode=pe_s)
    (out / "val_episode_nll_final.json").write_text(json.dumps({"unseen": pe_u, "seen": pe_s}, indent=1))
    summary = {"variant": args.variant, "seed": args.seed, "init": "scratch",
               "final_val_seen_nll": final_seen["nll_mean"], "steps": args.steps, "accum": args.accum,
               "lr": args.lr, "lam": args.lam if args.variant == "rls" else None,
               "trainable_params": n_params, "init_val_nll": init_eval["nll_mean"],
               "final_val_nll": final["nll_mean"], "n_val": final["n"],
               "wall_time_s": time.time() - t0}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
