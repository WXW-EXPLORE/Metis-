"""E011（阶段 1a）：训练可行性探针。

在 Metis-4B 上按训练器的方式运行一个 episode：K 个记忆 chunk（commit_memory=True，梯度经记忆写入回传），
最后一个 chunk 计算标签损失。骨干冻结（train.train_utils.freeze_backbone）。
输出：峰值显存、单步耗时、记忆参数梯度是否非零。输入为随机 token（只测可行性，不代表训练效果）。

用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E011_train_feasibility/probe.py --chunks 4 --chunk_len 512
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri  # noqa: E402
from train.train_utils import count_params, freeze_backbone  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", default="/hy-tmp/metis-models/Metis-4B")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--chunk_len", type=int, default=512)
    ap.add_argument("--answer_len", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    load = argparse.Namespace(checkpoint_path=args.checkpoint_path, model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, _tok, device = ri.load_model_and_tokenizer(load)
    model.train()
    n_frozen_free = freeze_backbone(model)
    total, trainable = count_params(model)
    print(f"trainable tensors={n_frozen_free} total_params={total:,} trainable_params={trainable:,}", flush=True)

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    times = []
    loss_val = None
    for step in range(args.steps):
        model.reset()
        torch.cuda.synchronize()
        t0 = time.time()
        loss_total = 0.0
        for c in range(args.chunks):
            ids = torch.randint(0, 200000, (1, args.chunk_len), device=device)
            mask = torch.ones_like(ids)
            last = c == args.chunks - 1
            if last:
                labels = torch.full_like(ids, -100)
                labels[:, -args.answer_len:] = torch.randint(0, 200000, (1, args.answer_len), device=device)
                out = model(input_ids=ids, attention_mask=mask, labels=labels, commit_memory=True,
                            use_cache=False, logits_to_keep=0)
                loss = out.loss
                loss_total = loss
            else:
                out = model(input_ids=ids, attention_mask=mask, commit_memory=True,
                            use_cache=False, logits_to_keep=1)
        loss_total.backward()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        loss_val = float(loss_total.item())
        print(f"step {step}: loss={loss_val:.4f} time={times[-1]:.2f}s "
              f"peak_mem={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)

    grad_norms = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_norms[name] = float(p.grad.norm().item())
    nonzero = sum(1 for v in grad_norms.values() if v > 0)
    zero_names = sorted(k for k, v in grad_norms.items() if v == 0)
    finite = all(torch.isfinite(torch.tensor(v)).item() for v in grad_norms.values())
    summary = {
        "chunks": args.chunks, "chunk_len": args.chunk_len, "grad_ckpt": args.grad_ckpt,
        "trainable_params": trainable, "total_params": total,
        "peak_mem_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "step_time_s_last": times[-1], "loss_last": loss_val,
        "params_with_grad": len(grad_norms), "params_with_nonzero_grad": nonzero,
        "all_grads_finite": finite,
        "zero_grad_params": zero_names,
        "step_times_s": times,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
