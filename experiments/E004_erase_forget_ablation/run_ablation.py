"""E004：Phase 0b，在 LoCoMo 真实读出路径上比较擦除与遗忘的贡献。

四种模式（见 docs/design_phase0b.md §3）：
    full      原始更新（基线）
    no_erase  去掉按键擦除
    no_forget 令 alpha=1（去掉全局遗忘）
    none      不 commit，无记忆

对每个证据会话：按模式 replay 记忆步骤，然后对该会话的每个问题计算答案 token 的 NLL。
输出逐条结果与汇总（配对差、bootstrap 95% 区间）。

用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E004_erase_forget_ablation/run_ablation.py \
        --checkpoint_path /hy-tmp/metis-models/Metis-4B \
        --memqa_file eval/data/memqa/locomo_evidence_sessions_tps16_20260626.jsonl \
        --max_sessions 100 --seed 0 \
        --out experiments/E004_erase_forget_ablation
"""
import argparse
import csv
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import run_inference as ri
from diagnostics.gate_ablation import install_mode

MODES = ["full", "no_erase", "no_forget", "none"]


def group_sessions(path):
    """按（对话, 证据会话集）分组，返回 OrderedDict[key] = {steps, qas}。"""
    groups = OrderedDict()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row["instance_id"].split("__qa_")[0]
            if key not in groups:
                groups[key] = {
                    "steps": [step["content"] for step in row["memory_steps"]],
                    "qas": [],
                }
            groups[key]["qas"].append({
                "instance_id": row["instance_id"],
                "question": row["question"],
                "answer": str(row["answer"]),
            })
    return groups


def answer_nll(model, tokenizer, device, question, answer):
    """答案 token 的平均 NLL（prompt 部分不计入）。"""
    messages = [{"role": "user", "content": question}]
    prompt_ids, _ = ri.encode_messages(tokenizer, messages, device, add_generation_prompt=True)
    ans_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ans_ids:
        return float("nan"), 0
    ans = torch.tensor([ans_ids], device=device, dtype=prompt_ids.dtype)
    input_ids = torch.cat([prompt_ids, ans], dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, prompt_ids.size(1):] = ans
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return float(out.loss.float().item()), len(ans_ids)


def replay(model, tokenizer, device, steps):
    model.reset()
    for content in steps:
        ri.commit_turn(model, tokenizer, device, prompt=content, response="", commit_mode="user")


def memory_states(model):
    out = {}
    for i, layer in enumerate(model.model.metis_blocks):
        lm = layer.local_memory
        if lm is None or getattr(lm, "state", None) is None:
            continue
        key = getattr(lm, "key_state", None)
        out[f"block{i}"] = (lm.state.detach().clone(), None if key is None else key.detach().clone())
    return out


def check_full_matches_original(model, tokenizer, device, steps, precision="bf16"):
    """真实模型检查。bf16：full 与原始代码逐位相等。fp32：只报告与原始 bf16 状态的差值。"""
    install_mode(model, "original")
    replay(model, tokenizer, device, steps)
    orig = memory_states(model)
    install_mode(model, "full", precision=precision)
    replay(model, tokenizer, device, steps)
    new = memory_states(model)
    assert orig.keys() == new.keys(), "layer sets differ"
    max_diff = 0.0
    for name in orig:
        s0, k0 = orig[name]
        s1, k1 = new[name]
        max_diff = max(max_diff, float((s0.float() - s1.float()).abs().max()))
        if k0 is not None and k1 is not None:
            max_diff = max(max_diff, float((k0.float() - k1.float()).abs().max()))
    return max_diff


def bootstrap_ci(values, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    means = np.array([rng.choice(values, size=values.size, replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(rows):
    by_qa = {}
    for r in rows:
        by_qa.setdefault(r["instance_id"], {})[r["mode"]] = r["nll"]
    out = []
    for mode in MODES:
        vals = [v[mode] for v in by_qa.values() if mode in v and np.isfinite(v[mode])]
        out.append({"mode": mode, "n": len(vals), "nll_mean": float(np.mean(vals)),
                    "nll_std": float(np.std(vals))})
    for mode in MODES:
        if mode == "full":
            continue
        diffs = [v[mode] - v["full"] for v in by_qa.values()
                 if mode in v and "full" in v and np.isfinite(v[mode]) and np.isfinite(v["full"])]
        diffs = np.asarray(diffs)
        lo, hi = bootstrap_ci(diffs)
        out.append({"mode": f"delta_{mode}_minus_full", "n": int(diffs.size),
                    "nll_mean": float(diffs.mean()), "nll_std": float(diffs.std(ddof=1) / np.sqrt(diffs.size)),
                    "ci95_lo": lo, "ci95_hi": hi})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--memqa_file", required=True)
    parser.add_argument("--max_sessions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw_dir", default="/hy-tmp/metis-outputs/E004")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", help="模型权重精度")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"],
                        help="记忆状态的存储与递推精度（fp32 为 Phase 0b 对照）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    load_args = argparse.Namespace(
        checkpoint_path=args.checkpoint_path, model_path=None, tokenizer_path=None,
        backbone_type="qwen3_5", device=args.device, dtype=args.dtype,
    )
    model, tokenizer, device = ri.load_model_and_tokenizer(load_args)
    model.eval()

    groups = group_sessions(args.memqa_file)
    keys = list(groups)
    chosen = random.Random(args.seed).sample(keys, min(args.max_sessions, len(keys)))
    print(f"sessions total={len(keys)} chosen={len(chosen)}", flush=True)

    with torch.no_grad():
        diff = check_full_matches_original(model, tokenizer, device, groups[chosen[0]]["steps"],
                                           precision=args.precision)
        print(f"full({args.precision})-vs-original max state diff = {diff:.3e}", flush=True)
        if args.precision == "bf16":
            assert diff == 0.0, "full mode does not reproduce original state"

        rows = []
        stats = {}
        for i, key in enumerate(chosen):
            g = groups[key]
            for mode in MODES:
                if mode == "none":
                    model.reset()
                    install_mode(model, "original")
                else:
                    install_mode(model, mode, stats=stats if mode == "full" else None,
                                 precision=args.precision)
                    replay(model, tokenizer, device, g["steps"])
                for qa in g["qas"]:
                    nll, n_ans = answer_nll(model, tokenizer, device, qa["question"], qa["answer"])
                    rows.append({"session": key, "instance_id": qa["instance_id"],
                                 "mode": mode, "nll": nll, "n_answer_tokens": n_ans})
            if (i + 1) % 10 == 0:
                print(f"progress {i + 1}/{len(chosen)}", flush=True)

    install_mode(model, "original")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_qa.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    with (out / "summary.csv").open("w", newline="") as f:
        fields = ["mode", "n", "nll_mean", "nll_std", "ci95_lo", "ci95_hi"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)

    layer_rows = []
    for name, layer in stats.items():
        if not layer:
            continue
        rel = torch.cat(layer["erase_rel"]).numpy()
        fg = torch.cat(layer["forget"]).numpy()
        layer_rows.append({"layer": name, "writes": int(rel.size),
                           "erase_rel_mean": float(rel.mean()), "erase_rel_p95": float(np.quantile(rel, 0.95)),
                           "forget_mean": float(fg.mean())})
    with (out / "state_stats_full.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        w.writeheader()
        w.writerows(layer_rows)

    config = {
        "checkpoint_path": args.checkpoint_path,
        "memqa_file": args.memqa_file,
        "sessions_total": len(keys),
        "sessions_used": len(chosen),
        "qa_rows": len({r["instance_id"] for r in rows}),
        "seed": args.seed,
        "modes": MODES,
        "dtype": args.dtype,
        "state_precision": args.precision,
        "answer_tokens": "tokenizer.encode(answer, add_special_tokens=False), no eos",
        "full_vs_original_max_diff": diff,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print("summary:", flush=True)
    for s in summary:
        print(s, flush=True)


if __name__ == "__main__":
    main()
