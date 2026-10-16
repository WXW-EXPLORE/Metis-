"""E002：统计训练后 Metis-4B 的写入门控 alpha、beta 分布。

做法：包裹每个 hyper memory 的 ``_apply_gated_delta_rule_update``，在原计算之前
按同一公式记录 alpha（每次写入）、beta（每个被选 token）及其 sigmoid 门控值。
记录只读，不改变写入结果。

用法（在仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=. python experiments/E002_gate_stats/collect_gates.py \
        --checkpoint_path /hy-tmp/metis-models/Metis-4B \
        --questions_file example/multi_turn_questions.json \
        --num_synthetic 200 --seed 0 \
        --out experiments/E002_gate_stats \
        --raw_dir /hy-tmp/metis-outputs/E002
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import run_inference as ri

METRICS = ["alpha_write", "alpha_gate_tok", "beta_gate_tok", "beta_tok", "n_selected", "n_tokens"]
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]
FOODS = ["beef burgers", "sushi", "dumplings", "pasta", "tacos", "ramen", "curry", "salad"]
CITIES = ["Beijing", "Paris", "Tokyo", "Berlin", "Lagos", "Toronto", "Lima", "Oslo"]
DISTRACTORS = [
    "Yesterday was a nice day.",
    "The weather is cold today.",
    "I finished reading a long book.",
    "My keyboard is a bit noisy.",
    "What is the capital of France?",
]


def synthetic_sessions(n, seed):
    """生成合成的记忆会话：若干事实、干扰句，最后提问。仅用于初步统计。"""
    rng = random.Random(seed)
    sessions = []
    for i in range(n):
        name = rng.choice(NAMES)
        food = rng.choice(FOODS)
        city = rng.choice(CITIES)
        age = rng.randint(5, 80)
        facts = [
            f"Please remember that {name} likes {food}.",
            f"Please remember that {name} lives in {city}.",
            f"Please remember that {name} is {age} years old.",
        ]
        rng.shuffle(facts)
        turns = []
        for fact in facts[: rng.randint(1, 3)]:
            turns.append(fact)
            for _ in range(rng.randint(0, 2)):
                turns.append(rng.choice(DISTRACTORS))
        turns.append(f"What does {name} like to eat?")
        sessions.append({"name": f"synthetic_{i:04d}", "system": None, "prompts": turns})
    return sessions


def memqa_sessions(path):
    """从 eval/data 的 memqa JSONL 读取记忆会话。

    每个（对话, 证据会话集）只取一次 memory_steps，去掉同一会话在多条 QA 中的重复。
    每一步记忆文本作为 user 消息 commit（commit_mode=user），不需要助手回复。
    """
    seen = set()
    sessions = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["source_sample_id"], row["instance_id"].split("__qa_")[0])
            if key in seen:
                continue
            seen.add(key)
            sessions.append({
                "name": row["instance_id"].split("__qa_")[0],
                "system": None,
                "prompts": [step["content"] for step in row["memory_steps"]],
                "commit_mode": "user",
            })
    return sessions


def scripted_reply(prompt):
    """commit 用的固定助手回复。统计只需要写入内容，不需要生成质量。"""
    if prompt.startswith("Please remember"):
        return "Got it."
    if prompt.startswith("What"):
        return "Noted."
    return "Okay."


def instrument(model, records):
    """包裹所有 hyper memory 的写入函数，记录门控值。返回被包裹的模块数。"""
    count = 0
    for name, mod in model.named_modules():
        if not hasattr(mod, "_apply_gated_delta_rule_update"):
            continue
        orig = mod._apply_gated_delta_rule_update
        block = name

        def wrapped(h_normed, weights, local_memory, _mod=mod, _orig=orig, _block=block):
            with torch.no_grad():
                w = weights.float()
                a_gate = torch.sigmoid(_mod.gated_delta_alpha(h_normed).squeeze(-1)).float()
                b_gate = torch.sigmoid(_mod.gated_delta_beta(h_normed).squeeze(-1)).float()
                mass = w.sum(dim=1).clamp(min=1e-6)
                alpha = (w * a_gate).sum(dim=1) / mass
                beta = w * (_mod.update_ratio * b_gate)
                sel = w > 0
                rec = records[_block]
                rec["alpha_write"].append(alpha.flatten().cpu().numpy())
                rec["alpha_gate_tok"].append(a_gate[sel].cpu().numpy())
                rec["beta_gate_tok"].append(b_gate[sel].cpu().numpy())
                rec["beta_tok"].append(beta[sel].cpu().numpy())
                rec["n_selected"].append(sel.sum(dim=1).float().cpu().numpy())
                rec["n_tokens"].append(np.full(w.shape[0], w.shape[1], dtype=np.float32))
            return _orig(h_normed, weights, local_memory)

        mod._apply_gated_delta_rule_update = wrapped
        count += 1
    return count


def summarize(records):
    rows = []
    for block in sorted(records, key=lambda b: int("".join(c for c in b if c.isdigit()) or 0)):
        for metric in METRICS:
            arr = np.concatenate(records[block][metric]) if records[block][metric] else np.array([])
            row = {"block": block, "metric": metric, "count": int(arr.size)}
            if arr.size:
                row.update({
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                })
                for q in QUANTILES:
                    row[f"p{int(q * 100):02d}"] = float(np.quantile(arr, q))
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--questions_file", default="example/multi_turn_questions.json")
    parser.add_argument("--memqa_file", default=None,
                        help="若指定，只使用该 memqa JSONL（真实评测数据），忽略 example 与合成会话")
    parser.add_argument("--num_synthetic", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    load_args = argparse.Namespace(
        checkpoint_path=args.checkpoint_path, model_path=None, tokenizer_path=None,
        backbone_type="qwen3_5", device=args.device, dtype=args.dtype,
    )
    model, tokenizer, device = ri.load_model_and_tokenizer(load_args)
    model.eval()

    records = defaultdict(lambda: defaultdict(list))
    n_wrapped = instrument(model, records)
    print(f"instrumented hyper memories: {n_wrapped}")

    if args.memqa_file:
        sessions = memqa_sessions(args.memqa_file)
    else:
        sessions = ri.load_question_sessions(args.questions_file)
        sessions += synthetic_sessions(args.num_synthetic, args.seed)
    print(f"sessions: {len(sessions)}")

    n_commits = 0
    with torch.no_grad():
        for sess in sessions:
            model.reset()
            mode = sess.get("commit_mode", "exchange")
            for prompt in sess["prompts"]:
                ri.commit_turn(
                    model, tokenizer, device,
                    prompt=prompt, response=scripted_reply(prompt),
                    commit_mode=mode,
                )
                n_commits += 1
    print(f"commits: {n_commits}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = summarize(records)
    with (out / "gate_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        raw_dir / "gates_raw.npz",
        **{f"{b}__{m}": np.concatenate(records[b][m]) for b in records for m in METRICS if records[b][m]},
    )

    config = {
        "checkpoint_path": args.checkpoint_path,
        "questions_file": None if args.memqa_file else args.questions_file,
        "memqa_file": args.memqa_file,
        "num_synthetic": 0 if args.memqa_file else args.num_synthetic,
        "num_sessions": len(sessions),
        "num_commits": n_commits,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": args.device,
        "commit_mode": "user (memqa)" if args.memqa_file else "exchange",
        "assistant_replies": "scripted (not generated)",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"wrote {out / 'gate_summary.csv'}")


if __name__ == "__main__":
    main()
