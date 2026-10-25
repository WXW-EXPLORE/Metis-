"""E025（门控 G1e）：训练后的读出下，单库 / 数据自适应分库 / 随机分库在真实对话（LoCoMo）上的对照。
设计：docs/design_G1e.md（预先登记）。准备步骤：prep.py（PCA 基与路由中心，只用 fit 对话）。

臂（单变量：只改分库结构，其余与 E013 相同）：
  K1_full     单库，RLS，1024×1024（install_banked K=1，即 install_rls）
  K4_CL_cons  4 库，每库 256×1024，路由中心为竞争学习原型（β=1，fit 数据得到，训练时固定）
  K4_random   4 库，路由中心为固定随机中心（同 E024 口径）
训练：骨干冻结，记忆参数从头初始化（E013 的 reinit_memory），门控不参与（与 E013 的 rls 变体相同）。
每个训练 episode = 一个 LoCoMo QA 行：其会话的记忆 chunk，然后对问题的答案计算 NLL（梯度经记忆写入回传）。
评测：test 对话的前 n_test 个 QA 行，答案 NLL（逐条保存，供配对分析）。

用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E025_G1e_train/train.py --arm K4_CL_cons --seed 0 --out experiments/E025_G1e_train/runs/K4_CL_cons_s0
"""
import argparse, importlib.util, json, math, random, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri
import holistic.banked_metis as BM
from holistic.banked_metis import install_banked
from train.train_utils import MasterWeightAdamW, freeze_backbone


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


E13 = _load("e13", "experiments/E013_1b_train/train.py")
LOCOMO = ROOT / "eval/data/memqa/locomo_evidence_sessions_tps16_20260626.jsonl"
GATE_NAMES = E13.GATE_NAMES
K, KD = 4, 256
ROUTE = {"on": False, "counts": None}
_orig_route = BM._route


def _counting_route(kn, centers):
    b = _orig_route(kn, centers)
    if ROUTE["on"]:
        ROUTE["counts"] += torch.bincount(b, minlength=ROUTE["counts"].numel()).cpu()
    return b


BM._route = _counting_route


def load_qa():
    """按对话首次出现的顺序划分：前 5 个为 fit，其余为 test。返回 (fit_rows, test_rows)。"""
    rows = [json.loads(l) for l in open(LOCOMO, encoding="utf-8")]
    convs = list(dict.fromkeys(r["source_sample_id"] for r in rows))
    fit_set = set(convs[:5])
    out = {"fit": [], "test": []}
    for r in rows:
        ep = {"episode_id": r["instance_id"], "memory_chunks": [st["content"] for st in r["memory_steps"]],
              "question": r["question"], "answer": str(r["answer"])}
        out["fit" if r["source_sample_id"] in fit_set else "test"].append(ep)
    return out["fit"], out["test"]


@torch.no_grad()
def write_only_counts(model, tok, dev, episodes):
    """只做记忆写入（不含问题），统计路由到各库的 token 数（用于 G1e-c，训练后的真实键）。"""
    ROUTE["counts"] = torch.zeros(K, dtype=torch.long); ROUTE["on"] = True
    try:
        for ep in episodes:
            model.reset()
            for chunk in ep["memory_chunks"]:
                ids, mask = ri.encode_messages(tok, [{"role": "user", "content": chunk}], dev,
                                               add_generation_prompt=False)
                model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)
    finally:
        ROUTE["on"] = False
    c = ROUTE["counts"].float()
    return (c / c.sum()).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", default="/hy-tmp/metis-models/Metis-4B")
    ap.add_argument("--arm", choices=["K1_full", "K4_CL_cons", "K4_random", "GDN"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prototypes", default=str(ROOT / "experiments/E025_G1e_train/prototypes.pt"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--max_train_tokens", type=int, default=None)
    ap.add_argument("--readout_den", choices=["orig", "abs"], default="orig")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)

    fit, test = load_qa()
    load = argparse.Namespace(checkpoint_path=args.checkpoint_path, model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load)
    if args.max_train_tokens is not None:
        # 试错：只从训练集中剔除超长 episode（评测集不变）
        n0 = len(fit)
        fit = [e for e in fit if sum(len(tok.encode(c, add_special_tokens=False)) for c in e["memory_chunks"])
               <= args.max_train_tokens]
        print(f"max_train_tokens={args.max_train_tokens}: fit rows {n0} -> {len(fit)}", flush=True)
    rng = random.Random(1000 + args.seed)
    train_eps = []
    while len(train_eps) < args.steps * args.accum:
        perm = list(fit); rng.shuffle(perm); train_eps += perm
    train_eps = train_eps[: args.steps * args.accum]
    val_eps = test[: args.n_test]
    (out / "data.json").write_text(json.dumps({"n_fit_rows": len(fit), "n_test_rows": len(test),
                                               "n_val": len(val_eps), "seed": args.seed, "arm": args.arm,
                                               "max_train_tokens": args.max_train_tokens, "readout_den": args.readout_den}, indent=1))

    freeze_backbone(model)
    if args.arm == "GDN":
        pass   # 原版 GDN 记忆（不替换），门控参数保持可训练
    elif args.arm == "K1_full":
        install_banked(model, lam=args.lam, readout_den=args.readout_den)
    else:
        proto = torch.load(args.prototypes, weights_only=False)
        centers = proto["C_cl"][args.seed] if args.arm == "K4_CL_cons" else proto["C_rand"][args.seed]
        centers = [c.to(dev) for c in centers]
        Qs = [q.to(dev) for q in proto["Q"]]
        install_banked(model, Q=Qs, centers=centers, lam=args.lam, K=K, kd=KD, readout_den=args.readout_den)
    E13.reinit_memory(model)
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if args.arm != "GDN" and any(g in name for g in GATE_NAMES):
            p.requires_grad = False
            continue
        params.append(p)
    n_params = sum(p.numel() for p in params)
    print(f"arm={args.arm} seed={args.seed} trainable tensors={len(params)} params={n_params:,}", flush=True)
    opt = MasterWeightAdamW(params, lr=args.lr, weight_decay=0.0)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    log, t0 = [], time.time()
    init_eval = E13.evaluate(model, tok, dev, val_eps)
    log.append({"step": 0, "val_nll": init_eval["nll_mean"]})
    print(f"step 0 val nll={init_eval['nll_mean']:.4f}", flush=True)
    ep_idx = 0
    model.train()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1)
        opt.zero_grad(set_to_none=True)
        losses = []
        for _ in range(args.accum):
            loss = E13.episode_loss(model, tok, dev, train_eps[ep_idx]); ep_idx += 1
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
            ev = E13.evaluate(model, tok, dev, val_eps)
            rec["val_nll"] = ev["nll_mean"]; rec["val_n_nonfinite"] = ev["n_nonfinite"]
            print(f"step {step} val nll={ev['nll_mean']:.4f}", flush=True)
        log.append(rec)
        print(f"step {step} train nll={rec['train_nll']:.4f} gnorm={rec['grad_norm']:.3f} t={rec['elapsed_s']:.0f}s",
              flush=True)
        (out / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")

    pe = []
    final = E13.evaluate(model, tok, dev, val_eps, per_episode=pe)
    (out / "val_episode_nll_final.json").write_text(json.dumps(pe, indent=1))
    shares = write_only_counts(model, tok, dev, val_eps) if args.arm != "K1_full" else [1.0]
    summary = {"arm": args.arm, "seed": args.seed, "init_val_nll": init_eval["nll_mean"],
               "final_val_nll": final["nll_mean"], "final_n_nonfinite": final["n_nonfinite"],
               "trainable_params": n_params, "steps": args.steps, "accum": args.accum, "lr": args.lr,
               "lam": args.lam, "n_val": len(val_eps), "post_train_bank_share": shares,
               "max_bank_share": max(shares), "elapsed_s": time.time() - t0}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("summary", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
