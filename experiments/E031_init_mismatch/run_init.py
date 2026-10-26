"""E031（B 的下一步）：初始不匹配诊断，不训练。设计：docs/design_E031_init_mismatch.md

复现 E025 训练脚本的初始化顺序（seed → 加载 → 安装 → reinit_memory），在 pilot 划分前 100 个 episode 上
评测 step-0 NLL；可选把记忆读出乘以 s（s=0 即无记忆地板）；记录边界层的记忆/注意力范数比。
"""
import argparse, importlib.util, json, math, random, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import run_inference as ri
from train.train_utils import freeze_backbone


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


T = _load("e25_train", "experiments/E025_G1e_train/train.py")
E13 = T.E13


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["GDN", "RLS"], required=True)
    ap.add_argument("--readout_den", choices=["orig", "abs"], default="abs")
    ap.add_argument("--read_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--mn_scale", type=float, default=None, help="E031b：有效 mem_norm 缩放 1+weight，设为 s 即 weight=s-1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    fit, test = T.load_qa(pilot=True)          # 只用 pilot 评测，不读测试集
    val_eps = test[: args.n_eval]
    load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load)
    freeze_backbone(model)
    if args.arm == "RLS":
        T.install_banked(model, lam=0.9, readout_den=args.readout_den)
    E13.reinit_memory(model)
    if args.mn_scale is not None:
        for block in model.model.metis_blocks:
            if block.local_memory is not None:
                block.mem_norm.weight.data.fill_(args.mn_scale - 1.0)
    if args.read_scale != 1.0:
        for block in model.model.metis_blocks:
            lm = block.local_memory
            if lm is None:
                continue
            orig, s = lm.read, args.read_scale
            lm.read = (lambda q, _o=orig, _s=s: _o(q) * _s)

    model.eval()
    nlls, ratios = [], {}
    with torch.no_grad():
        for ep in val_eps:
            nlls.append(float(E13.episode_loss(model, tok, dev, ep).item()))
            for i, block in enumerate(model.model.metis_blocks):
                r = getattr(block, "last_memory_attention_norm_ratio", None)
                if r is not None and math.isfinite(r):
                    ratios.setdefault(str(i), []).append(r)
    finite = [x for x in nlls if math.isfinite(x)]
    res = {"arm": args.arm, "readout_den": args.readout_den, "read_scale": args.read_scale, "mn_scale": args.mn_scale, "seed": args.seed,
           "n": len(finite), "n_nonfinite": len(nlls) - len(finite),
           "nll_mean": sum(finite) / len(finite),
           "ratio_mem_over_attn": {k: {"mean": sum(v) / len(v), "max": max(v), "n": len(v)} for k, v in ratios.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("arm", "readout_den", "read_scale", "nll_mean", "n_nonfinite")}), flush=True)


if __name__ == "__main__":
    main()
