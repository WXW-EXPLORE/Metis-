"""E025（门控 G1e）准备步骤：为每个记忆层计算 PCA 基与路由中心（只读，不训练）。设计：docs/design_G1e.md。

  - 键：预训练 Metis-4B 在 install_rls 下记录的真实写入键（与 E024 相同口径），按对话拆分：
        fit = 前 5 个对话（拟合 PCA 基与竞争学习原型），test = 其余对话（只用于报告路由负载）。
  - 每层：Q（1024×256，fit 键的前 256 个右奇异向量）；
        K4_CL（β=1，竞争学习，种子 2000+s）；K4_rand（种子 1000+s，256 维 PCA 空间内随机中心）。
  - 输出：prototypes.pt（训练脚本使用）与 prep_stats.json（fit/test 负载，按种子、按层，不取平均）。

用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E025_G1e_train/prep.py --out experiments/E025_G1e_train
"""
import argparse, importlib.util, json, sys
from collections import OrderedDict
from pathlib import Path
import torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri
import holistic.rls_metis as R
from train.train_utils import freeze_backbone


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


E18 = _load("e18", "experiments/E018_G1_equal_state/run.py")
E23 = _load("e23", "experiments/E023_G1c/run.py")
E24 = _load("e24", "experiments/E024_G1d/run.py")
K, LAM, SEEDS = 4, 0.9, [0, 1, 2]


@torch.no_grad()
def record_keys(model, tok, dev, eps, blocks):
    """与 E18.record 相同的写入过程，但只保存各记忆层的键（bf16，CPU）。
    E18.record 同时保存 v 与权重（fp32），在 67 GB 的容器内存上限下会被 OOM 杀掉（见 E025 README）。"""
    out = {b: [] for b in blocks}
    for ep in eps:
        model.reset()
        E18.CAP["list"] = []
        for chunk in ep["memory_chunks"]:
            ids, mask = ri.encode_messages(tok, [{"role": "user", "content": chunk}], dev, add_generation_prompt=False)
            model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)
        cap = E18.CAP["list"]
        E18.CAP["list"] = None
        nl = len(cap) // len(ep["memory_chunks"])
        for li, b in enumerate(blocks):
            out[b].append([F.normalize(cap[c * nl + li][0], dim=-1).to(torch.bfloat16)
                           for c in range(len(ep["memory_chunks"]))])
        del cap
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--readout_den", choices=["orig", "abs"], default="orig"); args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    eps = E24.load_locomo_episodes()
    convs = list(OrderedDict.fromkeys(e["conv"] for e in eps))
    fit = [e for e in eps if e["conv"] in set(convs[:5])]
    test = [e for e in eps if e["conv"] in set(convs[5:])]
    load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load); model.eval(); freeze_backbone(model)
    blocks = [i for i, b in enumerate(model.model.metis_blocks) if b.local_memory is not None]
    R.install_rls(model, lam=LAM, readout_den=args.readout_den)
    fit_d = record_keys(model, tok, dev, fit, blocks)
    test_d = record_keys(model, tok, dev, test, blocks)
    R.uninstall_rls(model)

    Qs, C_cl, C_rand = [], {s: [] for s in SEEDS}, {s: [] for s in SEEDS}
    stats = {"n_fit_episodes": len(fit), "n_test_episodes": len(test), "blocks": blocks, "layers": {}}
    for li, blk in enumerate(blocks):
        Xf = torch.cat([k.float() for seq in fit_d[blk] for k in seq], 0)
        Xt = torch.cat([k.float() for seq in test_d[blk] for k in seq], 0)
        _, _, Vh = torch.linalg.svd(Xf, full_matrices=False)
        Q = Vh[:256].T.contiguous(); Qs.append(Q)
        st = {"n_fit_keys": int(Xf.shape[0]), "n_test_keys": int(Xt.shape[0]), "cl": {}, "rand": {}}
        for s in SEEDS:
            C, u = E23.train_prototypes(Xf, 1.0, seed=2000 + s)
            C_cl[s].append(C)
            b = E23.route_scores(Xt, C, u, 1.0)
            st["cl"][str(s)] = torch.bincount(b, minlength=K).float().div(b.numel()).tolist()
            g = torch.Generator().manual_seed(1000 + s)
            Cr = F.normalize(torch.randn(K, 256, generator=g) @ Q.T, dim=-1)
            C_rand[s].append(Cr)
            b = (Xt @ Cr.T).argmax(-1)
            st["rand"][str(s)] = torch.bincount(b, minlength=K).float().div(b.numel()).tolist()
        stats["layers"][f"block{blk}"] = st
        print(f"block {blk}: fit={Xf.shape[0]} test={Xt.shape[0]} "
              f"CL max share={[round(max(st['cl'][str(s)]), 3) for s in SEEDS]} "
              f"rand max share={[round(max(st['rand'][str(s)]), 3) for s in SEEDS]}", flush=True)
        del Xf, Xt

    torch.save({"blocks": blocks, "Q": Qs, "C_cl": C_cl, "C_rand": C_rand}, out / "prototypes.pt")
    (out / "prep_stats.json").write_text(json.dumps(stats, indent=1))
    print("saved", out / "prototypes.pt", flush=True)


if __name__ == "__main__":
    main()
