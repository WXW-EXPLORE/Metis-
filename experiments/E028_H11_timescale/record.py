"""E028（H11）：（保存全部记忆层）录制真实写入键与值（预训练 Metis-4B，RLS 装入，abs 读出；开环，与 G1 相同口径）。

每个 episode 的每个 chunk、每个记忆层：保存被选中的 token（权重 w>0）的单位键 kn 与值 v（bf16，CPU）。
键的单位化：kn = normalize(kappa·sqrt(kv))，kappa 为 rls 写入时的键（已除以 sqrt(kv)）。

划分（设计 §10）：fit = 前 3 个对话；tune = 第 4–5 个对话；test = 第 6–10 个对话。

用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E028_H11_timescale/record.py --out /hy-tmp/e028_keys.pt
"""
import argparse, importlib.util, sys
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


E24 = _load("e24", "experiments/E024_G1d/run.py")
CAP = {"list": None}
_orig = R.rls_commit_update


def _capture(M, P, z, kappa, v, w, lam, p_max=None):
    if CAP["list"] is not None:
        CAP["list"].append((kappa.detach().float()[0].cpu(), v.detach().float()[0].cpu(), w.detach().float()[0].cpu()))
    return _orig(M, P, z, kappa, v, w, lam, p_max=p_max)


R.rls_commit_update = _capture


def split_of(eps):
    convs = list(OrderedDict.fromkeys(e["conv"] for e in eps))
    fit_c, tune_c, test_c = set(convs[:3]), set(convs[3:5]), set(convs[5:])
    out = {"fit": [], "tune": [], "test": []}
    for e in eps:
        for name, cs in [("fit", fit_c), ("tune", tune_c), ("test", test_c)]:
            if e["conv"] in cs:
                out[name].append(e)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--n_max", type=int, default=0, help="调试用：每个划分最多取前 n 个 episode（0 为全部）")
    args = ap.parse_args()
    eps = E24.load_locomo_episodes()
    sp = split_of(eps)
    if args.n_max:
        sp = {k: v[: args.n_max] for k, v in sp.items()}
    load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load)
    model.eval(); freeze_backbone(model)
    R.install_rls(model, lam=0.9, readout_den="abs")
    blocks = [i for i, b in enumerate(model.model.metis_blocks) if b.local_memory is not None]
    kv = model.model.metis_blocks[blocks[0]].local_memory.kv_dim
    KEEP = list(range(len(blocks)))   # 全部记忆层（磁盘已扩容，保存全部以便后续分析）
    data = {}
    with torch.no_grad():
        for split, items in sp.items():
            data[split] = []
            for ep in items:
                model.reset()
                CAP["list"] = []
                chunks_out = []
                for chunk in ep["memory_chunks"]:
                    ids, mask = ri.encode_messages(tok, [{"role": "user", "content": chunk}], dev, add_generation_prompt=False)
                    model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)
                cap = CAP["list"]; CAP["list"] = None
                nl = len(cap) // len(ep["memory_chunks"])
                for c in range(len(ep["memory_chunks"])):
                    per_layer = []
                    for li in KEEP:
                        kappa, v, w = cap[c * nl + li]
                        sel = w > 0
                        kn = F.normalize(kappa[sel] * (kv ** 0.5), dim=-1)
                        per_layer.append((kn.to(torch.bfloat16), v[sel].to(torch.bfloat16)))
                    chunks_out.append(per_layer)
                data[split].append({"episode_id": ep["episode_id"], "chunks": chunks_out})
            print(split, len(data[split]), "episodes", flush=True)
    torch.save({"blocks": blocks, "kv": kv, "data": data}, args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
