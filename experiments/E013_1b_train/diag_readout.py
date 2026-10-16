"""E013 诊断：原版 GDN 与 RLS 的记忆读出尺度（同一 episode，同一预训练骨干）。

输出每层读出的 RMS，以及 mem_norm 之后的 RMS，并比较有记忆与无记忆时的初始损失。
用法（仓库根目录）：PYTHONPATH=.:src python experiments/E013_1b_train/diag_readout.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri  # noqa: E402
from holistic.rls_metis import install_rls, uninstall_rls  # noqa: E402
from holistic.synthetic_memory_data import make_split  # noqa: E402
from train.train_utils import freeze_backbone  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("e013_train", Path(__file__).parent / "train.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
episode_loss = _mod.episode_loss


def read_rms(model, tokenizer, device, ep, collect):
    model.reset()
    for chunk in ep["memory_chunks"]:
        ids, mask = ri.encode_messages(tokenizer, [{"role": "user", "content": chunk}], device,
                                       add_generation_prompt=False)
        model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)
    stats = {}
    originals = {}
    for i, block in enumerate(model.model.metis_blocks):
        lm = block.local_memory
        if lm is None:
            continue
        orig = lm.read
        originals[i] = (lm, orig)

        def wrapped(q, _i=i, _orig=orig):
            out = _orig(q)
            stats.setdefault(_i, []).append(float(out.float().pow(2).mean().sqrt().item()))
            return out

        lm.read = wrapped
    prompt_ids, _ = ri.encode_messages(tokenizer, [{"role": "user", "content": ep["question"]}], device,
                                       add_generation_prompt=True)
    with torch.no_grad():
        model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), use_cache=False, logits_to_keep=1)
    for lm, orig in originals.values():
        lm.read = orig
    collect.update({k: sum(v) / len(v) for k, v in stats.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", default="/hy-tmp/metis-models/Metis-4B")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    load = argparse.Namespace(checkpoint_path=args.checkpoint_path, model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tokenizer, device = ri.load_model_and_tokenizer(load)
    freeze_backbone(model)
    eps = make_split("val", args.n, seed=9000)
    result = {}
    with torch.no_grad():
        for label in ["baseline", "rls"]:
            if label == "rls":
                install_rls(model, lam=0.9)
            rms_all = {}
            losses = []
            for ep in eps:
                col = {}
                read_rms(model, tokenizer, device, ep, col)
                for k, v in col.items():
                    rms_all.setdefault(k, []).append(v)
                losses.append(float(episode_loss(model, tokenizer, device, ep).item()))
            result[label] = {
                "read_rms_by_layer": {str(k): sum(v) / len(v) for k, v in sorted(rms_all.items())},
                "val_nll": sum(losses) / len(losses),
            }
            print(label, json.dumps(result[label], indent=1), flush=True)
            if label == "rls":
                uninstall_rls(model)
        # 无记忆：不 commit，直接查询
        nomem = []
        for ep in eps:
            model.reset()
            prompt_ids, _ = ri.encode_messages(tokenizer, [{"role": "user", "content": ep["question"]}], device,
                                               add_generation_prompt=True)
            ans = tokenizer.encode(ep["answer"], add_special_tokens=False)
            ans_t = torch.tensor([ans], device=device, dtype=prompt_ids.dtype)
            ids = torch.cat([prompt_ids, ans_t], 1)
            labels = torch.full_like(ids, -100)
            labels[:, prompt_ids.size(1):] = ans_t
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
            nomem.append(float(out.loss.item()))
        result["no_memory"] = {"val_nll": sum(nomem) / len(nomem)}
        print("no_memory", result["no_memory"], flush=True)
    out = ROOT / "experiments/E013_1b_train/diag_readout.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
