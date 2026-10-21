"""E025 显存诊断：峰值显存 = a + b·(总 token 数) + c·(chunk 数)。

做法（只前向与反向，不更新参数，不改训练代码）：
  - 配置与训练一致：K1_full、readout abs、reinit_memory、骨干冻结。
  - 文本取自训练集中 token 数最多的 episode 的记忆 chunk，拼接后截取前 T 个 token，切成 n 段。
  - 固定 n=4，改变 T（1000、2000、3000）；固定 T=2000，改变 n（1、2、4、8）。
  - 每个点：reset 峰值统计 → episode 前向 + 反向 → 记录 max_memory_allocated。
  - 之后用最小二乘拟合 a、b、c。
用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E025_G1e_train/diag_chunks.py --out experiments/E025_G1e_train/diag_chunks
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri
from holistic.banked_metis import install_banked
from train.train_utils import freeze_backbone


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


T_MOD = _load("e25train", "experiments/E025_G1e_train/train.py")
E13 = T_MOD.E13


def split_text(tok, text, T, n):
    """把文本截成 T 个 token，再切成 n 段（每段解码回文本，由 episode_loss 重新编码）。"""
    ids = tok.encode(text, add_special_tokens=False)[:T]
    step = len(ids) // n
    return [tok.decode(ids[i * step:(i + 1) * step if i < n - 1 else len(ids)]) for i in range(n)]


def peak_for(model, tok, dev, chunks):
    ep = {"memory_chunks": chunks, "question": "What is discussed?", "answer": "the conversation"}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    loss = E13.episode_loss(model, tok, dev, ep)
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 2**30
    for p in model.parameters():
        p.grad = None
    model.reset()
    return peak, float(loss.item())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    fit, _ = T_MOD.load_qa()
    longest = max(fit, key=lambda e: sum(len(c) for c in e["memory_chunks"]))
    text = "\n".join(longest["memory_chunks"])

    load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load)
    freeze_backbone(model)
    install_banked(model, lam=0.9, readout_den="abs")
    E13.reinit_memory(model)
    for name, p in model.named_parameters():
        if any(g in name for g in E13.GATE_NAMES):
            p.requires_grad = False

    base_mem = torch.cuda.memory_allocated() / 2**30
    rows = []
    for T, n in [(1000, 4), (2000, 4), (3000, 4), (2000, 1), (2000, 2), (2000, 8)]:
        chunks = split_text(tok, text, T, n)
        peak, loss = peak_for(model, tok, dev, chunks)
        rows.append({"T": T, "n": n, "peak_gb": peak, "loss": loss})
        print(json.dumps(rows[-1]), flush=True)

    import numpy as np
    X = np.array([[1.0, r["T"], r["n"]] for r in rows])
    y = np.array([r["peak_gb"] for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit_res = y - X @ coef
    result = {"base_allocated_gb_after_load": base_mem, "rows": rows,
              "fit": {"a_gb": coef[0], "b_gb_per_token": coef[1], "c_gb_per_chunk": coef[2]},
              "max_abs_residual_gb": float(np.abs(fit_res).max())}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print("fit", json.dumps(result["fit"]), "residual", result["max_abs_residual_gb"], flush=True)


if __name__ == "__main__":
    main()
