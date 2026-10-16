"""E020（门控 G2）：闭环删除的组合测试（真实 Metis-4B，RLS 单库，λ=0.9）。设计：docs/design_G2.md（预先登记）。

机制：
  A 降秩（近似）：对被删 chunk 用记录的写入输入做信息形式降秩（权重 λ^age）。
  B 快照加下游重放（精确）：恢复被删 chunk 之前的快照，以零权重重放该 chunk，再重放其后的 chunk。
  参考：整个会话以零权重的被删 chunk 从头重放（gold）。
数据：seed 9200，20 条 episode；被删 chunk = 包含答案的最后一个 chunk。

用法（仓库根目录）：PYTHONPATH=.:src python experiments/E020_G2_closed_loop_deletion/run.py --out experiments/E020_G2_closed_loop_deletion
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri  # noqa: E402
import holistic.rls_metis as R  # noqa: E402
from holistic.synthetic_memory_data import make_split  # noqa: E402
from train.train_utils import freeze_backbone  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("e15", ROOT / "experiments/E015_zero_shot_deletion/run.py")
E15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E15)

CAP = {"list": None, "zero_now": False}
_orig = R.rls_commit_update


def _wrapped(M, P, z, kappa, v, w, lam, p_max=None):
    if CAP["zero_now"]:
        w = w * 0
    if CAP["list"] is not None:
        CAP["list"].append((kappa.detach().double(), v.detach().double(), w.detach().double(), float(lam)))
    return _orig(M, P, z, kappa, v, w, lam, p_max=p_max)


R.rls_commit_update = _wrapped
N_LAYERS = 8


def rls_layers(model):
    return [b.local_memory for b in model.model.metis_blocks if b.local_memory is not None]


def get_states(model, dev):
    out = []
    for lm in rls_layers(model):
        if getattr(lm, "_state", None) is None:
            R._ensure_rls_state(lm, 1, lm.kv_dim, dev)
        out.append((lm._state.detach().clone(), lm._P.detach().clone(), lm._key_state.detach().clone()))
    return out


def set_states(model, states):
    for lm, (M, P, z) in zip(rls_layers(model), states):
        lm._state, lm._P, lm._key_state = M.clone(), P.clone(), z.clone()


def run_chunks(model, tok, dev, chunks, start, zero_idx=None, snapshots=None, capture=None):
    for i in range(start, len(chunks)):
        if snapshots is not None:
            snapshots[i] = get_states(model, dev)
        CAP["zero_now"] = (i == zero_idx)
        CAP["list"] = capture
        ids, mask = ri.encode_messages(tok, [{"role": "user", "content": chunks[i]}], dev,
                                       add_generation_prompt=False)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)
    CAP["zero_now"] = False
    CAP["list"] = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", default="/hy-tmp/metis-models/Metis-4B")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    load = argparse.Namespace(checkpoint_path=args.checkpoint_path, model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load)
    model.eval()
    freeze_backbone(model)
    R.install_rls(model, lam=0.9)
    eps = make_split("val", args.n, seed=9200)

    rows = []
    for ep in eps:
        chunks = ep["memory_chunks"]
        N = len(chunks)
        c = max(i for i, ch in enumerate(chunks) if ep["answer"] in ch)
        with torch.no_grad():
            # 完整会话（记录输入与快照）
            model.reset()
            capture, snaps = [], {}
            run_chunks(model, tok, dev, chunks, 0, snapshots=snaps, capture=capture)
            full = get_states(model, dev)
            nll_full = E15.answer_nll(model, tok, dev, ep)
            # 参考：从头以零权重重放被删 chunk
            model.reset()
            run_chunks(model, tok, dev, chunks, 0, zero_idx=c)
            ref = get_states(model, dev)
            nll_ref = E15.answer_nll(model, tok, dev, ep)
            # A：降秩
            A_states = E15.delete_chunk_downdate(full, capture, c, age=N - 1 - c)
            set_states(model, A_states)
            nll_A = E15.answer_nll(model, tok, dev, ep)
            errA = max(((a[0] - r[0]).norm() / r[0].norm().clamp_min(1e-12)).item() for a, r in zip(A_states, ref))
            # B：快照加下游重放
            set_states(model, snaps[c])
            run_chunks(model, tok, dev, chunks, c, zero_idx=c)
            B_states = get_states(model, dev)
            nll_B = E15.answer_nll(model, tok, dev, ep)
            errB = max(((b[0] - r[0]).norm() / r[0].norm().clamp_min(1e-12)).item() for b, r in zip(B_states, ref))
            # 无记忆
            model.reset()
            nll_nomem = E15.answer_nll(model, tok, dev, ep)
        rows.append({"episode_id": ep["episode_id"], "c_del": c, "N": N, "replayed_chunks_B": N - c,
                     "state_err_A": errA, "state_err_B": errB,
                     "nll_full": nll_full, "nll_ref": nll_ref, "nll_A": nll_A, "nll_B": nll_B,
                     "nll_no_memory": nll_nomem})
        print(json.dumps({k: rows[-1][k] for k in ["episode_id", "c_del", "state_err_A", "state_err_B", "nll_ref", "nll_B"]}),
              flush=True)

    R.uninstall_rls(model)

    def paired(key):
        d = np.array([r[key] - r["nll_ref"] for r in rows])
        rng = np.random.default_rng(0)
        means = np.array([rng.choice(d, size=d.size, replace=True).mean() for _ in range(5000)])
        return {"mean_diff": float(d.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}

    summary = {
        "n_episodes": len(rows),
        "state_err_B_max": max(r["state_err_B"] for r in rows),
        "state_err_A_mean": float(np.mean([r["state_err_A"] for r in rows])),
        "B_minus_ref": paired("nll_B"),
        "A_minus_ref": paired("nll_A"),
        "full_minus_ref": paired("nll_full"),
        "no_memory_minus_ref": paired("nll_no_memory"),
        "mean_replayed_chunks_B": float(np.mean([r["replayed_chunks_B"] for r in rows])),
        "mean_N": float(np.mean([r["N"] for r in rows])),
    }
    ci = summary["B_minus_ref"]["ci95"]
    passed = bool(summary["state_err_B_max"] < 1e-3 and -0.1 <= ci[0] and ci[1] <= 0.1)
    summary["G2_passed"] = passed
    (out / "result.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
