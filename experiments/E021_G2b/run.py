"""E021（门控 G2b）：未中心化分库接入真实 Metis，闭环删除的组合测试。设计：docs/design_G2b.md（预先登记）。

结构（等总状态）：每层 K=4 库，键 256 维（PCA 基 Q 由 seed 9000 前 10 条 episode 拟合），值 1024 维，RLS λ=0.9。
路由：argmax_b (k̂·c_b)，k̂ 为单位化完整键，c_b 为固定随机中心（位于 PCA 子空间），不中心化。
删除：A 降秩（每库，权重 λ^age，近似）；B 快照加下游重放（精确）；参考：零权重被删 chunk 从头重放。
数据：seed 9400，20 条新 episode；被删 chunk = 包含答案的最后一个 chunk。
前提：G2b-0 回归已通过（E021/regression_G2b0.py）。

用法（仓库根目录）：PYTHONPATH=.:src python experiments/E021_G2b/run.py --out experiments/E021_G2b
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri  # noqa: E402
import holistic.rls_metis as R  # noqa: E402
import holistic.banked_metis as B  # noqa: E402
from holistic.synthetic_memory_data import make_split  # noqa: E402
from train.train_utils import freeze_backbone  # noqa: E402

_s15 = importlib.util.spec_from_file_location("e15", ROOT / "experiments/E015_zero_shot_deletion/run.py")
E15 = importlib.util.module_from_spec(_s15)
_s15.loader.exec_module(E15)  # 仅用于 answer_nll

_s18 = importlib.util.spec_from_file_location("e18", ROOT / "experiments/E018_G1_equal_state/run.py")
E18 = importlib.util.module_from_spec(_s18)
_s18.loader.exec_module(E18)  # 仅用于 PCA 基的拟合（record）

K, KD, LAM, KV = 4, 256, 0.9, 1024


def run_chunks(model, tok, dev, chunks, start, zero_idx=None, snapshots=None):
    for i in range(start, len(chunks)):
        if snapshots is not None:
            snapshots[i] = B.get_bank_states(model, dev)
        B.ZERO["on"] = (i == zero_idx)
        ids, mask = ri.encode_messages(tok, [{"role": "user", "content": chunks[i]}], dev, add_generation_prompt=False)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)
    B.ZERO["on"] = False


def downdate_banks(states_full, log, n_layers, c, age):
    """对被删 chunk c 的每一层、每个库做信息形式降秩（权重 λ^age * mask）。log 的索引为 (commit*n_layers + l)*K + b。"""
    out = []
    for l in range(n_layers):
        banks_new = []
        for b in range(K):
            M, P, z = states_full[l][b]
            _, kp, v, mask, lam = log[(c * n_layers + l) * K + b]
            Md, Pd, zd = M[0].double(), P[0].double(), z[0].double()
            cc = mask * (lam ** age)
            Lam = torch.linalg.inv(Pd)
            bvec = Lam @ Md
            Kp = kp.to(Pd.device)
            V = v.to(Pd.device)
            cc = cc.to(Pd.device)
            Lam_new = Lam - (Kp.T * cc) @ Kp
            b_new = bvec - (Kp.T * cc) @ V
            z_new = zd - (Kp.T * cc).sum(dim=1, keepdim=True)
            P_new = torch.linalg.inv(Lam_new)
            M_new = P_new @ b_new
            banks_new.append([M_new[None].float(), P_new[None].float(), z_new[None].float()])
        out.append(banks_new)
    return out


def rel_err(states_a, states_r):
    errs = []
    for la, lr in zip(states_a, states_r):
        for (Ma, _, _), (Mr, _, _) in zip(la, lr):
            errs.append(((Ma - Mr).norm() / Mr.norm().clamp_min(1e-12)).item())
    return max(errs)


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

    # PCA 基：与 E018/E019 相同（seed 9000 前 10 条），键来自 RLS 记录（单位化后）
    R.install_rls(model, lam=LAM)
    fit = make_split("val", 10, seed=9000)
    fit_data = E18.record(model, tok, dev, fit, [3])
    R.uninstall_rls(model)
    seqs = fit_data[3]
    X = torch.cat([F.normalize(k, dim=-1) for seq in seqs for (k, _, _) in seq], 0)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    Q = Vh[:KD].T.contiguous().to(dev)                      # (1024, 256)
    g = torch.Generator().manual_seed(1000)
    centers = F.normalize(torch.randn(K, KD, generator=g) @ Q.T.cpu(), dim=-1).to(dev)   # (K, 1024)

    B.install_banked(model, Q=Q, centers=centers, lam=LAM, K=K, kd=KD)
    n_layers = sum(1 for b in model.model.metis_blocks if b.local_memory is not None)
    eps = make_split("val", args.n, seed=9400)

    rows = []
    loads_all = []
    for ep in eps:
        chunks = ep["memory_chunks"]
        N = len(chunks)
        c = max(i for i, ch in enumerate(chunks) if ep["answer"] in ch)
        with torch.no_grad():
            model.reset()
            B.CAPTURE["on"], B.CAPTURE["log"] = True, []
            snaps = {}
            run_chunks(model, tok, dev, chunks, 0, snapshots=snaps)
            log = B.CAPTURE["log"]
            B.CAPTURE["on"], B.CAPTURE["log"] = False, None
            full = B.get_bank_states(model, dev)
            nll_full = E15.answer_nll(model, tok, dev, ep)
            # 负载（按写入 token）
            for commit in range(N):
                cnt = [float(log[(commit * n_layers + 0) * K + b][3].sum().item()) for b in range(K)]
                loads_all.append(cnt)
            # 参考
            model.reset()
            run_chunks(model, tok, dev, chunks, 0, zero_idx=c)
            ref = B.get_bank_states(model, dev)
            nll_ref = E15.answer_nll(model, tok, dev, ep)
            # A：降秩
            A_states = downdate_banks(full, log, n_layers, c, age=N - 1 - c)
            B.set_bank_states(model, A_states)
            nll_A = E15.answer_nll(model, tok, dev, ep)
            errA = rel_err(A_states, ref)
            # B：快照加下游重放
            B.set_bank_states(model, snaps[c])
            run_chunks(model, tok, dev, chunks, c, zero_idx=c)
            Bst = B.get_bank_states(model, dev)
            nll_B = E15.answer_nll(model, tok, dev, ep)
            errB = rel_err(Bst, ref)
            # 无记忆
            model.reset()
            nll_nomem = E15.answer_nll(model, tok, dev, ep)
        rows.append({"episode_id": ep["episode_id"], "c_del": c, "N": N, "replayed_chunks_B": N - c,
                     "state_err_A": errA, "state_err_B": errB, "nll_full": nll_full, "nll_ref": nll_ref,
                     "nll_A": nll_A, "nll_B": nll_B, "nll_no_memory": nll_nomem})
        print(json.dumps({k: rows[-1][k] for k in ["episode_id", "c_del", "state_err_A", "state_err_B", "nll_ref", "nll_B"]}),
              flush=True)

    B.uninstall_banked(model)
    tot = np.sum(np.array(loads_all), axis=0)
    p = tot / tot.sum()
    load = {"share_per_bank": p.tolist(), "max_share": float(p.max()),
            "norm_entropy": float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / np.log(K))}

    def paired(key):
        d = np.array([r[key] - r["nll_ref"] for r in rows])
        rng = np.random.default_rng(0)
        means = np.array([rng.choice(d, size=d.size, replace=True).mean() for _ in range(5000)])
        return {"mean_diff": float(d.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}

    summary = {
        "n_episodes": len(rows), "K": K, "kd": KD, "lam": LAM,
        "state_err_B_max": max(r["state_err_B"] for r in rows),
        "state_err_A_mean": float(np.mean([r["state_err_A"] for r in rows])),
        "B_minus_ref": paired("nll_B"), "A_minus_ref": paired("nll_A"),
        "full_minus_ref": paired("nll_full"), "no_memory_minus_ref": paired("nll_no_memory"),
        "mean_replayed_chunks_B": float(np.mean([r["replayed_chunks_B"] for r in rows])),
        "mean_N": float(np.mean([r["N"] for r in rows])),
        "load": load,
    }
    ci = summary["B_minus_ref"]["ci95"]
    summary["G2b_passed_given_G2b0"] = bool(summary["state_err_B_max"] < 1e-3 and -0.1 <= ci[0] and ci[1] <= 0.1)
    (out / "result.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
