"""E026（门控 G2c）：分库 + 快照重放删除的组合有效性（真实模型，零样本）。设计：docs/design_G2c.md。

只检验组合是否冲突，不评价效果：
  H18a 删除精确：快照重放（跳过被删 chunk）后的状态 与 "从未写入该 chunk" 的参考状态 的相对误差。
  H18b 不误伤：删除前后，其余 chunk 的自检索余弦之差 D = after − before（配对，bootstrap 95% 区间）。
  H18c 负载：每次写入（每个 chunk）中，最大库占比的中位数（每个种子）。

自检索：用写入时捕获的键 kp（查询）与值 v，在给定状态上读出 kp·M/(kp·z+1)，与 v 求余弦（与 E018 的口径一致，
但直接使用写入时的键值，不额外前向）。

臂：K1_full（单库，banked 代码路径，K=1，kd=1024）与 K4_CL_cons（4 库，学习的原型，kd=256）。读出 abs。
回归 G2c-0：同一 episode 上，K1 的 banked 路径（orig 读出）与 install_rls 路径（orig）的最终状态逐位比较。

用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E026_G2c_composition/run.py --arm K4_CL_cons --seed 0 --out experiments/E026_G2c_composition/runs/K4_CL_cons_s0
    PYTHONPATH=.:src python experiments/E026_G2c_composition/run.py --regression --out experiments/E026_G2c_composition/regression
"""
import argparse, importlib.util, json, sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri
import holistic.banked_metis as B
import holistic.rls_metis as R
from holistic.banked_metis import install_banked, uninstall_banked
from train.train_utils import freeze_backbone


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


E24 = _load("e24", "experiments/E024_G1d/run.py")
PROTO = ROOT / "experiments/E025_G1e_train/abs/prototypes.pt"
K4, KD4, KD1 = 4, 256, 1024
LAM = 0.9
EPS_N = 30


def test_episodes(n=EPS_N):
    eps = E24.load_locomo_episodes()
    convs = list(OrderedDict.fromkeys(e["conv"] for e in eps))
    test = [e for e in eps if e["conv"] in set(convs[5:])]
    return test[:n]


def get_states(model, dev):
    return B.get_bank_states(model, dev)


def set_states(model, states):
    B.set_bank_states(model, states)


def run_chunks(model, tok, dev, chunks, start=0, snapshots=None):
    """与 E021 相同：依次写入 chunks[start:]；snapshots[i] 记录写入 chunk i 之前的状态。"""
    for i in range(start, len(chunks)):
        if snapshots is not None:
            snapshots[i] = get_states(model, dev)
        ids, mask = ri.encode_messages(tok, [{"role": "user", "content": chunks[i]}], dev, add_generation_prompt=False)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)


def rel_err_max(states_a, states_r):
    """两组状态（B.get_bank_states 的输出）之间各库 M 的相对误差的最大值。"""
    errs = []
    for la, lr in zip(states_a, states_r):
        if la is None and lr is None:
            continue
        if la is None or lr is None:
            return float("inf")
        for (Ma, _, _), (Mr, _, _) in zip(la, lr):
            errs.append(((Ma - Mr).norm() / Mr.norm().clamp_min(1e-12)).item())
    return max(errs) if errs else 0.0


def self_retrieval(log, states, n_commits, n_layers, K, skip):
    """对除 skip 外的所有已写入 token，计算 kp·M/(kp·z+1) 与 v 的余弦，返回均值。"""
    vals = []
    for commit in range(n_commits):
        if commit == skip:
            continue
        for l in range(n_layers):
            for b in range(K):
                _, kp, v, mask, _lam = log[(commit * n_layers + l) * K + b]
                sel = mask > 0
                if int(sel.sum()) == 0:
                    continue
                M, _P, z = states[l][b]
                kq = kp[sel].double(); vq = v[sel].double()
                out = (kq @ M[0].double()) / (kq @ z[0].double() + 1.0)
                vals.append(F.cosine_similarity(out, vq, dim=-1))
    if not vals:
        return float("nan")
    return float(torch.cat(vals).mean())


def loads_per_commit(log, n_commits, n_layers, K):
    """每次写入（一个 chunk，跨层求和）的各库 token 占比，返回最大库占比列表。"""
    out = []
    for commit in range(n_commits):
        cnt = np.zeros(K)
        for l in range(n_layers):
            for b in range(K):
                cnt[b] += float(log[(commit * n_layers + l) * K + b][3].sum())
        out.append(float(cnt.max() / cnt.sum()) if cnt.sum() > 0 else float("nan"))
    return out


def bootstrap_ci(x, n_boot=5000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_model():
    load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load)
    model.eval(); freeze_backbone(model)
    return model, tok, dev


def install_arm(model, dev, arm, seed, readout_den):
    if arm == "K1_full":
        install_banked(model, lam=LAM, K=1, kd=KD1, _force_banked_path=True, readout_den=readout_den)
    else:
        proto = torch.load(PROTO, weights_only=False)
        Qs = [q.to(dev) for q in proto["Q"]]
        centers = [c.to(dev) for c in proto["C_cl"][seed]]
        install_banked(model, Q=Qs, centers=centers, lam=LAM, K=K4, kd=KD4, readout_den=readout_den)


def regression(model, tok, dev, out):
    """G2c-0：K1 的 banked 路径（orig）与 install_rls（orig）在同一 episode 上的最终状态逐位比较。"""
    eps = test_episodes(3)
    results = []
    for ep in eps:
        chunks = ep["memory_chunks"]
        R.install_rls(model, lam=LAM)
        model.reset()
        run_chunks(model, tok, dev, chunks)
        rls = [(lm._state.clone(), lm._key_state.clone()) for lm in
               (b.local_memory for b in model.model.metis_blocks if b.local_memory is not None)]
        R.uninstall_rls(model)
        install_banked(model, lam=LAM, K=1, kd=KD1, _force_banked_path=True, readout_den="orig")
        model.reset()
        run_chunks(model, tok, dev, chunks)
        bank = get_states(model, dev)
        uninstall_banked(model)
        equal = all(torch.equal(M_r, bk[0][0]) and torch.equal(z_r, bk[0][2])
                    for (M_r, z_r), bk in zip(rls, bank) if bk is not None)
        results.append({"episode_id": ep["episode_id"], "n_chunks": len(chunks), "bitwise_equal": bool(equal)})
        print(ep["episode_id"], "bitwise_equal", equal, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "regression_G2c0.json").write_text(json.dumps(results, indent=1))
    print("G2c-0 passed =", all(r["bitwise_equal"] for r in results), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["K1_full", "K4_CL_cons"], default="K4_CL_cons")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readout_den", choices=["orig", "abs"], default="abs")
    ap.add_argument("--n", type=int, default=EPS_N)
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model, tok, dev = load_model()
    if args.regression:
        regression(model, tok, dev, out)
        return
    install_arm(model, dev, args.arm, args.seed, args.readout_den)
    n_layers = sum(1 for b in model.model.metis_blocks if b.local_memory is not None)
    K = 1 if args.arm == "K1_full" else K4

    rows, loads = [], []
    for ep in test_episodes(args.n):
        chunks = ep["memory_chunks"]
        N = len(chunks)
        if N < 2:
            continue
        with torch.no_grad():
            model.reset()
            B.CAPTURE["on"], B.CAPTURE["log"] = True, []
            snaps = {}
            run_chunks(model, tok, dev, chunks, 0, snapshots=snaps)
            log = B.CAPTURE["log"]
            B.CAPTURE["on"], B.CAPTURE["log"] = False, None
            full = get_states(model, dev)
            loads += loads_per_commit(log, N, n_layers, K)
            for j in range(N):
                # 快照重放删除：恢复写入 chunk j 之前的状态，再写入 j+1..N-1
                set_states(model, snaps[j])
                run_chunks(model, tok, dev, chunks, start=j + 1)
                replay = get_states(model, dev)
                # 参考：从未写入 chunk j
                model.reset()
                run_chunks(model, tok, dev, chunks[:j] + chunks[j + 1:])
                ref = get_states(model, dev)
                sr_before = self_retrieval(log, full, N, n_layers, K, skip=j)
                sr_after = self_retrieval(log, replay, N, n_layers, K, skip=j)
                rows.append({"episode_id": ep["episode_id"], "N": N, "j": j,
                             "err_max": rel_err_max(replay, ref),
                             "sr_before": sr_before, "sr_after": sr_after, "D": sr_after - sr_before})
                print(f"{ep['episode_id']} N={N} j={j} err={rows[-1]['err_max']:.2e} D={rows[-1]['D']:+.5f}",
                      flush=True)
            model.reset()

    errs = [r["err_max"] for r in rows]
    per_ep = OrderedDict()
    for r in rows:
        per_ep.setdefault(r["episode_id"], []).append(r["D"])
    ep_means = [float(np.nanmean(v)) for v in per_ep.values()]
    mean_D, lo, hi = bootstrap_ci(ep_means)
    summary = {
        "arm": args.arm, "seed": args.seed, "readout_den": args.readout_den, "n_episodes": len(per_ep),
        "n_deletions": len(rows), "G2c_a_max_err": float(np.max(errs)) if errs else None,
        "G2c_a_passed": bool(errs and max(errs) < 1e-6),
        "D_mean": mean_D, "D_ci95": [lo, hi], "G2c_b_passed": bool(lo > -0.01),
        "max_share_median_per_commit": float(np.nanmedian(loads)) if loads else None,
        "G2c_c_passed": bool(loads and np.nanmedian(loads) <= 0.6),
    }
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
