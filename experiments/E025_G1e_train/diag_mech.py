"""E025 机制诊断：RLS 读出分母与信息矩阵特征值 与 梯度尖峰的关系。只读插桩，不更新参数。

做法（不修改 src/ 与 metis/）：
  - 替换 holistic.banked_metis.read_rows：记录读出分母 den = q·z + 1 的最小值与各读出的 |num/den| 最大值；
  - 替换 holistic.rls_metis.rls_commit_update：记录写入后信息矩阵 Λ（即 P^{-1}）的特征值范围与 p_max 触发情况；
  - 对选定的 episode 计算梯度，记录总梯度范数与 W_k/query_proj/W_v 的分组范数。
用法（仓库根目录）：
    PYTHONPATH=.:src python experiments/E025_G1e_train/diag_mech.py --episodes 12,22,6,19,0 --out experiments/E025_G1e_train/diag_mech
"""
import argparse, json, random, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import importlib.util
import run_inference as ri
import holistic.banked_metis as BM
import holistic.rls_metis as R
from holistic.banked_metis import install_banked
from train.train_utils import freeze_backbone


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


T = _load("e25train", "experiments/E025_G1e_train/train.py")
E13 = T.E13
GROUPS = ["W_k", "W_v", "pool_score", "query_proj", "mem_norm", "query_norm"]
STAT = {"den_min": None, "ratio_max": None, "lam_min": None, "lam_max": None, "pmax_hits": 0, "writes": 0}


def _reset_stat():
    STAT.update({"den_min": float("inf"), "den_abs_min": float("inf"), "neg_frac_sum": 0.0, "n_readouts": 0,
                 "ratio_max": 0.0, "lam_min": float("inf"), "lam_max": 0.0, "pmax_hits": 0, "writes": 0})


def _wrap_reads(model):
    """K1 路径（install_rls）的读出：包装实例上的 read，复现原版的分母 q_flat·z + 1（与 metis_local_memory.read 一致）。"""
    for block in model.model.metis_blocks:
        lm = block.local_memory
        if lm is None:
            continue
        orig = lm.read

        def read(q, _lm=lm, _orig=orig):
            out = _orig(q)
            with torch.no_grad():
                qn = torch.nn.functional.normalize(q.float(), dim=-1)
                qf = qn.transpose(1, 2).reshape(qn.shape[0], qn.shape[2], -1)
                qr = qf.reshape(qf.shape[0], -1, _lm.kv_dim)
                den = (qr @ _lm._key_state + 1.0).reshape(-1)
                STAT["den_min"] = min(STAT["den_min"], float(den.min()))
                STAT["den_abs_min"] = min(STAT["den_abs_min"], float(den.abs().min()))
                STAT["neg_frac_sum"] += float((den < 0).sum())
                STAT["n_readouts"] += int(den.numel())
                STAT["ratio_max"] = max(STAT["ratio_max"], float(out.float().abs().max()))
            return out

        lm.read = read


_orig_commit = R.rls_commit_update


def _commit(M, P, z, kappa, v, w, lam, p_max=None):
    res = _orig_commit(M, P, z, kappa, v, w, lam, p_max=p_max)
    with torch.no_grad():
        M2, P2, z2 = res
        Lam = torch.linalg.inv(P2[0].float())
        ev = torch.linalg.eigvalsh(0.5 * (Lam + Lam.T))
        STAT["lam_min"] = min(STAT["lam_min"], float(ev.min()))
        STAT["lam_max"] = max(STAT["lam_max"], float(ev.max()))
        STAT["writes"] += 1
        pe = torch.linalg.eigvalsh(0.5 * (P2[0].float() + P2[0].float().T))
        if float(pe.max()) > (p_max if p_max is not None else float("inf")) * 0.999:
            STAT["pmax_hits"] += 1
    return res


R.rls_commit_update = _commit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="12,22,6,19,0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readout_den", choices=["orig", "abs"], default="orig")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    want = [int(x) for x in args.episodes.split(",")]

    fit, _ = T.load_qa()
    rng = random.Random(1000 + args.seed)
    perm = list(fit); rng.shuffle(perm)
    load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load)
    freeze_backbone(model)
    torch.manual_seed(args.seed)  # 固定参数重新初始化（reinit_memory），使不同读出模式配对
    install_banked(model, lam=0.9, readout_den=args.readout_den)
    _wrap_reads(model)
    E13.reinit_memory(model)
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(g in name for g in E13.GATE_NAMES):
            p.requires_grad = False
            continue
        params.append((name, p))

    model.train()
    f = open(out / "episodes.jsonl", "w")
    for i in want:
        ep = perm[i]
        for _, p in params:
            p.grad = None
        _reset_stat()
        rec = {"i": i, "n_chunks": len(ep["memory_chunks"])}
        loss = E13.episode_loss(model, tok, dev, ep)
        rec["loss"] = float(loss.item())
        loss.backward()
        per = {g: 0.0 for g in GROUPS + ["other"]}
        for name, p in params:
            if p.grad is not None:
                key = next((g for g in GROUPS if g in name), "other")
                per[key] += float(p.grad.double().pow(2).sum())
        rec["grad_norm"] = sum(per.values()) ** 0.5
        rec["grad_by_group"] = {k: round(v ** 0.5, 1) for k, v in per.items()}
        rec.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in STAT.items()})
        rec["neg_frac"] = round(STAT["neg_frac_sum"] / max(1, STAT["n_readouts"]), 6)
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(json.dumps(rec), flush=True)
        model.reset()
    f.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
