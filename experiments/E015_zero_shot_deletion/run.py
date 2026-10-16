"""E015：(a) RLS 直接装到预训练 Metis-4B 上的零样本代价；(b) RLS 降秩删除的实测（真实模型）。

(a) 对同一批验证 episode，比较：预训练 GDN、零样本 RLS（不训练，直接替换）、无记忆。
(b) 删除：记录 RLS 每次 commit 的输入（键、值、权重、遗忘因子）。删除包含答案的 chunk 时，
    用降秩撤销该 chunk 的信息（权重 lam^age），与"从未写入该 chunk"的参考（零权重 replay）比较，
    并比较答案的 NLL。
注：GDN 没有寻址删除接口，因此不做 GDN 的删除对照（只能重算，见 README）。

用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E015_zero_shot_deletion/run.py --out experiments/E015_zero_shot_deletion --n 30
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri  # noqa: E402
import holistic.rls_metis as R  # noqa: E402
from holistic.rls_metis import install_rls, uninstall_rls  # noqa: E402
from holistic.synthetic_memory_data import N_MEM, make_split  # noqa: E402
from train.train_utils import freeze_backbone  # noqa: E402

# 捕获 RLS 的每次 commit 输入（按层交错记录），并支持把某次 commit 的权重置零
CAP = {"list": None, "zero_commit": None, "commit_counter": 0}
_orig_commit = R.rls_commit_update


def _wrapped_commit(M, P, z, kappa, v, w, lam, p_max=None):
    commit_idx = CAP["commit_counter"]
    if CAP["zero_commit"] is not None and commit_idx // N_LAYERS == CAP["zero_commit"]:
        w = w * 0
    if CAP["list"] is not None:
        CAP["list"].append((kappa.detach().double(), v.detach().double(), w.detach().double(), float(lam)))
    CAP["commit_counter"] += 1
    return _orig_commit(M, P, z, kappa, v, w, lam, p_max=p_max)


R.rls_commit_update = _wrapped_commit
N_LAYERS = 8


def replay(model, tokenizer, device, ep, capture=False, zero_chunk=None):
    model.reset()
    CAP["list"] = [] if capture else None
    CAP["zero_commit"] = zero_chunk
    CAP["commit_counter"] = 0
    for chunk in ep["memory_chunks"]:
        ids, mask = ri.encode_messages(tokenizer, [{"role": "user", "content": chunk}], device,
                                       add_generation_prompt=False)
        model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)
    captured = CAP["list"]
    CAP["list"] = None
    CAP["zero_commit"] = None
    return captured


def layers(model):
    return [b.local_memory for b in model.model.metis_blocks if b.local_memory is not None]


def get_states(model):
    return [(lm._state.detach().clone(), lm._P.detach().clone(), lm._key_state.detach().clone())
            for lm in layers(model)]


def set_states(model, states):
    for lm, (M, P, z) in zip(layers(model), states):
        lm._state, lm._P, lm._key_state = M, P, z


def answer_nll(model, tokenizer, device, ep):
    prompt_ids, _ = ri.encode_messages(tokenizer, [{"role": "user", "content": ep["question"]}], device,
                                       add_generation_prompt=True)
    ans = tokenizer.encode(ep["answer"], add_special_tokens=False)
    ans_t = torch.tensor([ans], device=device, dtype=prompt_ids.dtype)
    ids = torch.cat([prompt_ids, ans_t], 1)
    labels = torch.full_like(ids, -100)
    labels[:, prompt_ids.size(1):] = ans_t
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
    return float(out.loss.item())


def delete_chunk_downdate(states_full, captured, c_del, age):
    """对 chunk c_del 的每一层做信息形式降秩（float64）。返回新的 (M, P, z) 列表。"""
    out = []
    for l, (M, P, z) in enumerate(states_full):
        kappa, v, w, lam = captured[c_del * N_LAYERS + l]
        Md, Pd, zd = M[0].double(), P[0].double(), z[0].double()
        c = (w[0] * (lam ** age))                                   # (T,)
        Lam = torch.linalg.inv(Pd)
        b = Lam @ Md
        K = kappa[0]                                                  # (T,d)
        V = v[0]                                                      # (T,dv)
        Lam_new = Lam - (K.T * c) @ K
        b_new = b - (K.T * c) @ V
        z_new = zd - (K.T * c).sum(dim=1, keepdim=True)
        P_new = torch.linalg.inv(Lam_new)
        M_new = P_new @ b_new
        out.append((M_new[None].float(), P_new[None].float(), z_new[None].float()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", default="/hy-tmp/metis-models/Metis-4B")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    load = argparse.Namespace(checkpoint_path=args.checkpoint_path, model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tokenizer, device = ri.load_model_and_tokenizer(load)
    model.eval()
    freeze_backbone(model)
    eps = make_split("val", args.n, seed=9000)

    # ── (a) 零样本代价 ─────────────────────────────────────────────
    res_a = {"gdn": [], "rls_zero_shot": [], "no_memory": []}
    with torch.no_grad():
        for ep in eps:
            replay(model, tokenizer, device, ep)
            res_a["gdn"].append(answer_nll(model, tokenizer, device, ep))
            model.reset()
            res_a["no_memory"].append(answer_nll(model, tokenizer, device, ep))
        install_rls(model, lam=0.9)
        for ep in eps:
            replay(model, tokenizer, device, ep)
            res_a["rls_zero_shot"].append(answer_nll(model, tokenizer, device, ep))
        uninstall_rls(model)
    summary_a = {k: sum(v) / len(v) for k, v in res_a.items()}
    print("zero-shot NLL:", summary_a, flush=True)

    # ── (b) 删除（RLS 降秩）─────────────────────────────────────────
    install_rls(model, lam=0.9)
    rows = []
    with torch.no_grad():
        for ep in eps[:10]:
            cap = replay(model, tokenizer, device, ep, capture=True)
            full = get_states(model)
            # 答案所在的最后一个 chunk（最新取值出现的位置）
            c_del = max(i for i, c in enumerate(ep["memory_chunks"]) if ep["answer"] in c)
            age = N_MEM - 1 - c_del
            ref_capture = replay(model, tokenizer, device, ep, capture=False, zero_chunk=c_del)
            ref = get_states(model)
            new = delete_chunk_downdate(full, cap, c_del, age)
            rel_M = []
            rel_P = []
            for (Mn, Pn, zn), (Mr, Pr, zr) in zip(new, ref):
                rel_M.append(((Mn - Mr).norm() / Mr.norm().clamp_min(1e-12)).item())
                rel_P.append(((Pn - Pr).norm() / Pr.norm().clamp_min(1e-12)).item())
            set_states(model, new)
            nll_deleted = answer_nll(model, tokenizer, device, ep)
            set_states(model, ref)
            nll_ref = answer_nll(model, tokenizer, device, ep)
            set_states(model, full)
            nll_full = answer_nll(model, tokenizer, device, ep)
            model.reset()
            nll_nomem = answer_nll(model, tokenizer, device, ep)
            rows.append({"episode_id": ep["episode_id"], "c_del": c_del, "age": age,
                         "state_rel_err_max": max(rel_M), "P_rel_err_max": max(rel_P),
                         "nll_before_delete": nll_full, "nll_after_delete": nll_deleted,
                         "nll_reference_no_chunk": nll_ref, "nll_no_memory": nll_nomem})
            print(json.dumps(rows[-1]), flush=True)
    uninstall_rls(model)

    summary_b = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k not in ("episode_id",)}
    result = {"zero_shot_nll_mean": summary_a, "zero_shot_n": len(eps),
              "deletion_per_episode": rows, "deletion_mean": summary_b}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print("deletion mean:", json.dumps(summary_b, indent=1), flush=True)


if __name__ == "__main__":
    main()
