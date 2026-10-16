"""G2b-0：分库代码路径（K=1，Q=None，强制）与 install_rls 在真实模型上的状态与读出一致性。
用法（仓库根目录）：PYTHONPATH=.:src python experiments/E021_G2b/regression_G2b0.py --out experiments/E021_G2b
"""
import argparse, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri
import holistic.rls_metis as R
import holistic.banked_metis as B
from holistic.synthetic_memory_data import make_split
from train.train_utils import freeze_backbone

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                          backbone_type="qwen3_5", device="cuda", dtype="bf16")
model, tok, dev = ri.load_model_and_tokenizer(load); model.eval(); freeze_backbone(model)
ep = make_split("val", 1, seed=9300)[0]

def run(install):
    install()
    model.reset()
    outs = []
    with torch.no_grad():
        for chunk in ep["memory_chunks"]:
            ids, mask = ri.encode_messages(tok, [{"role": "user", "content": chunk}], dev, add_generation_prompt=False)
            model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)
        states = []
        for b in model.model.metis_blocks:
            lm = b.local_memory
            if lm is None: continue
            if getattr(lm, "_banks", None) is not None:
                M, P, z = lm._banks[0]; states.append((M[0].clone(), P[0].clone(), z[0].clone()))
            else:
                states.append((lm._state[0].clone(), lm._P[0].clone(), lm._key_state[0].clone()))
        q, _ = ri.encode_messages(tok, [{"role": "user", "content": ep["question"]}], dev, add_generation_prompt=True)
        out = model(input_ids=q, attention_mask=torch.ones_like(q), use_cache=False, logits_to_keep=0).logits.clone()
    if hasattr(model, "model"):
        for b in model.model.metis_blocks:
            if b.local_memory is not None:
                pass
    return states, out

R.uninstall_rls(model); B.uninstall_banked(model)
st_a, lg_a = run(lambda: R.install_rls(model, lam=0.9))
R.uninstall_rls(model); B.uninstall_banked(model)
st_b, lg_b = run(lambda: B.install_banked(model, Q=None, centers=None, lam=0.9, K=1, _force_banked_path=True))
state_diffs = [float((a[0]-b[0]).abs().max()) for a, b in zip(st_a, st_b)]
p_diffs = [float((a[1]-b[1]).abs().max()) for a, b in zip(st_a, st_b)]
z_diffs = [float((a[2]-b[2]).abs().max()) for a, b in zip(st_a, st_b)]
res = {"state_max_abs_diff_per_layer": state_diffs, "P_max_abs_diff_per_layer": p_diffs,
       "z_max_abs_diff_per_layer": z_diffs, "logits_max_abs_diff": float((lg_a.float()-lg_b.float()).abs().max()),
       "logits_equal": bool(torch.equal(lg_a, lg_b)), "passed": bool(max(state_diffs) == 0.0 and max(p_diffs) == 0.0)}
Path(args.out).mkdir(parents=True, exist_ok=True)
(Path(args.out) / "G2b0_result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1), flush=True)
