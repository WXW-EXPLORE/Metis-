"""E014（1b 步骤一）：RLS 接入的回归测试。

检查：安装 RLS 后再卸载，模型的前向输出与记忆状态应与原版逐位一致（atol=0）。
同时检查：卸载后记忆读写路径已恢复为原版方法（无实例级残留）。

用法（仓库根目录，激活 /hy-tmp/venv-metis）：
    PYTHONPATH=.:src python experiments/E014_regression/run.py --out experiments/E014_regression
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


def forward_snapshot(model, tokenizer, device, ep):
    """对一个 episode 做前向，返回查询的 logits 与各层记忆状态。"""
    model.reset()
    for chunk in ep["memory_chunks"]:
        ids, mask = ri.encode_messages(tokenizer, [{"role": "user", "content": chunk}], device,
                                       add_generation_prompt=False)
        model(input_ids=ids, attention_mask=mask, commit_memory=True, use_cache=False, logits_to_keep=1)
    states = {}
    for i, block in enumerate(model.model.metis_blocks):
        lm = block.local_memory
        if lm is not None and getattr(lm, "_state", None) is not None:
            states[i] = lm._state.detach().clone()
    q, _ = ri.encode_messages(tokenizer, [{"role": "user", "content": ep["question"]}], device,
                              add_generation_prompt=True)
    with torch.no_grad():
        out = model(input_ids=q, attention_mask=torch.ones_like(q), use_cache=False, logits_to_keep=0)
    return out.logits.detach().clone(), states


def residual_instance_attrs(model):
    """卸载后仍残留的实例级方法（应为空）。"""
    found = []
    for i, block in enumerate(model.model.metis_blocks):
        lm, hm = block.local_memory, getattr(block, "hyper_memory", None)
        if lm is not None:
            for name in ("rls_state", "rls_set", "_ensure_ready", "read", "reset"):
                if name in lm.__dict__:
                    found.append(f"block{i}.local_memory.{name}")
        if hm is not None and "_apply_gated_delta_rule_update" in hm.__dict__:
            found.append(f"block{i}.hyper_memory._apply_gated_delta_rule_update")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", default="/hy-tmp/metis-models/Metis-4B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    load = argparse.Namespace(checkpoint_path=args.checkpoint_path, model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tokenizer, device = ri.load_model_and_tokenizer(load)
    model.eval()
    freeze_backbone(model)
    ep = make_split("val", 1, seed=9000)[0]

    logits_a, states_a = forward_snapshot(model, tokenizer, device, ep)
    install_rls(model, lam=0.9)
    _logits_rls, _states_rls = forward_snapshot(model, tokenizer, device, ep)
    uninstall_rls(model)
    logits_b, states_b = forward_snapshot(model, tokenizer, device, ep)

    logits_equal = torch.equal(logits_a, logits_b)
    states_equal = states_a.keys() == states_b.keys() and all(
        torch.equal(states_a[k], states_b[k]) for k in states_a)
    residual = residual_instance_attrs(model)
    result = {
        "logits_bitwise_equal_after_uninstall": bool(logits_equal),
        "states_bitwise_equal_after_uninstall": bool(states_equal),
        "n_state_layers": len(states_a),
        "max_abs_logit_diff": float((logits_a.float() - logits_b.float()).abs().max().item()),
        "residual_instance_methods": residual,
        "passed": bool(logits_equal and states_equal and not residual),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
