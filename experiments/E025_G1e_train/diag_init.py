"""E025 诊断：同一种子下，三个臂的可训练参数初始化是否一致（只读，不训练）。

动机：三个臂在同一种子下的初始验证 NLL 不同（例如种子 1：K1 9.83、K4_CL 10.42、K4_random 10.00）。
如果可训练参数的初始化相同，则差异来自读出结构（分库路由），而不是随机初始化不同。
做法：加载一次模型；对每个臂，安装对应的记忆，设置相同种子，执行 reinit_memory，计算可训练参数的校验和。
用法（仓库根目录）：PYTHONPATH=.:src python experiments/E025_G1e_train/diag_init.py --seed 1 --out experiments/E025_G1e_train/diag_init
"""
import argparse, hashlib, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_inference as ri
from holistic.banked_metis import install_banked, uninstall_banked
import holistic.rls_metis as R
from train.train_utils import freeze_backbone
import importlib.util


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


T = _load("e25train", "experiments/E025_G1e_train/train.py")
E13 = T.E13


def checksum(model):
    h = hashlib.sha256()
    n = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or any(g in name for g in E13.GATE_NAMES):
            continue
        h.update(name.encode()); h.update(p.detach().float().cpu().numpy().tobytes())
        n += p.numel()
    return h.hexdigest()[:16], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prototypes", default=str(ROOT / "experiments/E025_G1e_train/abs/prototypes.pt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load)
    freeze_backbone(model)
    proto = torch.load(args.prototypes, weights_only=False)
    res = {}
    arms = {
        "K1_full": None,
        "K4_CL_cons": (proto["Q"], [c.to(dev) for c in proto["C_cl"][args.seed]]),
        "K4_random": (proto["Q"], [c.to(dev) for c in proto["C_rand"][args.seed]]),
    }
    for arm, spec in arms.items():
        if spec is None:
            install_banked(model, lam=0.9, readout_den="abs")
        else:
            Qs = [q.to(dev) for q in spec[0]]
            install_banked(model, Q=Qs, centers=spec[1], lam=0.9, K=4, kd=256, readout_den="abs")
        torch.manual_seed(args.seed)
        E13.reinit_memory(model)
        sha, n = checksum(model)
        res[arm] = {"sha256_16": sha, "n_trainable_values": n}
        print(arm, res[arm], flush=True)
        if spec is None:
            R.uninstall_rls(model)
        else:
            uninstall_banked(model)
    (out / f"init_checksums_seed{args.seed}.json").write_text(json.dumps(res, indent=1))
    same = len({v["sha256_16"] for v in res.values()}) == 1
    print("IDENTICAL_INIT" if same else "DIFFERENT_INIT", flush=True)


if __name__ == "__main__":
    main()
