"""E032：模型级检验（设计 docs/design_integrated.md §5 第 5 项）+ mem_norm 的缩放不变性核实。

(1) 默认开关下，install_integrated 与 install_rls 在同一输入、同一种子初始化下的输出逐位一致；
(2) 实际 mem_norm 的类型，以及对读出乘 s 后的输出是否不变（解释 E031 的 S-R 无效）。
不训练。测试集不读取（使用 pilot 划分的第 5 个对话）。
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import run_inference as ri
from train.train_utils import freeze_backbone
from holistic.rls_metis import install_rls, uninstall_rls
from holistic.integrated import install_integrated


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


T = _load("e25_train", "experiments/E025_G1e_train/train.py")
E13 = T.E13


def loss_with(model, tok, dev, ep, installer, seed):
    uninstall_rls(model)
    installer(model)
    torch.manual_seed(seed)
    E13.reinit_memory(model)
    model.eval()
    with torch.no_grad():
        return float(E13.episode_loss(model, tok, dev, ep).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments/E032_model_check/result.json"))
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()
    _, test = T.load_qa(pilot=True)
    ep = test[0]
    load = argparse.Namespace(checkpoint_path="/hy-tmp/metis-models/Metis-4B", model_path=None, tokenizer_path=None,
                              backbone_type="qwen3_5", device="cuda", dtype="bf16")
    model, tok, dev = ri.load_model_and_tokenizer(load)
    freeze_backbone(model)
    res = {"episode_id": ep["episode_id"], "seed": args.seed}

    # (2) mem_norm 的类型与缩放不变性
    blk = next(b for b in model.model.metis_blocks if b.local_memory is not None)
    mn = blk.mem_norm
    res["mem_norm_class"] = type(mn).__name__
    x = torch.randn(1, 8, mn.weight.shape[0], device=dev, dtype=torch.float32)
    with torch.no_grad():
        y1 = mn(x.to(mn.weight.dtype)).float(); y01 = mn(0.1 * x.to(mn.weight.dtype)).float()
    res["mem_norm_scale_0p1_max_abs_diff"] = float((y1 - y01).abs().max())
    res["mem_norm_output_max_abs"] = float(y1.abs().max())

    # (1) 集成路径（默认开关）与 install_rls 逐位一致
    for den in ["abs", "orig"]:
        a = loss_with(model, tok, dev, ep, lambda m: install_integrated(m, lam=0.9, readout_den=den), args.seed)
        b = loss_with(model, tok, dev, ep, lambda m: install_rls(m, lam=0.9, readout_den=den), args.seed)
        res[f"loss_integrated_{den}"] = a
        res[f"loss_install_rls_{den}"] = b
        res[f"bitwise_equal_{den}"] = (a == b)
    uninstall_rls(model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
