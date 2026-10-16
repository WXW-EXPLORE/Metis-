"""Phase 0b：替换 gated delta 写入中的遗忘项或擦除项，用于消融。

公式与 metis/dev_beta/metis_hyper_memory.py 的 GatedDeltaRuleMixin._apply_gated_delta_rule_update
逐行一致，只在 mode 指定时替换 alpha（no_forget）或擦除项（no_erase）。
见 docs/design_phase0b.md §3。
"""
import torch
import torch.nn.functional as F

MODES = ("full", "no_erase", "no_forget")
PRECISIONS = ("bf16", "fp32")


def gated_delta_update(mod, h_normed, weights, local_memory, mode="full", stats=None, precision="bf16"):
    """precision="fp32" 时，投影结果与状态的递推都在 fp32 下进行（见 design_phase0b_fp32.md §2）。

    投影层 W_k、W_v 与门控层仍以原 dtype 计算，结果转为 fp32 后参与递推。
    precision="bf16" 时与原始公式逐位一致。
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}")
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision: {precision!r}")

    write_key = F.normalize(mod.W_k(h_normed), dim=-1) / (mod.kv_dim ** 0.5)
    write_value = mod.W_v(h_normed)
    alpha_gate = torch.sigmoid(mod.gated_delta_alpha(h_normed).squeeze(-1))
    beta_gate = torch.sigmoid(mod.gated_delta_beta(h_normed).squeeze(-1))

    if precision == "fp32":
        write_key = write_key.float()
        write_value = write_value.float()
        alpha_gate = alpha_gate.float()
        beta_gate = beta_gate.float()

    weights = weights.to(write_key.dtype)
    weight_mass = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
    alpha = (weights * alpha_gate).sum(dim=1) / weight_mass.squeeze(1)
    beta = weights * (mod.update_ratio * beta_gate)
    beta_exp = beta.unsqueeze(-1)

    if mode == "no_forget":
        alpha = torch.ones_like(alpha)

    bsz = write_key.size(0)
    state = local_memory.state
    if state is None:
        state = torch.zeros(
            bsz, mod.kv_dim, mod.kv_dim,
            device=write_key.device, dtype=write_key.dtype,
        )
    else:
        state = state.to(write_key.dtype)

    key_state = getattr(local_memory, "key_state", None)
    if key_state is not None:
        key_state = key_state.to(write_key.dtype)

    key_memory = torch.matmul(write_key, state)
    erase_state = torch.matmul(write_key.transpose(-1, -2), beta_exp * key_memory)
    if mode == "no_erase":
        erase_state = torch.zeros_like(erase_state)
    add_state = torch.matmul(write_key.transpose(-1, -2), beta_exp * write_value)
    new_state = alpha.view(bsz, 1, 1) * (state - erase_state) + add_state

    if stats is not None:
        a = alpha.view(bsz, 1, 1)
        erase_norm = (a * erase_state).flatten(1).norm(dim=1)
        state_norm = (a * state).flatten(1).norm(dim=1).clamp(min=1e-12)
        stats.setdefault("erase_rel", []).append((erase_norm / state_norm).float().cpu())
        stats.setdefault("forget", []).append((1 - alpha).float().cpu())

    has_key_state = hasattr(local_memory, "key_state")
    if key_state is None and has_key_state:
        key_state = torch.zeros(
            bsz, mod.kv_dim, 1,
            device=write_key.device, dtype=write_key.dtype,
        )

    if key_state is not None:
        key_memory_mass = torch.matmul(write_key, key_state)
        erase_key_state = torch.matmul(
            write_key.transpose(-1, -2),
            beta_exp * key_memory_mass,
        )
        if mode == "no_erase":
            erase_key_state = torch.zeros_like(erase_key_state)
        add_key_state = torch.matmul(write_key.transpose(-1, -2), beta_exp)
        new_key_state = (
            alpha.view(bsz, 1, 1) * (key_state - erase_key_state)
            + add_key_state
        )
        local_memory.write(new_state, new_key_state)
    else:
        local_memory.write(new_state)


def install_read_fp32(model):
    """为每层本地记忆的 read 设置 fp32 包裹：query 转 fp32 计算，输出转回原 dtype。

    状态为 None 时调用原始 read，保持原行为（它会创建 bf16 零状态）。
    """
    for layer in model.model.metis_blocks:
        lm = layer.local_memory
        if lm is None:
            continue
        orig = type(lm).read.__get__(lm, type(lm))

        def read_fp32(q, _lm=lm, _orig=orig):
            if _lm.state is None:
                return _orig(q)
            return _orig(q.float()).to(q.dtype)

        lm.read = read_fp32


def restore_read(model):
    """取消 install_read_fp32 的包裹，恢复类上的原始 read。"""
    for layer in model.model.metis_blocks:
        lm = layer.local_memory
        if lm is not None:
            lm.__dict__.pop("read", None)


def install_mode(model, mode, stats=None, precision="bf16"):
    """把模型中所有 hyper memory 的写入函数替换为指定模式与精度。

    mode 为 "original" 时恢复类上的原始写入方法（删除实例属性）。
    precision="fp32" 时同时包裹读出（见 install_read_fp32）；其他情况取消读出包裹。
    stats 若为 dict，则按模块名记录每层的统计（见 gated_delta_update）。
    返回被修改的模块数。
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision: {precision!r}")
    restore_read(model)
    if mode != "original" and precision == "fp32":
        install_read_fp32(model)

    count = 0
    for name, mod in model.named_modules():
        if not hasattr(mod, "_apply_gated_delta_rule_update"):
            continue
        if mode == "original":
            mod.__dict__.pop("_apply_gated_delta_rule_update", None)
        else:
            layer_stats = stats.setdefault(name, {}) if stats is not None else None

            def wrapped(h_normed, weights, local_memory, _mod=mod, _mode=mode,
                        _stats=layer_stats, _prec=precision):
                return gated_delta_update(_mod, h_normed, weights, local_memory, _mode, _stats, _prec)

            mod._apply_gated_delta_rule_update = wrapped
        count += 1
    return count
