"""1b：把 RLS 记忆接入 Metis 的 local_memory（不修改 metis/ 下的代码）。

时间单位为一次 commit（一个 chunk）：chunk 内所有被选中的 token 在同一时刻写入，
chunk 之间按遗忘因子 lam 衰减一次。信息形式（设计 0a §10，E012）：
    Lambda <- lam Lambda + sum_t w_t kappa_t kappa_t^T
    b      <- lam b      + sum_t w_t kappa_t v_t^T
    M = Lambda^{-1} b,   z <- lam z + sum_t w_t kappa_t
协方差 P = Lambda^{-1}。全部在 fp32 下计算（E012：bf16 不支持求逆，且协方差对舍入敏感）。

install_rls(model, lam) 替换每个 full-attention 层的 local_memory 的读写方法，
并替换其 hyper_memory 的写入函数。uninstall_rls(model) 恢复原始方法。
"""
import torch
import torch.nn.functional as F


def rls_commit_update(M, P, z, kappa, v, w, lam, p_max=None):
    """一次 commit 的 RLS 更新（块形式，精确）。

    M: (B,d,dv)  P: (B,d,d)  z: (B,d,1)  kappa: (B,T,d) 单位键  v: (B,T,dv)  w: (B,T) 权重  lam: 标量
    p_max: 协方差最大特征值的上界（设计 0a §14，防止未激励方向的 windup）；None 表示不限制。
    求逆在 fp32 中进行（fp64 在消费级 GPU 上慢约 5 倍；迹上界是主要的稳定性保护，E012 验证 fp32 下一致），
    返回 (M, P, z)，均为 fp32。
    """
    M, P, z = M.float(), P.float(), z.float()
    kappa, v, w = kappa.float(), v.float(), w.float()
    Lam0 = torch.linalg.inv(P)
    b0 = Lam0 @ M
    Lam = lam * Lam0 + torch.einsum("bt,btj,btk->bjk", w, kappa, kappa)
    b = lam * b0 + torch.einsum("bt,btj,btv->bjv", w, kappa, v)
    P_new = torch.linalg.inv(Lam)
    if p_max is not None:
        # 特征值下限（设计 0a §14）：只限制最大特征值（即未激励方向的协方差），不缩放整体。
        # 等价于 Lambda 的特征值下限 1/p_max。已激励方向（特征值 < p_max）保持不变。
        # 特征分解不参与反向传播（P0 = kv I 时特征值重复，eigh 的梯度为无穷大）；
        # 前向使用截断后的 P，反向直通到未截断的 P（straight-through）。
        Pd = P_new.detach()
        mu, V = torch.linalg.eigh(Pd)
        P_clip = (V * mu.clamp(max=p_max)[..., None, :]) @ V.transpose(-1, -2)
        P_clip = 0.5 * (P_clip + P_clip.transpose(-1, -2))
        P_new = P_new + (P_clip - Pd)
    M_new = P_new @ b
    z_new = lam * z + torch.einsum("bt,btj->bj", w, kappa)[..., None]
    return M_new, P_new, z_new


def readout_denominator(qz, mode="orig"):
    """读出分母（设计：docs/design_readout_norm.md）。
    orig：1 + q·z，与原版 metis_local_memory.read 相同（q·z 可为负，分母可穿过零）。
    abs ：1 + |q·z|，恒 >= 1（读出幅值有界，梯度不经过奇点）。状态递推不受影响，只改读出归一化。
    """
    if mode == "orig":
        return qz + 1.0
    if mode == "abs":
        return 1.0 + qz.abs()
    raise ValueError(f"unknown readout_den mode: {mode}")


def rls_read(lm, q, mode="orig"):
    """RLS 读出，形式与原版 metis_local_memory.read 相同（num_kv_groups=1 与 >1 两个分支），分母由 mode 决定。
    q: (bsz, h, seq, d) 原始查询。状态 lm._state（M）与 lm._key_state（z）须已存在。"""
    bsz, _h, seq_len, _d = q.shape
    q = F.normalize(q, dim=-1)
    q_flat = q.transpose(1, 2).reshape(bsz, seq_len, -1)
    if lm.num_kv_groups > 1:
        q_2d = q_flat.view(bsz, seq_len * lm.num_kv_groups, lm.kv_dim)
        num = torch.matmul(q_2d, lm._state)
        den = readout_denominator(torch.matmul(q_2d, lm._key_state), mode)
        return (num / den).view(bsz, seq_len, lm.q_dim).contiguous()
    num = torch.matmul(q_flat, lm._state)
    den = readout_denominator(torch.matmul(q_flat, lm._key_state), mode)
    return (num / den).contiguous()


def _rls_apply(mod, lam, kv):
    """返回替换 _apply_gated_delta_rule_update 的函数（实例方法形式）。

    权重：被选中 token 的指示值（weights 在前向中归一化为 1/n_sel，乘以 n_sel 得到 1），梯度经直通估计保留。
    """
    def apply(h_normed, weights, local_memory):
        # 键采用与 Metis 相同的尺度 normalize(W_k h)/sqrt(kv)（使读出分母 q^T z + 1 保持稳定）。
        # 与单位键、先验 I 的 RLS 等价，仅输出整体相差 sqrt(kv)，由 mem_norm 吸收（已验证，余弦 1.0）。
        kappa = F.normalize(mod.W_k(h_normed), dim=-1) / (kv ** 0.5)
        v = mod.W_v(h_normed)
        n_sel = (weights > 0).sum(dim=1, keepdim=True).clamp(min=1).to(weights.dtype)
        w = weights * n_sel                                        # 被选 token 为 1，未选为 0
        _ensure_rls_state(local_memory, kappa.shape[0], kv, kappa.device)
        M, P, z = local_memory.rls_state()
        M, P, z = rls_commit_update(M, P, z, kappa, v, w, lam, p_max=kv * kv)
        local_memory.rls_set(M, P, z)
    return apply


def _ensure_rls_state(local_memory, bsz, kv, device):
    """状态为空时创建 fp32 零状态与 P = kv * I（先验与 Metis 键尺度 1/sqrt(kv) 对应，见 _rls_apply）。"""
    if getattr(local_memory, "_state", None) is None:
        local_memory._state = torch.zeros(bsz, kv, kv, device=device, dtype=torch.float32)
        local_memory._key_state = torch.zeros(bsz, kv, 1, device=device, dtype=torch.float32)
        local_memory._P = (kv * torch.eye(kv, device=device, dtype=torch.float32)).expand(bsz, kv, kv).clone()


def install_rls(model, lam=0.9, readout_den="orig"):
    """把模型的记忆换为 RLS。返回被替换的层数。readout_den 见 readout_denominator（默认 orig，与原版一致）。"""
    n = 0
    for block in model.model.metis_blocks:
        lm = block.local_memory
        hm = getattr(block, "hyper_memory", None)
        if lm is None or hm is None:
            continue
        kv = lm.kv_dim

        def rls_state(_lm=lm):
            return _lm._state, _lm._P, _lm._key_state

        def rls_set(M, P, z, _lm=lm):
            _lm._state, _lm._P, _lm._key_state = M, P, z

        def ensure(bsz, device, dtype, _lm=lm, _kv=kv):
            _ensure_rls_state(_lm, bsz, _kv, device)

        orig_read = type(lm).read.__get__(lm, type(lm))

        def read(q, _lm=lm, _orig=orig_read, _kv=kv, _mode=readout_den):
            # 读出在 fp32 下计算，输出转回查询的 dtype（与原版接口一致）；空状态时先初始化 fp32 零状态
            _ensure_rls_state(_lm, q.shape[0], _kv, q.device)
            if _mode == "orig":
                return _orig(q.float()).to(q.dtype)
            return rls_read(_lm, q.float(), _mode).to(q.dtype)

        def reset(_lm=lm):
            _lm._state = None
            _lm._key_state = None
            _lm.__dict__.pop("_P", None)

        def ensure_ready(bsz, device, dtype, _lm=lm, _kv=kv):
            _ensure_rls_state(_lm, bsz, _kv, device)

        lm.rls_state = rls_state
        lm.rls_set = rls_set
        lm._ensure_ready = ensure_ready
        lm.read = read
        lm.reset = reset
        hm._apply_gated_delta_rule_update = _rls_apply(hm, lam, kv)
        n += 1
    return n


def uninstall_rls(model):
    """恢复原始方法（删除实例属性）。"""
    for block in model.model.metis_blocks:
        lm = block.local_memory
        hm = getattr(block, "hyper_memory", None)
        if lm is not None:
            for name in ("rls_state", "rls_set", "_ensure_ready", "read", "reset"):
                lm.__dict__.pop(name, None)
            lm.__dict__.pop("_P", None)
            lm._state = None
            lm._key_state = None
        if hm is not None:
            hm.__dict__.pop("_apply_gated_delta_rule_update", None)
