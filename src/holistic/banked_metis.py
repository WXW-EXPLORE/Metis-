"""G2b：未中心化内容路由的分库接入真实 Metis 的记忆层（不修改 metis/ 下的代码）。设计：docs/design_G2b.md。

每层：K 个库；每库键维度 kd；值维度 dv（与原版相同）。
  键：kp = normalize(k̂ @ Q) / sqrt(kd)，Q 为固定的 PCA 基（kd 列）。Q=None 时 kd=kv，即原版的键尺度。
  路由：b = argmax_b (k̂ · c_b)，k̂ 为单位化的完整键，c_b 为固定中心（不使用数据统计量）。
  写入：每库一次 RLS commit（与 install_rls 相同的更新规则，p_max = kd²），权重为指示值。
  读出：查询按同一路由选库，读出 (q_b M_b)/(q_b z_b + 1)，q_b 为单位化的查询（K=1 且 Q=None 时与原版一致）。

回归：K=1 且 Q=None 时直接调用 install_rls，与原版路径完全相同（G2b-0 检验）。
卸载：uninstall_banked 恢复原版方法。
"""
import torch
import torch.nn.functional as F

import holistic.rls_metis as R

CAPTURE = {"on": False, "log": None}
ZERO = {"on": False}   # 参考：对当前 chunk 的所有写入权重置零（仍推进时间）


def _route(kn, centers):
    """kn: (N, kv) 单位键。返回 (N,) 库索引。K=1 时全为 0。"""
    if centers is None or centers.shape[0] == 1:
        return torch.zeros(kn.shape[0], dtype=torch.long, device=kn.device)
    return (kn @ centers.T).argmax(-1)


def _ensure(lm, K, kd, dv, device):
    if getattr(lm, "_banks", None) is None:
        banks = []
        for _ in range(K):
            M = torch.zeros(1, kd, dv, device=device, dtype=torch.float32)
            P = (kd * torch.eye(kd, device=device, dtype=torch.float32)).expand(1, kd, kd).clone()
            z = torch.zeros(1, kd, 1, device=device, dtype=torch.float32)
            banks.append([M, P, z])
        lm._banks = banks


def _make_apply(lm_ref, hm, Q, centers, lam, kd, dv, K):
    def apply(h_normed, weights, local_memory):
        kn_full = F.normalize(hm.W_k(h_normed), dim=-1)[0].float()    # (T, kv)
        vv = hm.W_v(h_normed)[0].float()                                # (T, dv)
        n_sel = (weights > 0).sum(dim=1, keepdim=True).clamp(min=1).to(weights.dtype)
        w_sel = (weights * n_sel)[0].float()                            # 指示值（被选 token 为 1）
        b_idx = _route(kn_full, centers)
        if Q is None:
            kp_all = kn_full / (kd ** 0.5)
        else:
            kp_all = F.normalize(kn_full @ Q, dim=-1) / (kd ** 0.5)
        _ensure(local_memory, K, kd, dv, kn_full.device)
        for b in range(K):
            M, P, z = local_memory._banks[b]
            mask = (b_idx == b).float() * w_sel
            if ZERO["on"]:
                mask = mask * 0
            if CAPTURE["on"]:
                CAPTURE["log"].append((b, kp_all.detach().double().clone(), vv.detach().double().clone(),
                                       mask.detach().double().clone(), float(lam)))
            M, P, z = R.rls_commit_update(M, P, z, kp_all[None], vv[None], mask[None], lam, p_max=kd * kd)
            local_memory._banks[b] = [M, P, z]
    return apply


def read_rows(lm, rows_kv, Q, centers, kd, K):
    """rows_kv: (N, kv) 查询行（已在每个头的维度上归一化，与父类一致；不再对整行归一化）。返回 (N, dv)。
    路由使用单位化的行；读出使用原始行（K=1 且 Q=None 时与父类读出完全一致）。
    """
    b_idx = _route(F.normalize(rows_kv.float(), dim=-1), centers)
    dv = lm._banks[0][0].shape[-1]
    out = torch.zeros(rows_kv.shape[0], dv, device=rows_kv.device, dtype=torch.float32)
    qr = rows_kv.float()
    for b in range(K):
        M, P, z = lm._banks[b]
        sel = (b_idx == b).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        qb = qr[sel] if Q is None else qr[sel] @ Q
        num = qb @ M[0]
        den = qb @ z[0] + 1.0
        out[sel] = num / den
    return out


def install_banked(model, Q=None, centers=None, lam=0.9, K=1, kd=None, _force_banked_path=False):
    """把模型的记忆换为（分库的）RLS。K=1 且 Q=None 时默认调用 install_rls（回归路径）。
    _force_banked_path=True 时强制使用分库代码路径（用于 G2b-0 回归检验）。返回安装的层数。"""
    if K == 1 and Q is None and not _force_banked_path:
        R.install_rls(model, lam=lam)
        return sum(1 for b in model.model.metis_blocks if b.local_memory is not None)
    n = 0
    for block in model.model.metis_blocks:
        lm = block.local_memory
        hm = getattr(block, "hyper_memory", None)
        if lm is None or hm is None:
            continue
        kv = lm.kv_dim
        kd_ = kd if kd is not None else kv
        dv = kv
        if Q is not None and Q.shape[0] != kv:
            raise ValueError("Q must have kv rows")
        lm._banks = None
        hm._apply_gated_delta_rule_update = _make_apply(lm, hm, Q, centers, lam, kd_, dv, K)

        def read(q, _lm=lm, _Q=Q, _c=centers, _kd=kd_, _K=K, _dv=dv, _kv=kv):
            bsz, heads, seq, d = q.shape
            orig_dtype = q.dtype
            q = F.normalize(q.float(), dim=-1)                          # 与父类一致：先转 fp32，再在每个头的维度上归一化
            qf = q.transpose(1, 2).reshape(bsz, seq, -1)               # (bsz, seq, q_dim)
            rows = qf.reshape(-1, _kv)                                  # 每个 kv 向量一行（与原版 view 一致）
            _ensure(_lm, _K, _kd, _dv, q.device)
            out = read_rows(_lm, rows, _Q, _c, _kd, _K)
            return out.reshape(bsz, seq, -1).to(orig_dtype)

        def reset(_lm=lm):
            _lm._state = None
            _lm._key_state = None
            _lm._banks = None

        lm.read = read
        lm.reset = reset
        n += 1
    return n


def uninstall_banked(model):
    for block in model.model.metis_blocks:
        lm = block.local_memory
        hm = getattr(block, "hyper_memory", None)
        if lm is not None:
            for name in ("read", "reset", "_banks"):
                lm.__dict__.pop(name, None)
        if hm is not None:
            hm.__dict__.pop("_apply_gated_delta_rule_update", None)


def get_bank_states(model, dev):
    out = []
    for block in model.model.metis_blocks:
        lm = block.local_memory
        if lm is None:
            continue
        if getattr(lm, "_banks", None) is None:
            out.append(None)
        else:
            out.append([[M.clone(), P.clone(), z.clone()] for M, P, z in lm._banks])
    return out


def set_bank_states(model, states):
    i = 0
    for block in model.model.metis_blocks:
        lm = block.local_memory
        if lm is None:
            continue
        st = states[i]
        i += 1
        lm._banks = None if st is None else [[M.clone(), P.clone(), z.clone()] for M, P, z in st]
