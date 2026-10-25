"""H11：多时间尺度（快、慢两组记忆）的纯函数实现（设计：docs/design_H11_timescale.md）。

组成：
  - assign_hard：每个 token 按内容（最大余弦的中心）硬分配到一组；每个 token 只进入一组。
  - group_write：对一个 chunk 的 token，按分配把它们写入对应组（各组用各自的遗忘因子 λ 的 RLS 更新）。
  - self_retrieval：用键 kp 在状态上读出 kp·M/(kp·z+1)，与 v 求余弦（与 E026 的口径一致）。

约定：状态为 (M, P, z)，形状与 rls_metis.rls_commit_update 相同（B=1）。
注意：信息形式的 RLS 在没有写入时 M 不变（z 按 λ 缩放），因此"年龄"按之后的写入次数计，而不是时间。
"""
import torch
import torch.nn.functional as F

from holistic.rls_metis import rls_commit_update


def assign_hard(kn, centers):
    """kn: (T, d) 单位键；centers: (G, d) 单位中心。返回 (T,) 的组索引（最大余弦）。"""
    return (kn @ centers.T).argmax(-1)


def group_write(states, kn, v, centers, lams, p_max=None):
    """把一个 chunk 的 token 按分配写入各组。

    states: 长度为 G 的列表，每个元素为 (M, P, z)，形状 (1, kd, dv)/(1, kd, kd)/(1, kd, 1)。
    kn: (T, kd) 键（已按组的键映射处理，这里要求各组共用同一键空间）；v: (T, dv)。
    centers: (G, kd)。lams: 长度为 G 的遗忘因子列表。
    每个组在本次写入中都执行一次 commit（没有分配到该组的 token 的权重为 0，只推进时间）。
    返回新的 states 列表。
    """
    G = len(states)
    idx = assign_hard(kn, centers)
    out = []
    for g in range(G):
        w = (idx == g).float()[None]                       # (1, T)
        M, P, z = states[g]
        M, P, z = rls_commit_update(M, P, z, kn[None], v[None], w, lams[g], p_max=p_max)
        out.append((M, P, z))
    return out, idx


def self_retrieval(state, kn, v):
    """用 kn 作为查询，在一个组的状态上读出，与 v 的余弦（逐 token）。返回 (T,) 余弦。"""
    M, _P, z = state
    out = (kn @ M[0]) / (kn @ z[0] + 1.0)
    return F.cosine_similarity(out, v, dim=-1)
