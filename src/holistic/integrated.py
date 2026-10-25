"""集成候选模型的第一版：RLS 载体 + 快照重放删除（设计：docs/design_integrated.md）。

组成：
  - install_integrated：安装 RLS 载体（即 rls_metis.install_rls）；slow_value=True（接口 (b) 的慢层）尚未实现，直接报错。
  - get_rls_states / set_rls_states：读写每个记忆层的 RLS 状态 (M, P, z)，用于快照。
  - delete_chunk：从快照恢复，然后按顺序重放快照之后的 chunk（不含被删的 chunk）。

约定：状态只由写入决定（与读出分母无关，见 tests/test_g2c_replay.py），因此删除只需要快照与重放。
"""
import torch

from holistic.rls_metis import install_rls


def install_integrated(model, lam=0.9, readout_den="orig", slow_value=False):
    """安装集成模型的记忆。返回被替换的层数。
    slow_value=False（默认）：与 install_rls 完全相同的路径，保证基线可复现。
    slow_value=True：接口 (b) 的慢层，解码方案未决（见设计文档第 3 节），暂不允许使用。
    """
    if slow_value:
        raise NotImplementedError("slow_value (interface b) is blocked on the decode design; see docs/design_integrated.md §3")
    return install_rls(model, lam=lam, readout_den=readout_den)


def _blocks_with_memory(model):
    return [b.local_memory for b in model.model.metis_blocks if b.local_memory is not None]


def get_rls_states(model):
    """每个记忆层的 (M, P, z) 副本；尚未写入的层为 None。"""
    out = []
    for lm in _blocks_with_memory(model):
        if getattr(lm, "_state", None) is None:
            out.append(None)
        else:
            out.append((lm._state.clone(), lm._P.clone(), lm._key_state.clone()))
    return out


def set_rls_states(model, states):
    """从 get_rls_states 的结果恢复（写入副本，不与快照共享张量）。"""
    lms = _blocks_with_memory(model)
    if len(lms) != len(states):
        raise ValueError(f"expected {len(lms)} layer states, got {len(states)}")
    for lm, st in zip(lms, states):
        if st is None:
            lm._state = None
            lm._key_state = None
            lm.__dict__.pop("_P", None)
        else:
            M, P, z = st
            lm._state, lm._P, lm._key_state = M.clone(), P.clone(), z.clone()


@torch.no_grad()
def delete_chunk(model, snapshot_before, chunks_after, write_fn):
    """删除一个 chunk：从其写入之前的快照恢复，再重放其后的 chunk。

    snapshot_before: 写入被删 chunk 之前的 get_rls_states 结果。
    chunks_after: 被删 chunk 之后的 chunk 列表（不含被删的那个）。
    write_fn(model, chunk)：把一个 chunk 写入记忆（调用方负责，例如一次 commit_memory 前向）。
    """
    set_rls_states(model, snapshot_before)
    for chunk in chunks_after:
        write_fn(model, chunk)
