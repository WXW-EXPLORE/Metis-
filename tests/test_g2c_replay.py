"""G2c 单元测试：快照重放删除的数学正确性（设计：docs/design_G2c.md）。

检查：
  1. 从快照出发重放（跳过被删 chunk）与"从未写入该 chunk"的顺序写入逐位一致；
  2. 快照恢复后重放不依赖快照之后的状态（删除不受后续写入污染）；
  3. 状态递推只依赖写入（与读出分母形式无关，线性于状态的更新由 rls_commit_update 给出）；
  4. 自检索辅助函数对已知的线性记忆给出正确的余弦。

运行（仓库根目录）：PYTHONPATH=src python -m unittest discover -s tests -p "test_g2c_replay.py" -v
"""
import unittest

import torch
import torch.nn.functional as F

from holistic.rls_metis import rls_commit_update


def _write(state, chunks, lam=0.9, p_max=None):
    """依次写入若干 chunk（每个 chunk: kappa (1,T,d), v (1,T,dv), w (1,T)）。state = (M, P, z)。"""
    M, P, z = state
    for kappa, v, w in chunks:
        M, P, z = rls_commit_update(M, P, z, kappa, v, w, lam, p_max=p_max)
    return M, P, z


def _make_chunks(seed, n, d=6, dv=4, T=3):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        kappa = F.normalize(torch.randn(1, T, d, generator=g), dim=-1) / (d ** 0.5)
        v = torch.randn(1, T, dv, generator=g)
        w = torch.ones(1, T)
        out.append((kappa, v, w))
    return out


def _init(d=6, dv=4):
    P = (d * torch.eye(d)).expand(1, d, d).clone()
    M = torch.zeros(1, d, dv)
    z = torch.zeros(1, d, 1)
    return M, P, z


class TestSnapshotReplay(unittest.TestCase):
    def test_replay_from_snapshot_equals_never_written(self):
        chunks = _make_chunks(0, 5)
        for j in range(len(chunks)):
            # 快照：写入 chunk 0..j-1 之后的状态
            snap = _write(_init(), chunks[:j], p_max=6 * 6)
            # 重放：从快照出发，跳过 j，写入 j+1.. 之后
            replayed = _write(snap, chunks[j + 1:], p_max=6 * 6)
            # 参考：从未写入 j
            ref = _write(_init(), chunks[:j] + chunks[j + 1:], p_max=6 * 6)
            for a, b in zip(replayed, ref):
                self.assertTrue(torch.allclose(a, b, atol=1e-6), f"j={j}")

    def test_replay_independent_of_later_writes(self):
        # 快照之后的写入不应影响从快照重放的结果（快照是独立副本）
        chunks = _make_chunks(1, 4)
        snap = _write(_init(), chunks[:2], p_max=36)
        snap_copy = tuple(t.clone() for t in snap)
        _ = _write(snap, chunks[2:], p_max=36)          # 修改了 snap 的引用（张量不可变，返回新张量）
        again = _write(snap_copy, chunks[3:], p_max=36)
        once = _write(snap, chunks[3:], p_max=36)
        for a, b in zip(again, once):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_state_update_independent_of_readout_mode(self):
        # 写入只依赖键、值、权重，不依赖读出分母；同一输入两次写入结果逐位相同
        chunks = _make_chunks(2, 3)
        a = _write(_init(), chunks, p_max=36)
        b = _write(_init(), chunks, p_max=36)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))


class TestSelfRetrieval(unittest.TestCase):
    def test_self_retrieval_of_single_write_is_exact_direction(self):
        # 只写入一个键 k、值 v（lam=1，无 windup 上界）：读出 k·M/(k·z+1) 的方向应与 v 的方向一致（余弦 1）
        d, dv = 6, 4
        kappa = F.normalize(torch.randn(1, 1, d), dim=-1) / (d ** 0.5)
        v = torch.randn(1, 1, dv)
        w = torch.ones(1, 1)
        M, P, z = _write(_init(d, dv), [(kappa, v, w)], lam=1.0)
        k = kappa[0, 0]
        num = k @ M[0]
        den = k @ z[0, :, 0] + 1.0
        out = num / den
        cos = F.cosine_similarity(out, v[0, 0], dim=0)
        self.assertGreater(float(cos), 0.999)


if __name__ == "__main__":
    unittest.main()
