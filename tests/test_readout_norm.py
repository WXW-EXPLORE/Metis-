"""读出分母的单元测试（设计：docs/design_readout_norm.md）。

检查：
  1. abs 分母恒 >= 1（含大幅负的 q·z）；
  2. orig 分母与原版 1 + q·z 逐位一致（回归）；
  3. abs 在 q·z >= 0 时与 orig 一致；
  4. rls_read(mode="orig") 与原版读出公式（metis_local_memory.read）一致；
  5. q·z = 0 处梯度有限；
  6. 分库路径 read_rows 的 mode 与 rls_read 一致（K=1）。

运行（仓库根目录）：PYTHONPATH=src python -m unittest discover -s tests -p "test_readout_norm.py" -v
"""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from holistic.rls_metis import readout_denominator, rls_read
from holistic.banked_metis import read_rows


def _orig_read_formula(q, state, key_state):
    """原版读出（metis/dev_beta/metis_local_memory.py:175–194，num_kv_groups=1 分支）的手工复现。"""
    bsz, _h, seq_len, _d = q.shape
    q = F.normalize(q, dim=-1)
    q_flat = q.transpose(1, 2).reshape(bsz, seq_len, -1)
    out = torch.matmul(q_flat, state)
    norm_factor = torch.matmul(q_flat, key_state)
    return (out / (norm_factor + 1.0)).contiguous()


class TestDenominator(unittest.TestCase):
    def test_abs_denominator_at_least_one(self):
        torch.manual_seed(0)
        qz = torch.randn(10000) * 50.0
        den = readout_denominator(qz, "abs")
        self.assertTrue(bool((den >= 1.0).all()))

    def test_orig_denominator_matches_original_formula(self):
        torch.manual_seed(1)
        qz = torch.randn(1000) * 3.0
        self.assertTrue(torch.equal(readout_denominator(qz, "orig"), qz + 1.0))

    def test_abs_equals_orig_when_nonnegative(self):
        torch.manual_seed(2)
        qz = torch.rand(1000) * 5.0
        self.assertTrue(torch.allclose(readout_denominator(qz, "abs"), readout_denominator(qz, "orig")))

    def test_abs_is_one_plus_abs_for_negative_values(self):
        qz = torch.tensor([-3.0, -1.0, -0.5, 0.0, 2.0])
        expected = torch.tensor([4.0, 2.0, 1.5, 1.0, 3.0])
        self.assertTrue(torch.allclose(readout_denominator(qz, "abs"), expected))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            readout_denominator(torch.zeros(1), "bogus")


def _fake_lm(bsz, kv, groups=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(bsz, kv, kv, generator=g) * 0.1
    key_state = torch.randn(bsz, kv, 1, generator=g) * 0.5
    return SimpleNamespace(kv_dim=kv, num_kv_groups=groups, q_dim=kv * groups,
                           _state=state, _key_state=key_state)


class TestRlsRead(unittest.TestCase):
    def test_orig_matches_original_formula(self):
        torch.manual_seed(3)
        bsz, h, seq, d, kv = 2, 4, 5, 8, 8
        lm = _fake_lm(bsz, kv)
        q = torch.randn(bsz, h, seq, d)
        # 原版公式要求 q_dim = h*d = kv（num_kv_groups=1 时 kv_dim 即 q_flat 的最后一维）
        lm.kv_dim = h * d
        lm._state = torch.randn(bsz, h * d, h * d) * 0.1
        lm._key_state = torch.randn(bsz, h * d, 1) * 0.5
        out = rls_read(lm, q, "orig")
        ref = _orig_read_formula(q, lm._state, lm._key_state)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6))

    def test_abs_is_finite_and_bounded_where_orig_diverges(self):
        torch.manual_seed(4)
        bsz, h, seq, d = 1, 2, 3, 4
        lm = _fake_lm(bsz, h * d)
        lm.kv_dim = h * d
        # 人为构造 q·z 接近 -1（原版分母接近零）
        q = torch.randn(bsz, h, seq, d)
        qf = F.normalize(q, dim=-1).transpose(1, 2).reshape(bsz, seq, -1)
        # 令第一个查询的 q·z = -1（原版分母为零）
        z = -qf[0, 0] / (qf[0, 0] @ qf[0, 0])
        lm._key_state = z.reshape(bsz, h * d, 1)
        self.assertAlmostEqual(float(qf[0, 0] @ lm._key_state[0, :, 0]), -1.0, places=5)
        out_abs = rls_read(lm, q, "abs")
        self.assertTrue(bool(torch.isfinite(out_abs).all()))

    def test_gradient_finite_at_zero_inner_product(self):
        torch.manual_seed(5)
        bsz, h, seq, d = 1, 2, 2, 4
        lm = _fake_lm(bsz, h * d)
        lm.kv_dim = h * d
        lm._key_state = torch.zeros(bsz, h * d, 1)  # q·z = 0 for every query
        q = torch.randn(bsz, h, seq, d, requires_grad=True)
        out = rls_read(lm, q, "abs")
        out.sum().backward()
        self.assertTrue(bool(torch.isfinite(q.grad).all()))

    def test_gqa_path_shape(self):
        torch.manual_seed(6)
        bsz, h, seq, d = 1, 4, 3, 4
        kv = 8
        lm = SimpleNamespace(kv_dim=kv, num_kv_groups=2, q_dim=h * d,
                             _state=torch.randn(bsz, kv, kv) * 0.1, _key_state=torch.randn(bsz, kv, 1) * 0.5)
        q = torch.randn(bsz, h, seq, d)
        out = rls_read(lm, q, "abs")
        self.assertEqual(tuple(out.shape), (bsz, seq, h * d))


class TestBankedRead(unittest.TestCase):
    def test_read_rows_k1_matches_rls_read_in_both_modes(self):
        torch.manual_seed(7)
        kv = 8
        M = torch.randn(1, kv, kv) * 0.1
        z = torch.randn(1, kv, 1) * 0.5
        P = torch.eye(kv).expand(1, kv, kv).clone()
        lm = SimpleNamespace(_banks=[[M, P, z]])
        rows = F.normalize(torch.randn(6, kv), dim=-1)  # read_rows 的输入已在调用方归一化
        for mode in ("orig", "abs"):
            out = read_rows(lm, rows, None, None, kv, 1, mode)
            num = rows @ M[0]
            den = readout_denominator(rows @ z[0], mode)
            self.assertTrue(torch.allclose(out, num / den, atol=1e-6), mode)


if __name__ == "__main__":
    unittest.main()
