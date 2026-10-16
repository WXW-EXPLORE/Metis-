"""0a 参考记忆的单元测试（见 docs/design_0a.md §6）。

运行（仓库根目录）：PYTHONPATH=src python -m unittest discover -s tests -p "test_banked_memory.py" -v
"""
import unittest

import torch
import torch.nn.functional as F

from holistic.banked_memory import BankedLinearMemory


def reference_gdn_single_bank(keys, values, alpha, beta=1.0):
    """独立实现的单库 GDN 递推（与 Metis 的形式一致），用作对照。

    keys: (T,d) 单个序列；返回 (M, z)，M: (d,dv)，z: (d,1)。
    """
    d = keys.shape[1]
    dv = values.shape[1]
    M = torch.zeros(d, dv, dtype=keys.dtype)
    z = torch.zeros(d, 1, dtype=keys.dtype)
    for t in range(keys.shape[0]):
        k = F.normalize(keys[t], dim=0)
        v = values[t]
        M = alpha * (M - beta * torch.outer(k, k @ M)) + beta * torch.outer(k, v)
        z = alpha * (z - beta * (k @ z) * k[:, None]) + beta * k[:, None]
    return M, z


class TestDefaultMatchesReference(unittest.TestCase):
    def test_single_bank_default_matches_gdn_reference(self):
        torch.manual_seed(0)
        mem = BankedLinearMemory(key_dim=16, value_dim=8, n_banks=1, alphas=[0.9]).double()
        keys = torch.randn(30, 16, dtype=torch.float64)
        vals = torch.randn(30, 8, dtype=torch.float64)
        mem.write(keys[None], vals[None])
        M_ref, z_ref = reference_gdn_single_bank(keys, vals, alpha=0.9)
        self.assertTrue(torch.allclose(mem.state[0, 0], M_ref, atol=1e-10))
        self.assertTrue(torch.allclose(mem.z[0, 0], z_ref, atol=1e-10))

    def test_single_bank_read_matches_reference(self):
        torch.manual_seed(1)
        mem = BankedLinearMemory(key_dim=16, value_dim=8, n_banks=1, alphas=[0.95]).double()
        keys = torch.randn(20, 16, dtype=torch.float64)
        vals = torch.randn(20, 8, dtype=torch.float64)
        mem.write(keys[None], vals[None])
        q = torch.randn(1, 16, dtype=torch.float64)
        M_ref, z_ref = reference_gdn_single_bank(keys, vals, alpha=0.95)
        qn = F.normalize(q[0], dim=0)
        y_ref = (qn @ M_ref) / (qn @ z_ref + 1.0)
        self.assertTrue(torch.allclose(mem.read(q)[0], y_ref, atol=1e-10))


class TestLinearity(unittest.TestCase):
    def test_state_is_affine_in_initial_state_for_fixed_routing(self):
        # 固定路由与 alpha、beta 时，写入后的状态对初始状态仿射
        torch.manual_seed(2)
        keys = torch.randn(1, 10, 16, dtype=torch.float64)
        vals = torch.randn(1, 10, 8, dtype=torch.float64)

        def run(init):
            mem = BankedLinearMemory(16, 8, n_banks=3, alphas=[0.9, 0.95, 0.99], seed=5).double()
            mem.write(keys[:, :1], vals[:, :1])        # 先写一次以建立状态形状
            mem._state = init.clone()                  # 替换为指定初始状态
            mem.write(keys[:, 1:], vals[:, 1:])
            return mem.state

        torch.manual_seed(3)
        M1 = torch.randn(1, 3, 16, 8, dtype=torch.float64)
        M2 = torch.randn(1, 3, 16, 8, dtype=torch.float64)
        zero = torch.zeros_like(M1)
        g1 = run(M1) - run(zero)
        g2 = run(M2) - run(zero)
        g12 = run(M1 + M2) - run(zero)
        self.assertTrue(torch.allclose(g12, g1 + g2, atol=1e-9))


class TestZeroInputAndStability(unittest.TestCase):
    def test_no_write_leaves_zero_state_zero(self):
        mem = BankedLinearMemory(16, 8, n_banks=2)
        y = mem.read(torch.randn(3, 16))
        self.assertTrue(torch.equal(y, torch.zeros_like(y)))
        self.assertTrue(torch.equal(mem.state, torch.zeros_like(mem.state)))

    def test_zero_value_write_decays_state_by_alpha(self):
        mem = BankedLinearMemory(16, 8, n_banks=2, alphas=[0.8, 0.9], seed=0).double()
        torch.manual_seed(4)
        mem._ensure(1, torch.device("cpu"), torch.float64)
        mem._state = torch.randn(1, 2, 16, 8, dtype=torch.float64)
        before = mem.state.clone()
        # 键为零向量时 kappa 归一化后仍为单位向量，因此用零值写入：只有衰减与擦除
        mem.write(torch.randn(1, 1, 16, dtype=torch.float64), torch.zeros(1, 1, 8, dtype=torch.float64), beta=0.0)
        self.assertTrue(torch.allclose(mem.state[0, 0], 0.8 * before[0, 0], atol=1e-12))
        self.assertTrue(torch.allclose(mem.state[0, 1], 0.9 * before[0, 1], atol=1e-12))

    def test_long_sequence_stays_finite_and_bounded(self):
        torch.manual_seed(5)
        mem = BankedLinearMemory(16, 8, n_banks=4, alphas=[0.7, 0.9, 0.99, 0.999]).double()
        keys = torch.randn(1, 10000, 16, dtype=torch.float64)
        vals = torch.randn(1, 10000, 8, dtype=torch.float64)
        mem.write(keys, vals)
        self.assertTrue(torch.isfinite(mem.state).all())
        self.assertTrue(torch.isfinite(mem.z).all())
        # 单位键、beta=1 时，每库 ||M||_F <= max||v|| / (1 - alpha) 的保守上界
        bound = vals.norm(dim=-1).max().item() / (1 - 0.999)
        self.assertLess(mem.state.abs().max().item(), bound)


class TestDeleteRecovery(unittest.TestCase):
    def test_delete_removes_target_and_keeps_orthogonal_items(self):
        torch.manual_seed(6)
        d = 32
        mem = BankedLinearMemory(d, 8, n_banks=1, alphas=[1.0]).double()
        # 构造正交键：目标键与其余键正交
        Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
        keys = Q[:, :6].T[None]                         # (1,6,d)，两两正交
        vals = torch.randn(1, 6, 8, dtype=torch.float64)
        mem.write(keys, vals)
        target = keys[0, 0][None]                       # (1,d)
        others = keys[0, 1:]
        read_before = mem.read(others[None])
        mem.delete(target, beta_d=1.0)
        self.assertTrue(torch.allclose(mem.read(target)[0], torch.zeros(8, dtype=torch.float64), atol=1e-10))
        self.assertTrue(torch.allclose(mem.read(others[None]), read_before, atol=1e-10))

    def test_delete_does_not_change_alpha(self):
        mem = BankedLinearMemory(16, 8, n_banks=2, alphas=[0.8, 0.9])
        mem.write(torch.randn(1, 3, 16), torch.randn(1, 3, 8))
        alpha_before = mem.alpha.clone()
        mem.delete(torch.randn(1, 16))
        self.assertTrue(torch.equal(mem.alpha, alpha_before))


class TestDecoupling(unittest.TestCase):
    def test_orthogonal_centers_have_zero_loss(self):
        mem = BankedLinearMemory(8, 4, n_banks=4)
        with torch.no_grad():
            q, _ = torch.linalg.qr(torch.randn(8, 4))
            mem.centers.copy_(q.T)
        self.assertLess(mem.decoupling_loss().item(), 1e-10)

    def test_single_bank_has_zero_loss(self):
        self.assertLess(BankedLinearMemory(8, 4, n_banks=1).decoupling_loss().item(), 1e-12)


class TestBatchInterface(unittest.TestCase):
    def test_batch_mismatch_raises_instead_of_clearing_state(self):
        mem = BankedLinearMemory(16, 8, n_banks=2)
        mem.write(torch.randn(1, 3, 16), torch.randn(1, 3, 8))
        before = mem.state.clone()
        with self.assertRaises(ValueError):
            mem.read(torch.randn(4, 16))
        with self.assertRaises(ValueError):
            mem.write(torch.randn(2, 1, 16), torch.randn(2, 1, 8))
        self.assertTrue(torch.equal(mem.state, before))

    def test_multi_query_read_matches_single_queries(self):
        torch.manual_seed(8)
        mem = BankedLinearMemory(16, 8, n_banks=3, top_k=2).double()
        mem.write(torch.randn(1, 6, 16, dtype=torch.float64), torch.randn(1, 6, 8, dtype=torch.float64))
        q = torch.randn(1, 5, 16, dtype=torch.float64)
        multi = mem.read(q)
        self.assertEqual(tuple(multi.shape), (1, 5, 8))
        for i in range(5):
            self.assertTrue(torch.allclose(multi[:, i], mem.read(q[:, i]), atol=1e-12))


class TestGradientsAndInterface(unittest.TestCase):
    def test_gradients_reach_centers_alpha_and_beta(self):
        torch.manual_seed(7)
        mem = BankedLinearMemory(16, 8, n_banks=3, alphas=[0.9, 0.95, 0.99], top_k=2)
        beta = torch.full((1, 6), 0.5, requires_grad=True)
        keys = torch.randn(1, 6, 16)
        vals = torch.randn(1, 6, 8)
        mem.write(keys, vals, beta=beta)
        y = mem.read(torch.randn(1, 16))
        loss = y.pow(2).sum() + mem.decoupling_loss()
        loss.backward()
        self.assertIsNotNone(mem.centers.grad)
        self.assertGreater(mem.centers.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(beta.grad)
        self.assertGreater(beta.grad.abs().sum().item(), 0.0)

    def test_shapes_follow_interface(self):
        mem = BankedLinearMemory(16, 8, n_banks=4, bank_key_dim=4)
        mem.write(torch.randn(2, 5, 16), torch.randn(2, 5, 8))
        self.assertEqual(tuple(mem.state.shape), (2, 4, 4, 8))
        self.assertEqual(tuple(mem.z.shape), (2, 4, 4, 1))
        self.assertEqual(tuple(mem.read(torch.randn(2, 16)).shape), (2, 8))

    def test_reset_clears_state(self):
        mem = BankedLinearMemory(16, 8, n_banks=2)
        mem.write(torch.randn(1, 2, 16), torch.randn(1, 2, 8))
        mem.reset()
        self.assertIsNone(mem.state)
        self.assertIsNone(mem.z)


class TestRoutingAndProjection(unittest.TestCase):
    def test_top_k_routing_has_k_nonzero_weights(self):
        mem = BankedLinearMemory(16, 8, n_banks=5, top_k=2)
        g = mem.route(F.normalize(torch.randn(7, 16), dim=-1))
        self.assertTrue(torch.all((g > 0).sum(dim=-1) == 2))
        self.assertTrue(torch.allclose(g.sum(dim=-1), torch.ones(7), atol=1e-6))

    def test_projection_is_row_orthonormal(self):
        mem = BankedLinearMemory(16, 8, n_banks=3, bank_key_dim=4)
        for b in range(3):
            P = mem.proj[b]
            self.assertTrue(torch.allclose(P @ P.T, torch.eye(4), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
