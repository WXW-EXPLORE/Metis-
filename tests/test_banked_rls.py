"""0a-v2：RLS 更新规则的单元测试（见 docs/design_0a.md §10.7）。

运行（仓库根目录）：PYTHONPATH=src python -m unittest discover -s tests -p "test_banked_rls.py" -v
"""
import unittest

import torch
import torch.nn.functional as F

from holistic.banked_memory import BankedLinearMemory, gram_spread_loss, weighted_gram


def ridge_solution(keys, vals, delta):
    """(delta I + K^T K)^{-1} K^T V，K 为单位键矩阵 (T,d)，V (T,dv)。"""
    d = keys.shape[1]
    A = delta * torch.eye(d, dtype=keys.dtype) + keys.T @ keys
    return torch.linalg.solve(A, keys.T @ vals)


class TestRidgeEquivalence(unittest.TestCase):
    def test_lambda_one_matches_ridge_least_squares(self):
        # 主检验：lambda=1 时 RLS 状态等于带岭项 delta 的最小二乘解（设计 §10.7-1）
        torch.manual_seed(0)
        delta = 0.5
        mem = BankedLinearMemory(12, 6, n_banks=1, alphas=[1.0], update_rule="rls", p0=1.0 / delta).double()
        keys = torch.randn(40, 12, dtype=torch.float64)
        vals = torch.randn(40, 6, dtype=torch.float64)
        mem.write(keys[None], vals[None])
        Kn = F.normalize(keys, dim=-1)
        M_ref = ridge_solution(Kn, vals, delta)
        self.assertTrue(torch.allclose(mem.state[0, 0], M_ref, atol=1e-8))

    def test_covariance_matches_inverse_of_regularized_gram(self):
        torch.manual_seed(1)
        delta = 0.3
        mem = BankedLinearMemory(10, 4, n_banks=1, alphas=[1.0], update_rule="rls", p0=1.0 / delta).double()
        keys = torch.randn(25, 10, dtype=torch.float64)
        mem.write(keys[None], torch.randn(1, 25, 4, dtype=torch.float64))
        Kn = F.normalize(keys, dim=-1)
        P_ref = torch.linalg.inv(delta * torch.eye(10, dtype=torch.float64) + Kn.T @ Kn)
        self.assertTrue(torch.allclose(mem.P[0, 0], P_ref, atol=1e-8))


class TestPole(unittest.TestCase):
    def test_zero_value_write_scales_key_direction_by_lambda_over_lambda_plus_s(self):
        # 有效极点 lambda / (lambda + s)，s = kappa^T P kappa（设计 §10.2）
        torch.manual_seed(2)
        lam = 0.9
        mem = BankedLinearMemory(8, 4, n_banks=1, alphas=[lam], update_rule="rls").double()
        mem.write(torch.randn(1, 3, 8, dtype=torch.float64), torch.randn(1, 3, 4, dtype=torch.float64))
        kappa = F.normalize(torch.randn(8, dtype=torch.float64), dim=0)[None, None]
        s = (kappa[0, 0] @ mem.P[0, 0] @ kappa[0, 0]).item()
        before = (kappa[0, 0] @ mem.state[0, 0])           # kappa^T M  (dv,)
        mem._rls_step(kappa, torch.ones(1, 1, dtype=torch.float64),
                      torch.zeros(1, 4, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
        after = kappa[0, 0] @ mem.state[0, 0]
        ratio = (after / before).mean().item()
        self.assertAlmostEqual(ratio, lam / (lam + s), places=8)


class TestCovarianceStability(unittest.TestCase):
    def test_covariance_stays_positive_definite_and_clamped(self):
        torch.manual_seed(3)
        mem = BankedLinearMemory(8, 4, n_banks=2, alphas=[0.9, 0.95], update_rule="rls",
                                 p0=1.0, p_max=5.0).double()
        mem.write(torch.randn(1, 2000, 8, dtype=torch.float64), torch.randn(1, 2000, 4, dtype=torch.float64))
        eig = torch.linalg.eigvalsh(mem.P)
        self.assertGreater(eig.min().item(), 0.0)
        tr = mem.P.diagonal(dim1=-2, dim2=-1).sum(-1)
        self.assertLessEqual(tr.max().item(), 5.0 + 1e-8)
        self.assertTrue(torch.isfinite(mem.state).all())

    def test_windup_without_excitation_is_clamped(self):
        # 只激励一个方向，其余方向的 P 按 1/lambda 增长，应被 p_max 限制
        mem = BankedLinearMemory(8, 4, n_banks=1, alphas=[0.8], update_rule="rls", p0=1.0, p_max=10.0).double()
        e0 = torch.zeros(1, 1, 8, dtype=torch.float64)
        e0[..., 0] = 1.0
        mem.write(e0.expand(1, 500, 8).contiguous(), torch.randn(1, 500, 4, dtype=torch.float64))
        self.assertGreater(mem.clamp_count, 0)
        self.assertLessEqual(mem.P.diagonal(dim1=-2, dim2=-1).sum(-1).max().item(), 10.0 + 1e-8)


class TestDowndate(unittest.TestCase):
    def test_lambda_one_delete_restores_ridge_without_the_item(self):
        # 降秩在 lambda=1 时精确：删除第 0 条后，状态应等于其余条目的岭解（设计 §10.4）
        torch.manual_seed(4)
        delta = 0.5
        mem = BankedLinearMemory(12, 6, n_banks=1, alphas=[1.0], update_rule="rls", p0=1.0 / delta).double()
        keys = torch.randn(20, 12, dtype=torch.float64)
        vals = torch.randn(20, 6, dtype=torch.float64)
        mem.write(keys[None], vals[None])
        Kn = F.normalize(keys, dim=-1)
        mem.delete(Kn[0:1], beta_d=1.0, value=vals[0:1])
        M_ref = ridge_solution(Kn[1:], vals[1:], delta)
        self.assertTrue(torch.allclose(mem.state[0, 0], M_ref, atol=1e-8))
        P_ref = torch.linalg.inv(delta * torch.eye(12, dtype=torch.float64) + Kn[1:].T @ Kn[1:])
        self.assertTrue(torch.allclose(mem.P[0, 0], P_ref, atol=1e-7))
        self.assertEqual(mem.fallback_count, 0)

    def test_fallback_to_projection_when_condition_fails(self):
        # 若 w kappa^T P kappa >= 1，退回投影删除并计数（设计 §10.4）
        # 删除一个从未写入的键：其 s = kappa^T P kappa 约为 p0 > 1，应退回投影并计数
        mem = BankedLinearMemory(8, 4, n_banks=1, alphas=[1.0], update_rule="rls", p0=50.0).double()
        keys = F.normalize(torch.randn(1, 2, 8, dtype=torch.float64), dim=-1)
        mem.write(keys, torch.randn(1, 2, 4, dtype=torch.float64))
        unseen = F.normalize(torch.randn(1, 8, dtype=torch.float64), dim=-1)
        mem.delete(unseen, beta_d=1.0, value=torch.randn(1, 4, dtype=torch.float64))
        self.assertGreater(mem.fallback_count, 0)
        self.assertTrue(torch.isfinite(mem.state).all())


    def test_lambda_below_one_delete_uses_projection_and_removes_target_exactly(self):
        # lambda<1 时退回投影删除：目标读出精确为零（设计 0a §10.4 修订，N6）
        torch.manual_seed(9)
        mem = BankedLinearMemory(8, 4, n_banks=1, alphas=[0.99], update_rule="rls", p0=1.0).double()
        keys = F.normalize(torch.randn(1, 6, 8, dtype=torch.float64), dim=-1)
        vals = torch.randn(1, 6, 4, dtype=torch.float64)
        mem.write(keys, vals)
        mem.delete(keys[:, 0], beta_d=1.0, value=vals[:, 0])
        self.assertLess(mem.read(keys[:, :1])[0, 0].norm().item(), 1e-10)
        self.assertGreater(mem.fallback_count, 0)


class TestGradientsAndGram(unittest.TestCase):
    def test_gradients_reach_centers_through_rls_write(self):
        torch.manual_seed(5)
        mem = BankedLinearMemory(16, 8, n_banks=3, alphas=[0.9, 0.95, 0.99], top_k=2, update_rule="rls")
        beta = torch.full((1, 5), 0.5, requires_grad=True)
        mem.write(torch.randn(1, 5, 16), torch.randn(1, 5, 8), beta=beta)
        y = mem.read(torch.randn(1, 16))
        y.pow(2).sum().backward()
        self.assertGreater(mem.centers.grad.abs().sum().item(), 0.0)
        self.assertGreater(beta.grad.abs().sum().item(), 0.0)

    def test_gram_spread_zero_for_full_orthonormal_basis(self):
        # 8 个正交单位键张满 8 维、权重相同：W 正比于 I，损失为 0
        Q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
        self.assertLess(gram_spread_loss(weighted_gram(Q.T, 1.0)).item(), 1e-10)

    def test_gram_spread_positive_for_correlated_keys(self):
        corr = F.normalize(torch.ones(6, 8, dtype=torch.float64) + 0.05 * torch.randn(6, 8, dtype=torch.float64), dim=-1)
        self.assertGreater(gram_spread_loss(weighted_gram(corr, 0.9)).item(), 1.0)

    def test_gram_spread_is_basis_invariant(self):
        # 旋转坐标基不改变损失（非对角损失不具备此性质）
        torch.manual_seed(6)
        keys = F.normalize(torch.randn(10, 6, dtype=torch.float64), dim=-1)
        R, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))
        a = gram_spread_loss(weighted_gram(keys, 0.95))
        b = gram_spread_loss(weighted_gram(keys @ R, 0.95))
        self.assertTrue(torch.allclose(a, b, atol=1e-10))


if __name__ == "__main__":
    unittest.main()


class TestRoutingAndProjectionModes(unittest.TestCase):
    def test_straight_through_gives_gradient_to_centers_under_hard_routing(self):
        torch.manual_seed(10)
        mem = BankedLinearMemory(8, 4, n_banks=4, top_k=1, straight_through=True)
        mem.write(torch.randn(1, 3, 8), torch.randn(1, 3, 4))
        # 线性损失：若查询落入空库，y=0，y^2 的梯度会退化为 0，因此不用平方
        mem.read(torch.randn(1, 8)).sum().backward()
        self.assertGreater(mem.centers.grad.abs().sum().item(), 0.0)

    def test_hard_routing_without_straight_through_has_no_center_gradient(self):
        # 对照：不加直通估计时，one-hot 的 softmax 梯度为 0（说明直通估计是必要的）
        torch.manual_seed(10)
        mem = BankedLinearMemory(8, 4, n_banks=4, top_k=1, straight_through=False)
        mem.write(torch.randn(1, 3, 8), torch.randn(1, 3, 4))
        mem.read(torch.randn(1, 8)).pow(2).sum().backward()
        self.assertEqual(mem.centers.grad.abs().sum().item(), 0.0)

    def test_load_balance_loss_minimal_for_uniform_routing(self):
        mem = BankedLinearMemory(8, 4, n_banks=2)
        with torch.no_grad():
            mem.centers.zero_()           # 所有中心相同 -> 路由完全均匀
        kn = F.normalize(torch.randn(100, 8), dim=-1)
        self.assertAlmostEqual(mem.load_balance_loss(kn).item(), 1.0, places=5)

    def test_block_projection_selects_disjoint_coordinates(self):
        mem = BankedLinearMemory(16, 4, n_banks=4, bank_key_dim=4, proj_mode="block")
        for b in range(4):
            idx = mem.proj[b].argmax(dim=-1)
            self.assertEqual(sorted(idx.tolist()), list(range(4 * b, 4 * b + 4)))


class TestForgettingWeightedDowndate(unittest.TestCase):
    def test_exact_delete_with_forgetting_matches_null_step(self):
        # lambda<1：删除第 j 条（年龄 age=T-j）后，应等于把第 j 步换成零权重写入的结果
        torch.manual_seed(11)
        lam, T, j = 0.9, 12, 4
        keys = F.normalize(torch.randn(1, T, 8, dtype=torch.float64), dim=-1)
        vals = torch.randn(1, T, 4, dtype=torch.float64)
        beta = torch.ones(1, T, dtype=torch.float64)

        full = BankedLinearMemory(8, 4, n_banks=1, alphas=[lam], update_rule="rls", p0=1.0).double()
        full.write(keys, vals, beta=beta)
        full.delete(keys[:, j], beta_d=1.0, value=vals[:, j], age=T - (j + 1))

        beta_null = beta.clone()
        beta_null[0, j] = 0.0
        ref = BankedLinearMemory(8, 4, n_banks=1, alphas=[lam], update_rule="rls", p0=1.0).double()
        ref.write(keys, vals, beta=beta_null)

        self.assertTrue(torch.allclose(full.state, ref.state, atol=1e-8))
        self.assertTrue(torch.allclose(full.P, ref.P, atol=1e-8))
        self.assertTrue(torch.allclose(full.z, ref.z, atol=1e-8))
        self.assertEqual(full.fallback_count, 0)


class TestChunkForm(unittest.TestCase):
    def test_chunk_write_equals_sequential_rls(self):
        # 块形式（信息形式闭式）与逐 token RLS 数值等价
        torch.manual_seed(12)
        lam = 0.97
        k = torch.randn(2, 40, 12, dtype=torch.float64)
        v = torch.randn(2, 40, 6, dtype=torch.float64)
        beta = torch.rand(2, 40, dtype=torch.float64) + 0.1
        seq = BankedLinearMemory(12, 6, n_banks=1, alphas=[lam], update_rule="rls", p0=0.5).double()
        chk = BankedLinearMemory(12, 6, n_banks=1, alphas=[lam], update_rule="rls", p0=0.5).double()
        seq.write(k[:, :15], v[:, :15], beta=beta[:, :15])
        seq.write(k[:, 15:], v[:, 15:], beta=beta[:, 15:])
        chk.write_chunk(k[:, :15], v[:, :15], beta=beta[:, :15])
        chk.write_chunk(k[:, 15:], v[:, 15:], beta=beta[:, 15:])
        self.assertTrue(torch.allclose(seq.state, chk.state, atol=1e-8))
        self.assertTrue(torch.allclose(seq.P, chk.P, atol=1e-8))
        self.assertTrue(torch.allclose(seq.z, chk.z, atol=1e-8))
