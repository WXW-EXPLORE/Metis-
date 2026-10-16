"""1b：RLS commit 更新的单元测试（src/holistic/rls_metis.py）。

运行（仓库根目录）：PYTHONPATH=src python -m unittest discover -s tests -p "test_rls_metis.py" -v
"""
import unittest

import torch
import torch.nn.functional as F

from holistic.rls_metis import rls_commit_update


class TestCommitUpdate(unittest.TestCase):
    def test_lambda_one_matches_ridge_over_all_commits(self):
        # lam=1、P0=I/delta：多次 commit 后 M 等于带岭项 delta 的加权最小二乘解
        torch.manual_seed(0)
        d, dv, delta = 10, 4, 0.5
        B = 1
        M = torch.zeros(B, d, dv)
        P = torch.eye(d).expand(B, d, d).clone() / delta
        z = torch.zeros(B, d, 1)
        K_all, V_all, W_all = [], [], []
        for _ in range(3):
            k = F.normalize(torch.randn(B, 5, d), dim=-1)
            v = torch.randn(B, 5, dv)
            w = torch.ones(B, 5)
            M, P, z = rls_commit_update(M, P, z, k, v, w, lam=1.0)
            K_all.append(k[0]); V_all.append(v[0]); W_all.append(w[0])
        K = torch.cat(K_all); V = torch.cat(V_all)
        A = delta * torch.eye(d) + K.T @ K
        M_ref = torch.linalg.solve(A, K.T @ V)
        self.assertTrue(torch.allclose(M[0], M_ref, atol=1e-4))

    def test_two_commits_match_manual_recursion(self):
        # lam=0.8：与显式的信息形式递推 Lambda <- lam Lambda + K^T K 逐步一致
        torch.manual_seed(1)
        d, dv, lam = 6, 3, 0.8
        M = torch.zeros(1, d, dv)
        P = torch.eye(d).expand(1, d, d).clone()
        z = torch.zeros(1, d, 1)
        Lam = torch.eye(d)
        b = torch.zeros(d, dv)
        for _ in range(2):
            k = F.normalize(torch.randn(1, 4, d), dim=-1)
            v = torch.randn(1, 4, dv)
            w = torch.rand(1, 4) + 0.5
            M, P, z = rls_commit_update(M, P, z, k, v, w, lam=lam)
            Lam = lam * Lam + (k[0] * w[0, :, None]).T @ k[0]
            b = lam * b + (k[0] * w[0, :, None]).T @ v[0]
        self.assertTrue(torch.allclose(M[0], torch.linalg.solve(Lam, b), atol=1e-5))
        self.assertTrue(torch.allclose(P[0], torch.linalg.inv(Lam), atol=1e-5))

    def test_zero_weights_only_decay(self):
        # 权重全为 0（未选中的 chunk）：M 不变，P 按 1/lam 增长（遗忘），z 按 lam 衰减
        torch.manual_seed(2)
        d, dv, lam = 6, 3, 0.9
        M = torch.randn(1, d, dv)
        P = torch.eye(d).expand(1, d, d).clone()
        z = torch.randn(1, d, 1)
        k = F.normalize(torch.randn(1, 4, d), dim=-1)
        v = torch.randn(1, 4, dv)
        M2, P2, z2 = rls_commit_update(M, P, z, k, v, torch.zeros(1, 4), lam=lam)
        self.assertTrue(torch.allclose(M2, M, atol=1e-5))
        self.assertTrue(torch.allclose(z2, lam * z, atol=1e-6))
        self.assertTrue(torch.allclose(P2, P / lam, atol=1e-5))

    def test_outputs_are_float32_and_finite(self):
        torch.manual_seed(3)
        d, dv = 8, 4
        M = torch.zeros(2, d, dv, dtype=torch.bfloat16)
        P = torch.eye(d).expand(2, d, d).clone()
        z = torch.zeros(2, d, 1)
        k = F.normalize(torch.randn(2, 6, d), dim=-1).to(torch.bfloat16)
        v = torch.randn(2, 6, dv).to(torch.bfloat16)
        w = torch.ones(2, 6, dtype=torch.bfloat16)
        M2, P2, z2 = rls_commit_update(M, P, z, k, v, w, lam=0.9)
        for t in (M2, P2, z2):
            self.assertEqual(t.dtype, torch.float32)
            self.assertTrue(torch.isfinite(t).all())


if __name__ == "__main__":
    unittest.main()


class TestWindupGuard(unittest.TestCase):
    def test_max_eig_bounded_and_no_crash_under_forgetting(self):
        # 设计 0a §14：未激励方向的 windup 被特征值上界 p_max 限制；300 次 commit 不崩溃
        torch.manual_seed(4)
        kv = 256
        M = torch.zeros(1, kv, 16)
        P = (kv * torch.eye(kv)).expand(1, kv, kv).clone()
        z = torch.zeros(1, kv, 1)
        for _ in range(300):
            k = F.normalize(torch.randn(1, 16, kv), dim=-1) / kv ** 0.5
            v = torch.randn(1, 16, 16)
            M, P, z = rls_commit_update(M, P, z, k, v, torch.ones(1, 16), 0.9, p_max=kv * kv)
            self.assertLessEqual(torch.linalg.eigvalsh(P[0].double()).max().item(), kv * kv * (1 + 1e-4))
        self.assertTrue(torch.isfinite(M).all() and torch.isfinite(P).all())

    def test_low_rank_keys_do_not_crash(self):
        # 低秩键（24 维子空间）是崩溃的候选场景：应保持可求逆，最大特征值受限
        torch.manual_seed(5)
        kv, r = 512, 24
        U = torch.linalg.qr(torch.randn(kv, r))[0]
        M = torch.zeros(1, kv, 16)
        P = (kv * torch.eye(kv)).expand(1, kv, kv).clone()
        z = torch.zeros(1, kv, 1)
        for _ in range(250):
            k = F.normalize(torch.randn(1, 16, r) @ U.T, dim=-1) / kv ** 0.5
            M, P, z = rls_commit_update(M, P, z, k, torch.randn(1, 16, 16), torch.ones(1, 16), 0.9, p_max=kv * kv)
        self.assertTrue(torch.isfinite(P).all())
        self.assertLessEqual(torch.linalg.eigvalsh(P[0].double()).max().item(), kv * kv * (1 + 1e-4))


class TestWindupGradient(unittest.TestCase):
    def test_gradient_finite_with_repeated_eigenvalues(self):
        # P0 = kv I 的特征值重复；经过截断的前向梯度必须有限（训练第 0 步的崩溃场景）
        torch.manual_seed(6)
        kv = 64
        k = (F.normalize(torch.randn(1, 8, kv), dim=-1) / kv ** 0.5).requires_grad_(True)
        v = torch.randn(1, 8, 4)
        P = (kv * torch.eye(kv)).expand(1, kv, kv).clone()
        M, P2, z = rls_commit_update(torch.zeros(1, kv, 4), P, torch.zeros(1, kv, 1), k, v,
                                     torch.ones(1, 8), 0.9, p_max=kv * kv)
        M.pow(2).sum().backward()
        self.assertTrue(torch.isfinite(k.grad).all())
