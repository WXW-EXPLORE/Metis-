"""Phase 0 参考线性记忆的单元测试（见 docs/design_diagnostics.md §7）。

运行：PYTHONPATH=src python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from diagnostics.synthetic_memory import (
    crosstalk_probe,
    decay_probe,
    delete_projection,
    deletion_probe,
    make_target_and_distractors,
    simulate,
)


def random_unit_sequence(rng, L, d):
    k = rng.standard_normal((L, d))
    return k / np.linalg.norm(k, axis=1, keepdims=True)


class TestLinearity(unittest.TestCase):
    def test_linear_in_values_for_fixed_keys(self):
        rng = np.random.default_rng(0)
        L, d, dv, lam = 50, 8, 6, 0.9
        keys = random_unit_sequence(rng, L, d)
        v1 = rng.standard_normal((L, dv))
        v2 = rng.standard_normal((L, dv))
        a, b = 1.7, -0.4
        lhs = simulate(keys, a * v1 + b * v2, lam)
        rhs = a * simulate(keys, v1, lam) + b * simulate(keys, v2, lam)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_linear_in_initial_state(self):
        rng = np.random.default_rng(1)
        L, d, dv, lam = 30, 8, 6, 0.95
        keys = np.zeros((L, d))
        values = np.zeros((L, dv))
        M1 = rng.standard_normal((d, dv))
        M2 = rng.standard_normal((d, dv))
        lhs = simulate(keys, values, lam, M0=2.0 * M1 - M2)
        rhs = 2.0 * simulate(keys, values, lam, M0=M1) - simulate(keys, values, lam, M0=M2)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)


class TestZeroInput(unittest.TestCase):
    def test_zero_input_from_zero_state_stays_zero(self):
        M = simulate(np.zeros((10, 4)), np.zeros((10, 3)), 0.9)
        np.testing.assert_array_equal(M, np.zeros((4, 3)))

    def test_zero_input_decays_initial_state_geometrically(self):
        rng = np.random.default_rng(2)
        M0 = rng.standard_normal((4, 3))
        L, lam = 17, 0.8
        M = simulate(np.zeros((L, 4)), np.zeros((L, 3)), lam, M0=M0)
        np.testing.assert_allclose(M, lam**L * M0, atol=1e-12)


class TestStability(unittest.TestCase):
    def test_norm_bounded_by_max_value_norm_over_long_sequence(self):
        # 单位键时有 ||M_{t+1}||_F <= lam ||M_t||_F + (1-lam) ||v_t||，故 ||M_t||_F <= max_t ||v_t||
        rng = np.random.default_rng(3)
        L, d, dv, lam = 100_000, 16, 16, 0.9
        keys = random_unit_sequence(rng, L, d)
        values = rng.standard_normal((L, dv))
        M = simulate(keys, values, lam)
        self.assertTrue(np.all(np.isfinite(M)))
        self.assertLessEqual(np.linalg.norm(M), np.max(np.linalg.norm(values, axis=1)) + 1e-9)


class TestDecay(unittest.TestCase):
    def test_amplitude_matches_closed_form(self):
        rng = np.random.default_rng(4)
        k_c, v_c, _, _ = make_target_and_distractors(rng, 32, 32, 0, 0.0)
        for lam in (0.9, 0.99, 0.999):
            for age in (1, 10, 100):
                amp = decay_probe(k_c, v_c, lam, age)
                self.assertAlmostEqual(amp, (1 - lam) * lam**age, places=10)


class TestCrosstalk(unittest.TestCase):
    def test_crosstalk_matches_analytic_form(self):
        # 串扰 = rho * sum_j (1-lam) lam^{age-j} v_j（读出为 q=k_c 时 q·k_j = rho）
        d, dv, lam, age, n, rho = 32, 32, 0.95, 40, 5, 0.6
        rng = np.random.default_rng(5)
        out = crosstalk_probe(rng, d, dv, lam, age, n, rho)
        rng2 = np.random.default_rng(5)
        k_c, v_c, dk, dvals = make_target_and_distractors(rng2, d, dv, n, rho)
        idx = np.arange(age - n + 1, age + 1)
        coef = (1 - lam) * lam ** (age - idx)
        analytic = rho * np.sum(coef[:, None] * dvals, axis=0)
        self.assertAlmostEqual(out["crosstalk_norm"], np.linalg.norm(analytic), places=9)

    def test_zero_rho_gives_no_crosstalk(self):
        rng = np.random.default_rng(6)
        out = crosstalk_probe(rng, 32, 32, 0.9, 30, 8, 0.0)
        self.assertLess(out["crosstalk_norm"], 1e-12)
        self.assertTrue(np.isnan(out["snr"]))

    def test_distractor_cosine_equals_rho(self):
        rng = np.random.default_rng(7)
        for rho in (0.0, 0.3, 0.9):
            k_c, _, dk, _ = make_target_and_distractors(rng, 64, 64, 20, rho)
            np.testing.assert_allclose(dk @ k_c, rho, atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(dk, axis=1), 1.0, atol=1e-12)


class TestDeletion(unittest.TestCase):
    def test_projection_removes_target_exactly(self):
        rng = np.random.default_rng(8)
        d, dv = 32, 32
        M = rng.standard_normal((d, dv))
        k_hat = rng.standard_normal(d)
        k_hat /= np.linalg.norm(k_hat)
        M2 = delete_projection(M, k_hat)
        np.testing.assert_allclose(M2.T @ k_hat, 0.0, atol=1e-12)

    def test_projection_with_orthogonal_keys_leaves_others_unchanged(self):
        # rho=0 时干扰键与目标键正交，删除后其余读出应不变（恢复性）
        rng = np.random.default_rng(9)
        out = deletion_probe(rng, 64, 64, 0.99, 100, 8, 0.0, "projection")
        self.assertLess(out["collateral"], 1e-12)
        self.assertLess(out["target_residual"], 1e-12)

    def test_collateral_grows_with_key_correlation(self):
        vals = []
        for rho in (0.0, 0.3, 0.9):
            rng = np.random.default_rng(10)
            vals.append(deletion_probe(rng, 64, 64, 0.99, 100, 8, rho, "projection")["collateral"])
        self.assertLess(vals[0], vals[1])
        self.assertLess(vals[1], vals[2])

    def test_global_deletion_removes_target_and_others_equally(self):
        # 全局缩放对所有条目等比例作用，无法选择性删除
        rng = np.random.default_rng(11)
        out = deletion_probe(rng, 64, 64, 0.99, 100, 8, 0.6, "global", gamma=0.5)
        self.assertAlmostEqual(out["target_residual"], 0.5, places=10)
        self.assertAlmostEqual(out["collateral"], 0.5, places=10)


if __name__ == "__main__":
    unittest.main()
