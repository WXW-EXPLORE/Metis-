"""Phase 0b 消融替换的单元测试（见 docs/design_phase0b.md §7）。

参照实现逐行转录自 metis/dev_beta/metis_hyper_memory.py 的 _apply_gated_delta_rule_update。
运行：/hy-tmp/venv-metis/bin/python -m unittest discover -s tests -v   （仓库根目录）
"""
import copy
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from diagnostics.gate_ablation import gated_delta_update


class Stub(nn.Module):
    def __init__(self, d_in=16, kv=8):
        super().__init__()
        self.kv_dim = kv
        self.update_ratio = 0.9
        self.W_k = nn.Linear(d_in, kv, bias=False)
        self.W_v = nn.Linear(d_in, kv, bias=False)
        self.gated_delta_alpha = nn.Linear(d_in, 1, bias=True)
        self.gated_delta_beta = nn.Linear(d_in, 1, bias=True)


class LocalMem:
    def __init__(self, state, key_state):
        self.state = state
        self.key_state = key_state
        self.written = None

    def write(self, new_state, new_key_state=None):
        self.state = new_state
        if new_key_state is not None:
            self.key_state = new_key_state
        self.written = True


def reference(mod, h, w, state, key_state):
    """逐行转录的原始公式，返回 (new_state, new_key_state)。"""
    write_key = F.normalize(mod.W_k(h), dim=-1) / (mod.kv_dim ** 0.5)
    write_value = mod.W_v(h)
    w = w.to(write_key.dtype)
    weight_mass = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
    alpha_gate = torch.sigmoid(mod.gated_delta_alpha(h).squeeze(-1))
    beta_gate = torch.sigmoid(mod.gated_delta_beta(h).squeeze(-1))
    alpha = (w * alpha_gate).sum(dim=1) / weight_mass.squeeze(1)
    beta = w * (mod.update_ratio * beta_gate)
    beta_exp = beta.unsqueeze(-1)
    bsz = write_key.size(0)
    key_memory = torch.matmul(write_key, state)
    erase_state = torch.matmul(write_key.transpose(-1, -2), beta_exp * key_memory)
    add_state = torch.matmul(write_key.transpose(-1, -2), beta_exp * write_value)
    new_state = alpha.view(bsz, 1, 1) * (state - erase_state) + add_state
    key_memory_mass = torch.matmul(write_key, key_state)
    erase_key = torch.matmul(write_key.transpose(-1, -2), beta_exp * key_memory_mass)
    add_key = torch.matmul(write_key.transpose(-1, -2), beta_exp)
    new_key = alpha.view(bsz, 1, 1) * (key_state - erase_key) + add_key
    return new_state, new_key


def make_inputs(seed, bsz=2, T=5, d_in=16, kv=8):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(bsz, T, d_in, generator=g, dtype=torch.float64)
    w = torch.rand(bsz, T, generator=g, dtype=torch.float64)
    state = torch.randn(bsz, kv, kv, generator=g, dtype=torch.float64)
    key_state = torch.randn(bsz, kv, 1, generator=g, dtype=torch.float64)
    return h, w, state, key_state


class TestFullMatchesReference(unittest.TestCase):
    def test_full_mode_is_bitwise_identical_to_reference(self):
        torch.manual_seed(0)
        mod = Stub().double()
        h, w, state, key_state = make_inputs(1)
        lm = LocalMem(state.clone(), key_state.clone())
        gated_delta_update(mod, h, w, lm, mode="full")
        ref_state, ref_key = reference(mod, h, w, state, key_state)
        self.assertTrue(torch.equal(lm.state, ref_state))
        self.assertTrue(torch.equal(lm.key_state, ref_key))


class TestAblations(unittest.TestCase):
    def test_no_erase_matches_direct_formula(self):
        # 直接公式：M' = alpha*M + A，且键的累计状态 z' = alpha*z + add_z（无擦除）
        torch.manual_seed(0)
        mod = Stub().double()
        h, w, state, key_state = make_inputs(2)
        lm = LocalMem(state.clone(), key_state.clone())
        gated_delta_update(mod, h, w, lm, mode="no_erase")

        write_key = F.normalize(mod.W_k(h), dim=-1) / (mod.kv_dim ** 0.5)
        write_value = mod.W_v(h)
        ww = w.to(write_key.dtype)
        alpha = _alpha(mod, h, w).view(-1, 1, 1)
        beta = ww * (mod.update_ratio * torch.sigmoid(mod.gated_delta_beta(h).squeeze(-1)))
        beta_exp = beta.unsqueeze(-1)
        A = write_key.transpose(-1, -2) @ (beta_exp * write_value)
        add_z = write_key.transpose(-1, -2) @ beta_exp
        self.assertTrue(torch.allclose(lm.state, alpha * state + A, atol=1e-12))
        self.assertTrue(torch.allclose(lm.key_state, alpha * key_state + add_z, atol=1e-12))

    def test_no_forget_matches_direct_formula(self):
        # 直接公式：M' = M - E + A，E = K^T beta K M
        torch.manual_seed(0)
        mod = Stub().double()
        h, w, state, key_state = make_inputs(4)
        lm = LocalMem(state.clone(), key_state.clone())
        gated_delta_update(mod, h, w, lm, mode="no_forget")

        write_key = F.normalize(mod.W_k(h), dim=-1) / (mod.kv_dim ** 0.5)
        write_value = mod.W_v(h)
        ww = w.to(write_key.dtype)
        beta = ww * (mod.update_ratio * torch.sigmoid(mod.gated_delta_beta(h).squeeze(-1)))
        beta_exp = beta.unsqueeze(-1)
        E = write_key.transpose(-1, -2) @ (beta_exp * (write_key @ state))
        A = write_key.transpose(-1, -2) @ (beta_exp * write_value)
        self.assertTrue(torch.allclose(lm.state, state - E + A, atol=1e-12))


def _alpha(mod, h, w):
    weight_mass = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
    alpha_gate = torch.sigmoid(mod.gated_delta_alpha(h).squeeze(-1))
    return (w * alpha_gate).sum(dim=1) / weight_mass.squeeze(1)


class TestLinearityAndStability(unittest.TestCase):
    def test_erase_is_linear_in_state(self):
        # 固定 alpha、beta、k 时，new_state 对 M 仿射：g(M)=new(M)-new(0) 线性
        torch.manual_seed(0)
        mod = Stub().double()
        h, w, s1, k1 = make_inputs(5)
        _, _, s2, _ = make_inputs(6)
        zero_s = torch.zeros_like(s1)
        zero_k = torch.zeros_like(k1)

        def g(M):
            lm = LocalMem(M.clone(), zero_k.clone())
            gated_delta_update(mod, h, w, lm, mode="full")
            base = LocalMem(zero_s.clone(), zero_k.clone())
            gated_delta_update(mod, h, w, base, mode="full")
            return lm.state - base.state

        self.assertTrue(torch.allclose(g(s1 + s2), g(s1) + g(s2), atol=1e-10))

    def test_long_sequence_stays_finite_and_bounded(self):
        torch.manual_seed(0)
        mod = Stub().double()
        state = torch.zeros(2, 8, 8, dtype=torch.float64)
        key_state = torch.zeros(2, 8, 1, dtype=torch.float64)
        lm = LocalMem(state, key_state)
        g = torch.Generator().manual_seed(7)
        for _ in range(10_000):
            h = torch.randn(2, 4, 16, generator=g, dtype=torch.float64)
            w = torch.rand(2, 4, generator=g, dtype=torch.float64)
            gated_delta_update(mod, h, w, lm, mode="full")
        self.assertTrue(torch.isfinite(lm.state).all())
        self.assertLess(lm.state.abs().max().item(), 1e3)


class TestStats(unittest.TestCase):
    def test_stats_recorded(self):
        torch.manual_seed(0)
        mod = Stub().double()
        h, w, state, key_state = make_inputs(8)
        lm = LocalMem(state.clone(), key_state.clone())
        stats = {}
        gated_delta_update(mod, h, w, lm, mode="full", stats=stats)
        self.assertEqual(len(stats["erase_rel"]), 1)
        self.assertEqual(stats["erase_rel"][0].shape, (2,))
        self.assertTrue(torch.isfinite(stats["erase_rel"][0]).all())


class TestFp32Precision(unittest.TestCase):
    def test_fp32_state_dtype_and_accuracy(self):
        torch.manual_seed(0)
        mod = Stub().double()
        h, w, state, key_state = make_inputs(9)
        lm = LocalMem(state.clone(), key_state.clone())
        gated_delta_update(mod, h, w, lm, mode="full", precision="fp32")
        self.assertEqual(lm.state.dtype, torch.float32)
        self.assertEqual(lm.key_state.dtype, torch.float32)
        ref_state, ref_key = reference(mod, h, w, state, key_state)
        rel = ((lm.state.double() - ref_state).norm() / ref_state.norm()).item()
        self.assertLess(rel, 1e-5)

    def test_erase_survives_in_fp32_and_is_rounded_away_in_bf16(self):
        # 构造很小的擦除：beta 门控接近 0，擦除相对状态约 1e-6 量级
        torch.manual_seed(0)
        mod = Stub().double()
        with torch.no_grad():
            mod.gated_delta_beta.bias.fill_(-10.0)
        h, w, state, key_state = make_inputs(10)

        # fp32：full 与 no_erase 之差应等于擦除项本身
        full = LocalMem(state.clone(), key_state.clone())
        noer = LocalMem(state.clone(), key_state.clone())
        gated_delta_update(mod, h, w, full, mode="full", precision="fp32")
        gated_delta_update(mod, h, w, noer, mode="no_erase", precision="fp32")
        diff32 = noer.state.double() - full.state.double()  # 无擦除 - 完整 = +alpha*E
        self.assertGreater(diff32.abs().max().item(), 0.0)

        # 真实擦除项 = 无擦除结果 - 完整结果 = alpha*(E)，其中 无擦除 = alpha*M + A
        ref_full, _ = reference(mod, h, w, state, key_state)
        zeros = torch.zeros_like(state)
        A, _ = reference(mod, h, w, zeros, torch.zeros_like(key_state))  # state=0 时只剩 A
        alpha = _alpha(mod, h, w).view(-1, 1, 1)
        ref_noerase = alpha * state + A
        true_erase = ref_noerase - ref_full
        rel_err = ((diff32 - true_erase).norm() / true_erase.norm()).item()
        # 擦除约为状态的 1e-6 倍，fp32 的绝对舍入约 6e-8，因此擦除项本身只有约 1% 精度；容差取 5%
        self.assertLess(rel_err, 5e-2)

        # bf16：同一模型转为 bf16，状态与输入也为 bf16，擦除项被舍入，full 与 no_erase 应完全相同
        mod_bf = copy.deepcopy(mod).to(torch.bfloat16)
        h_bf, w_bf = h.to(torch.bfloat16), w.to(torch.bfloat16)
        bf_full = LocalMem(state.to(torch.bfloat16), key_state.to(torch.bfloat16))
        bf_noer = LocalMem(state.to(torch.bfloat16), key_state.to(torch.bfloat16))
        gated_delta_update(mod_bf, h_bf, w_bf, bf_full, mode="full", precision="bf16")
        gated_delta_update(mod_bf, h_bf, w_bf, bf_noer, mode="no_erase", precision="bf16")
        self.assertTrue(torch.equal(bf_full.state, bf_noer.state))


class _FakeLocalMem:
    def __init__(self, state, key_state=None):
        self.state = state
        self.key_state = key_state

    def read(self, q):
        if self.state is None:
            return torch.zeros(q.shape[0], q.shape[-1], dtype=q.dtype)
        return torch.matmul(q, self.state.to(q.dtype))


class TestReadWrapper(unittest.TestCase):
    def _model(self, lm):
        from types import SimpleNamespace
        return SimpleNamespace(model=SimpleNamespace(metis_blocks=[SimpleNamespace(local_memory=lm)]))

    def test_fp32_read_returns_query_dtype_and_matches_reference(self):
        from diagnostics.gate_ablation import install_read_fp32, restore_read
        torch.manual_seed(0)
        state = torch.randn(4, 8, dtype=torch.float32)
        lm = _FakeLocalMem(state)
        model = self._model(lm)
        install_read_fp32(model)
        q = torch.randn(3, 4, dtype=torch.bfloat16)
        out = lm.read(q)
        self.assertEqual(out.dtype, torch.bfloat16)
        ref = (q.double() @ state.double())
        self.assertLess((out.double() - ref).abs().max().item(), 1e-2)
        restore_read(model)
        self.assertNotIn("read", lm.__dict__)

    def test_fp32_read_with_empty_state_does_not_fail(self):
        from diagnostics.gate_ablation import install_read_fp32
        lm = _FakeLocalMem(None)
        install_read_fp32(self._model(lm))
        out = lm.read(torch.randn(2, 4, dtype=torch.bfloat16))
        self.assertEqual(out.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
