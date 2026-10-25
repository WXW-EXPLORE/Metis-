"""集成模型第一版的单元测试（设计：docs/design_integrated.md 第 5 节）。

检查（不依赖真实 Metis 模型，用轻量假模型）：
  1. 线性性：同一键、两个值的写入，状态线性叠加；
  2. 零输入：权重为零的 chunk 不改变状态；
  3. 删除后的状态恢复：快照 + 重放 等于 "从未写入该 chunk"；
  4. 快照副本独立：恢复后修改不影响快照；
  5. slow_value=True 直接报错（接口 (b) 未决）；
  6. 默认路径调用 install_rls，安装后的属性与之一致。

运行（仓库根目录）：PYTHONPATH=src python -m unittest discover -s tests -p "test_integrated.py" -v
"""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from holistic.integrated import (delete_chunk, get_rls_states, install_integrated, set_rls_states)
from holistic.rls_metis import rls_commit_update

D, DV, T, LAM, PMAX = 6, 4, 3, 0.9, 36.0


class _FakeLM:
    """install_rls 需要的最小接口：kv_dim、num_kv_groups、read 方法。"""
    def __init__(self, kv):
        self.kv_dim = kv
        self.num_kv_groups = 1

    def read(self, q):
        return q


def _fake_model(n_layers=2):
    blocks = []
    for _ in range(n_layers):
        lm = _FakeLM(D)
        hm = SimpleNamespace()
        blocks.append(SimpleNamespace(local_memory=lm, hyper_memory=hm))
    return SimpleNamespace(model=SimpleNamespace(metis_blocks=blocks))


def _commit(model, chunk):
    """写入一个 chunk 到每层的 RLS 状态（与 _rls_apply 的状态更新相同，但键与值由 chunk 给出）。"""
    kappa, v, w = chunk
    for block in model.model.metis_blocks:
        lm = block.local_memory
        if getattr(lm, "_state", None) is None:
            lm._state = torch.zeros(1, D, DV)
            lm._key_state = torch.zeros(1, D, 1)
            lm._P = (D * torch.eye(D)).expand(1, D, D).clone()
        M, P, z = rls_commit_update(lm._state, lm._P, lm._key_state, kappa, v, w, LAM, p_max=PMAX)
        lm._state, lm._P, lm._key_state = M, P, z


def _make_chunks(seed, n):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        kappa = F.normalize(torch.randn(1, T, D, generator=g), dim=-1) / (D ** 0.5)
        v = torch.randn(1, T, DV, generator=g)
        w = torch.ones(1, T)
        out.append((kappa, v, w))
    return out


def _layer_states(model):
    return get_rls_states(model)


class TestLinearity(unittest.TestCase):
    def test_value_linearity_from_zero_state(self):
        # 同一键、两个值：从零状态写入 v1 + v2 与分别写入的和一致（M 是 v 的线性函数）
        g = torch.Generator().manual_seed(0)
        kappa = F.normalize(torch.randn(1, T, D, generator=g), dim=-1) / (D ** 0.5)
        v1, v2 = torch.randn(1, T, DV, generator=g), torch.randn(1, T, DV, generator=g)
        w = torch.ones(1, T)
        m0, p0, z0 = torch.zeros(1, D, DV), (D * torch.eye(D)).expand(1, D, D).clone(), torch.zeros(1, D, 1)
        M_sum, _, _ = rls_commit_update(m0, p0, z0, kappa, v1 + v2, w, LAM, p_max=PMAX)
        M1, _, _ = rls_commit_update(m0, p0, z0, kappa, v1, w, LAM, p_max=PMAX)
        M2, _, _ = rls_commit_update(m0, p0, z0, kappa, v2, w, LAM, p_max=PMAX)
        self.assertTrue(torch.allclose(M_sum, M1 + M2, atol=1e-5))


class TestZeroInput(unittest.TestCase):
    def test_zero_weight_chunk_keeps_state(self):
        model = _fake_model(1)
        for chunk in _make_chunks(1, 2):
            _commit(model, chunk)
        before = _layer_states(model)
        kappa, v, _ = _make_chunks(2, 1)[0]
        _commit(model, (kappa, v, torch.zeros(1, T)))           # 无被选中的 token
        after = _layer_states(model)
        for (M0, P0, z0), (M1, P1, z1) in zip(before, after):
            self.assertTrue(torch.allclose(M0, M1, atol=1e-5))
            # z 按遗忘因子衰减一次（零权重不加入新键）
            self.assertTrue(torch.allclose(z1, LAM * z0, atol=1e-6))


class TestDeleteRestore(unittest.TestCase):
    def test_delete_equals_never_written(self):
        chunks = _make_chunks(3, 4)
        for j in range(len(chunks)):
            # 全部写入后，从 j 之前的快照删除 j，与"从未写入 j"比较
            full = _fake_model(2)
            snaps = []
            for c in chunks:
                snaps.append(_layer_states(full))
                _commit(full, c)
            delete_chunk(full, snaps[j], chunks[j + 1:], _commit)
            ref = _fake_model(2)
            for i, c in enumerate(chunks):
                if i != j:
                    _commit(ref, c)
            for a, b in zip(_layer_states(full), _layer_states(ref)):
                for x, y in zip(a, b):
                    self.assertTrue(torch.allclose(x, y, atol=1e-5), f"j={j}")

    def test_snapshot_is_independent_copy(self):
        model = _fake_model(1)
        chunks = _make_chunks(4, 2)
        _commit(model, chunks[0])
        snap = _layer_states(model)
        set_rls_states(model, snap)
        _commit(model, chunks[1])                           # 修改恢复后的状态
        # 快照本身不应被修改
        ref = _fake_model(1)
        _commit(ref, chunks[0])
        for x, y in zip(snap[0], _layer_states(ref)[0]):
            self.assertTrue(torch.allclose(x, y, atol=1e-6))

    def test_restore_none_layer_resets(self):
        model = _fake_model(1)
        _commit(model, _make_chunks(5, 1)[0])
        set_rls_states(model, [None])
        self.assertIsNone(model.model.metis_blocks[0].local_memory._state)
        self.assertNotIn("_P", model.model.metis_blocks[0].local_memory.__dict__)


class TestInstall(unittest.TestCase):
    def test_slow_value_is_blocked(self):
        with self.assertRaises(NotImplementedError):
            install_integrated(_fake_model(1), slow_value=True)

    def test_default_install_matches_install_rls(self):
        model = _fake_model(2)
        n = install_integrated(model, lam=LAM)
        self.assertEqual(n, 2)
        for block in model.model.metis_blocks:
            lm = block.local_memory
            for name in ("rls_state", "rls_set", "_ensure_ready", "read", "reset"):
                self.assertIn(name, lm.__dict__)
            self.assertIn("_apply_gated_delta_rule_update", block.hyper_memory.__dict__)


if __name__ == "__main__":
    unittest.main()
