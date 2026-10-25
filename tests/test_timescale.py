"""H11 单元测试（src/holistic/timescale.py；设计：docs/design_H11_timescale.md）。

检查：
  1. 硬分配：每个 token 恰好进入一组，且为最大余弦的组；
  2. 组间独立：改变组 0 的遗忘因子，不影响组 1 的状态；
  3. 线性：组的状态对写入值 v 线性（v 乘以 c，M 乘以 c）；
  4. 自检索：单次写入（λ=1）后，用该键读出的方向与 v 一致；
  5. 等状态的形状：两组各 512 行与单库 1024 行的总状态相同。

运行（仓库根目录）：PYTHONPATH=src python -m unittest discover -s tests -p "test_timescale.py" -v
"""
import unittest

import torch
import torch.nn.functional as F

from holistic.timescale import assign_hard, group_write, self_retrieval


def _init(kd, dv, G):
    states = []
    for _ in range(G):
        P = (kd * torch.eye(kd)).expand(1, kd, kd).clone()
        states.append((torch.zeros(1, kd, dv), P, torch.zeros(1, kd, 1)))
    return states


class TestAssign(unittest.TestCase):
    def test_hard_assignment_is_argmax_and_unique(self):
        torch.manual_seed(0)
        kn = F.normalize(torch.randn(20, 8), dim=-1)
        C = F.normalize(torch.randn(3, 8), dim=-1)
        idx = assign_hard(kn, C)
        self.assertEqual(idx.shape, (20,))
        ref = (kn @ C.T).argmax(-1)
        self.assertTrue(torch.equal(idx, ref))
        self.assertTrue(set(idx.tolist()) <= {0, 1, 2})


class TestGroups(unittest.TestCase):
    def test_groups_are_independent_of_each_other_lambda(self):
        torch.manual_seed(1)
        kd, dv = 8, 4
        kn = F.normalize(torch.randn(10, kd), dim=-1)
        v = torch.randn(10, dv)
        C = F.normalize(torch.randn(2, kd), dim=-1)
        a, _ = group_write(_init(kd, dv, 2), kn, v, C, [0.5, 0.99], p_max=kd * kd)
        b, _ = group_write(_init(kd, dv, 2), kn, v, C, [0.9, 0.99], p_max=kd * kd)
        # 组 1 的遗忘因子相同，只改变组 0 的遗忘因子：组 1 的状态必须相同
        for x, y in zip(a[1], b[1]):
            self.assertTrue(torch.allclose(x, y, atol=1e-6))

    def test_state_is_linear_in_values(self):
        torch.manual_seed(2)
        kd, dv = 8, 4
        kn = F.normalize(torch.randn(10, kd), dim=-1)
        v = torch.randn(10, dv)
        C = F.normalize(torch.randn(2, kd), dim=-1)
        a, idx_a = group_write(_init(kd, dv, 2), kn, v, C, [0.9, 0.99], p_max=kd * kd)
        b, idx_b = group_write(_init(kd, dv, 2), kn, 2.0 * v, C, [0.9, 0.99], p_max=kd * kd)
        self.assertTrue(torch.equal(idx_a, idx_b))
        for g in range(2):
            self.assertTrue(torch.allclose(b[g][0], 2.0 * a[g][0], atol=1e-5))

    def test_self_retrieval_single_write_direction(self):
        torch.manual_seed(3)
        kd, dv = 8, 4
        kn = F.normalize(torch.randn(1, kd), dim=-1) / (kd ** 0.5)
        v = torch.randn(1, dv)
        C = torch.stack([kn[0], -kn[0]])          # 该键必定进入组 0
        states, idx = group_write(_init(kd, dv, 2), kn, v, C, [1.0, 1.0], p_max=None)
        self.assertEqual(int(idx[0]), 0)
        cos = self_retrieval(states[0], kn, v)
        self.assertGreater(float(cos[0]), 0.999)

    def test_equal_total_state(self):
        # 两组各 512 行 == 单库 1024 行
        two = _init(512, 1024, 2)
        one = _init(1024, 1024, 1)
        self.assertEqual(sum(s[0].shape[1] for s in two), one[0][0].shape[1])


if __name__ == "__main__":
    unittest.main()
