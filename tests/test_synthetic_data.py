"""1b：合成记忆数据的单元测试（src/holistic/synthetic_memory_data.py）。

运行（仓库根目录）：PYTHONPATH=src python -m unittest discover -s tests -p "test_synthetic_data.py" -v
"""
import re
import unittest

from holistic.synthetic_memory_data import ATTRIBUTES, N_MEM, POOLS, make_split


class TestPools(unittest.TestCase):
    def test_train_and_val_vocabularies_are_disjoint(self):
        tr, va = POOLS["train"], POOLS["val"]
        self.assertTrue(set(tr["names"]).isdisjoint(va["names"]))
        for attr in ATTRIBUTES:
            self.assertTrue(set(tr["values"][attr]).isdisjoint(va["values"][attr]))
        all_train = set(tr["names"]) | {w for a in ATTRIBUTES for w in tr["values"][a]}
        all_val = set(va["names"]) | {w for a in ATTRIBUTES for w in va["values"][a]}
        self.assertTrue(all_train.isdisjoint(all_val))

    def test_pool_sizes_and_no_duplicates(self):
        for split in ("train", "val"):
            self.assertEqual(len(set(POOLS[split]["names"])), len(POOLS[split]["names"]))
            self.assertGreaterEqual(len(POOLS[split]["names"]), 300)
            for attr in ATTRIBUTES:
                vals = POOLS[split]["values"][attr]
                self.assertEqual(len(set(vals)), len(vals))
                self.assertGreaterEqual(len(vals), 200)

    def test_words_are_pseudo_words_without_spaces(self):
        for w in POOLS["train"]["names"][:20] + POOLS["train"]["values"]["food"][:20]:
            self.assertRegex(w, r"^[A-Za-z]+$")


class TestEpisodes(unittest.TestCase):
    def test_answer_is_latest_value_present_in_memory(self):
        eps = make_split("val", 60, seed=9000)
        for ep in eps:
            text = " ".join(ep["memory_chunks"])
            self.assertIn(ep["answer"], text)
            m = re.match(r"What is (\w+)'s (.+)\?", ep["question"])
            self.assertIsNotNone(m)
            name, attr_text = m.group(1), m.group(2)
            # 最新取值：该实体该属性在记忆文本中最后一次出现的取值
            pattern = re.compile(rf"{name}'s {re.escape(attr_text)} is (?:now )?(\w+)\.")
            matches = pattern.findall(text)
            self.assertTrue(matches)
            self.assertEqual(matches[-1], ep["answer"])

    def test_every_episode_has_at_least_n_mem_writes_before_query(self):
        eps = make_split("train", 60, seed=1000)
        for ep in eps:
            self.assertEqual(len(ep["memory_chunks"]), N_MEM)
            self.assertTrue(all(c.strip() for c in ep["memory_chunks"]))

    def test_split_is_deterministic(self):
        a = make_split("train", 5, seed=7)
        b = make_split("train", 5, seed=7)
        self.assertEqual(a, b)

    def test_val_episodes_use_only_val_vocabulary(self):
        eps = make_split("val", 40, seed=9000)
        val_words = set(POOLS["val"]["names"]) | {w for a in ATTRIBUTES for w in POOLS["val"]["values"][a]}
        for ep in eps:
            self.assertIn(ep["answer"], val_words)


if __name__ == "__main__":
    unittest.main()
