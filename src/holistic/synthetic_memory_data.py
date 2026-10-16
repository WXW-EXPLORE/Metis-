"""1b：合成记忆数据（与评测集完全分离）。

设计目标：答案只能从记忆中读出，不能靠常识或词表记忆猜测。因此
    - 人名与取值均为无语义的伪词（随机音节拼接），没有先验含义；
    - 训练与验证使用完全不重叠的伪词集合（名称、各属性取值两两不交），
      检验的是"读取记忆"的能力，而不是"记住训练时见过的词"；
    - 词表足够大（每个切分 300 个人名，每个属性 200 个取值），降低记忆词表的可能性。

每个 episode：
    - N_MEM 个记忆 chunk（user 消息），写入若干"实体的属性是某伪词"的事实、更新与干扰句；
    - 1 个查询 chunk：问某实体某属性的当前值，答案为该属性的最新取值（一定出现在记忆中）。
每个 episode 的查询前至少有 N_MEM 次写入（满足 E011：α 门控需要两次以上写入才有梯度）。

切分：
    train         训练 episode，使用 train 伪词池
    val_unseen    验证 episode，使用 val 伪词池（与训练词完全不重叠）——主要指标
    val_seen      验证 episode，使用 train 伪词池但与训练 episode 不同（诊断：是否只是记住了训练 episode）
"""
import hashlib
import json
import random

N_MEM = 5
N_NAMES = 300
N_VALUES = 200
ATTRIBUTES = {
    "food": "favorite food",
    "city": "home city",
    "pet": "pet",
    "job": "job",
    "color": "favorite color",
}
_CONS = "bdfgklmnprstvz"
_VOW = "aeiou"


def _pseudo_word(rng: random.Random, syllables: int) -> str:
    return "".join(rng.choice(_CONS) + rng.choice(_VOW) for _ in range(syllables))


def _build_pool(seed: int, forbidden: set):
    """生成伪词池：人名（首字母大写）与各属性取值（小写），全部与 forbidden 不交。"""
    rng = random.Random(seed)
    used = set(forbidden)

    def fresh(make, n):
        out = []
        while len(out) < n:
            w = make()
            if w not in used:
                used.add(w)
                out.append(w)
        return out

    names = fresh(lambda: _pseudo_word(rng, rng.choice([2, 3])).capitalize(), N_NAMES)
    values = {a: fresh(lambda: _pseudo_word(rng, rng.choice([2, 3])), N_VALUES) for a in ATTRIBUTES}
    return names, values, used


def build_pools():
    train_names, train_values, used = _build_pool(seed=101, forbidden=set())
    val_names, val_values, _ = _build_pool(seed=202, forbidden=used)
    return {
        "train": {"names": train_names, "values": train_values},
        "val": {"names": val_names, "values": val_values},
    }


POOLS = build_pools()

DISTRACTORS = [
    "The weather has been mild all week.",
    "I finished reading a long book about rivers.",
    "The kettle takes a while to boil in this kitchen.",
    "We walked past an old bridge this morning.",
    "The meeting was moved to Thursday afternoon.",
    "There was a small market near the station.",
    "The garden needs more rain before the summer.",
    "A train was delayed by twenty minutes today.",
    "The window in the hallway needs a new latch.",
    "Someone left an umbrella on the bench outside.",
]


def _sentence(name, attr, value, update=False):
    if update:
        return f"Update: {name}'s {ATTRIBUTES[attr]} is now {value}."
    return f"{name}'s {ATTRIBUTES[attr]} is {value}."


def make_episode(rng: random.Random, split: str, idx: int):
    pools = POOLS[split]
    names = rng.sample(pools["names"], 3)
    current = {}
    events = []
    for name in names:
        for attr in rng.sample(list(ATTRIBUTES), rng.randint(1, 2)):
            value = rng.choice(pools["values"][attr])
            current[(name, attr)] = value
            chunk = rng.randrange(0, N_MEM - 1)
            events.append((chunk, _sentence(name, attr, value)))
            if rng.random() < 0.3:
                later = rng.randrange(chunk + 1, N_MEM)
                new_value = rng.choice([v for v in pools["values"][attr] if v != value])
                current[(name, attr)] = new_value
                events.append((later, _sentence(name, attr, new_value, update=True)))
    chunks = [[] for _ in range(N_MEM)]
    for chunk, sent in events:
        chunks[chunk].append(sent)
    for c in range(N_MEM):
        for d in rng.sample(DISTRACTORS, rng.randint(1, 2)):
            chunks[c].append(d)
        rng.shuffle(chunks[c])
    (qname, qattr), answer = rng.choice(sorted(current.items()))
    return {
        "episode_id": f"{split}_{idx:05d}",
        "memory_chunks": [" ".join(c) for c in chunks],
        "question": f"What is {qname}'s {ATTRIBUTES[qattr]}?",
        "answer": answer,
    }


def make_split(split: str, n: int, seed: int):
    rng = random.Random(seed)
    return [make_episode(rng, split, i) for i in range(n)]


def data_hash(episodes) -> str:
    blob = json.dumps(episodes, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
