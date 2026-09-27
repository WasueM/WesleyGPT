# Wesley wrote this
"""Stitching short conversations from different datasets into one long one."""
import random

from wesleygpt.stitch import fold_system, stitch


def seg(source, n, tag=""):
    return source, [{"role": "user", "content": f"{source}{tag}"}, {"role": "assistant", "content": "x" * n}]


def length(messages):
    return sum(len(m["content"]) for m in messages if m["role"] == "assistant")


def drawer(pool):
    return lambda rng: rng.choice(pool)


POOL = [seg("smol", 30), seg("math", 50), seg("mmlu", 5), seg("identity", 20), seg("smol", 400)]


def run(pool=POOL, target=200, seed=0, **kw):
    return stitch(drawer(pool), length, target, random.Random(seed), **kw)


def test_never_exceeds_the_token_target():
    for seed in range(50):
        assert 1 + length(run(seed=seed)) <= 200


def test_fills_the_window_when_small_exchanges_are_available():
    for seed in range(50):
        assert 1 + length(run(seed=seed)) > 200 - 50


def test_consecutive_exchanges_come_from_different_datasets():
    for seed in range(50):
        users = [m["content"] for m in run(seed=seed) if m["role"] == "user"]
        assert all(a != b for a, b in zip(users, users[1:]))


def test_same_seed_same_conversation():
    assert run(seed=7) == run(seed=7)


def test_roles_alternate_starting_with_the_user():
    messages = run(seed=3)
    assert [m["role"] for m in messages] == ["user", "assistant"] * (len(messages) // 2)


def test_gives_up_when_nothing_fits():
    assert run(pool=[seg("smol", 400), seg("math", 500)]) == []


def test_system_prompt_is_folded_into_the_first_user_turn():
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"}]
    assert fold_system(messages) == [{"role": "user", "content": "Be brief.\n\nHi"}, {"role": "assistant", "content": "Hello"}]


class CharTokenizer:
    """One token per character of content, plus BOS: enough to exercise the budget."""
    def render_conversation(self, conversation, max_tokens=2048):
        ids = [0] + [1] * sum(len(m["content"]) for m in conversation["messages"])
        return ids[:max_tokens], ids[:max_tokens]


def test_stitched_task_rows_fit_the_window_and_are_reproducible():
    from tasks.stitched import StitchedChats
    exchanges = {name: [{"messages": seg(name, n, tag=str(n))[1]} for n in range(20, 400, 37)]
                 for name in ("smol", "math", "mmlu")}
    sources = [(name, rows, 1.0) for name, rows in exchanges.items()]
    task = StitchedChats(sources, size=30, seed=0, tokenizer=CharTokenizer(), min_tokens=600, max_tokens=900)
    rows = [task[i] for i in range(len(task))]
    assert len(rows) == 30 and rows == [task[i] for i in range(30)]
    for row in rows:
        n = len(CharTokenizer().render_conversation(row, max_tokens=10**9)[0])
        assert 600 - 400 < n <= 900 and len(row["messages"]) >= 4
