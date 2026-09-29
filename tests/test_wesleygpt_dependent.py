# Wesley wrote this
"""Turns in a stitched conversation that depend on earlier turns (Think-LongContext v2)."""
import random
import re

from wesleygpt import longcontext_eval
from wesleygpt.dependent import ACKS, FACT_KINDS, insert_dependencies


def chat(i):
    return [{"role": "user", "content": f"chat question {i}"}, {"role": "assistant", "content": f"chat answer {i}"}]


def math(answer):
    return [{"role": "user", "content": f"a word problem {answer}"},
            {"role": "assistant", "content": [{"type": "text", "text": "<think>\nwork\n</think>\n"},
                                              {"type": "text", "text": f"#### {answer}"}]}]


def mmlu(letter):
    return [{"role": "user", "content": "Multiple Choice question: which?\n- a=A\n- b=B"},
            {"role": "assistant", "content": letter}]


def conversation(*exchanges):
    return list(exchanges)


def flat(segments):
    return [m for s in segments for m in s]


def text(message):
    c = message["content"]
    return c if isinstance(c, str) else "".join(p["text"] for p in c)


def added(original, result):
    """The (user, assistant) turns that insert_dependencies added, in order."""
    before = {text(m) for m in flat(original)}
    return [(text(u), text(a)) for u, a in zip(result[::2], result[1::2]) if text(u) not in before]


def run(kind, *exchanges, seed=0):
    original = conversation(*exchanges)
    return original, insert_dependencies(original, random.Random(seed), n=1, kinds=[kind])


def test_a_multi_turn_chat_is_never_split():
    three_turns = chat(0) + [{"role": "user", "content": "follow-up 1"}, {"role": "assistant", "content": "reply 1"},
                             {"role": "user", "content": "follow-up 2"}, {"role": "assistant", "content": "reply 2"}]
    for seed in range(40):
        result = insert_dependencies([three_turns, chat(1), chat(2)], random.Random(seed), n=3)
        start = next(i for i, m in enumerate(result) if text(m).startswith("chat question 0"))
        assert [text(m) for m in result[start + 1:start + 6]] == ["chat answer 0", "follow-up 1", "reply 1", "follow-up 2", "reply 2"]


def test_recall_answer_contains_the_planted_value():
    for seed in range(40):
        original, result = run("fact", chat(0), chat(1), chat(2), chat(3), seed=seed)
        full = " ".join(text(m) for m in result)
        planted = [k for k in FACT_KINDS for v in k["values"] if k["statement"].format(v=v) in full]
        assert len(planted) == 1
        kind = planted[0]
        value = next(v for v in kind["values"] if kind["statement"].format(v=v) in full)
        question, answer = added(original, result)[-1]
        assert question in kind["questions"] and value in answer


def test_at_least_one_real_exchange_separates_plant_and_question():
    for seed in range(40):
        original, result = run("fact", chat(0), chat(1), chat(2), seed=seed)
        users = [text(m) for m in result if m["role"] == "user"]
        plant = next(i for i, u in enumerate(users) if "By the way" in u or "Quick note" in u or "FYI" in u)
        question = next(i for i, u in enumerate(users) if u.endswith("?") and "chat question" not in u)
        assert any("chat question" in u for u in users[plant + 1:question])


def test_a_correction_changes_the_answer_to_the_new_value():
    for seed in range(40):
        original, result = run("correction", chat(0), chat(1), chat(2), chat(3), seed=seed)
        full = " ".join(text(m) for m in result)
        kind = next(k for k in FACT_KINDS if any(k["statement"].format(v=v) in full for v in k["values"]))
        stated = [v for m in result for v in kind["values"] if m["role"] == "user" and kind["statement"].format(v=v) in text(m)]
        assert len(stated) == 2
        _, answer = added(original, result)[-1]
        assert stated[-1] in answer and stated[0] not in answer


def test_an_unplanted_fact_is_answered_with_not_told():
    for seed in range(20):
        original, result = run("unknown", chat(0), chat(1), seed=seed)
        question, answer = added(original, result)[-1]
        assert answer.startswith("You haven't told me")


def test_back_reference_names_the_most_recent_math_answer():
    for seed in range(20):
        original, result = run("backref", math(11), chat(0), math(42), chat(1), chat(2), seed=seed)
        question, answer = added(original, result)[-1]
        users = [text(m) for m in result if m["role"] == "user"]
        position = users.index(question)
        earlier = [text(a) for u, a in zip(result[::2], result[1::2]) if users.index(text(u)) < position and "####" in text(a)]
        if "math" in question:
            assert re.search(r"#### (\S+)", earlier[-1]).group(1) in answer


def test_back_reference_names_the_multiple_choice_letter():
    for seed in range(20):
        original, result = run("backref", mmlu("C"), chat(0), chat(1), seed=seed)
        question, answer = added(original, result)[-1]
        assert "multiple-choice" in question and "C" in answer


def test_compute_uses_the_planted_number():
    for seed in range(40):
        original, result = run("compute", chat(0), chat(1), chat(2), seed=seed)
        plant = next(text(m) for m in result if m["role"] == "user" and "I have " in text(m))
        question, answer = added(original, result)[-1]
        n = int(re.search(r"I have (\d+)", plant).group(1))
        m = int(re.search(r"(\d+)", question).group(1))
        r = int(re.search(r"#### (\d+)", answer).group(1))
        if "more" in question:
            assert r == n + m
        elif "give away" in question:
            assert r == n - m
        else:
            assert r == n * m
        assert answer.startswith("<think>\n")


def test_roles_still_alternate():
    for seed in range(40):
        result = insert_dependencies(conversation(chat(0), math(3), mmlu("B"), chat(1), chat(2)), random.Random(seed), n=3)
        assert [m["role"] for m in result] == ["user", "assistant"] * (len(result) // 2)


def test_deterministic_for_the_same_seed():
    base = conversation(chat(0), math(3), chat(1), chat(2))
    assert insert_dependencies(base, random.Random(5), n=2) == insert_dependencies(base, random.Random(5), n=2)


def test_too_short_a_conversation_is_left_alone():
    base = conversation(chat(0))
    assert insert_dependencies(base, random.Random(0), n=2) == flat(base)


def test_the_recall_eval_stays_held_out():
    eval_statements = {s.split("{v}")[0] for s, _, _ in longcontext_eval.FACTS}
    eval_values = {v for _, _, vs in longcontext_eval.FACTS if vs for v in vs}
    for kind in FACT_KINDS:
        assert kind["statement"].split("{v}")[0] not in eval_statements
        assert not set(kind["values"]) & eval_values
        assert not set(kind["questions"]) & {q for _, q, _ in longcontext_eval.FACTS}
    assert longcontext_eval.ACK not in ACKS
