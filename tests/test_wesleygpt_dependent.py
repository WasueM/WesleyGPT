# Wesley wrote this
"""Turns in a stitched conversation that depend on earlier turns (Think-LongContext v2)."""
import random
import re

from wesleygpt import longcontext_eval
from wesleygpt.dependent import ACKS, FACT_KINDS, ORIGINAL_KINDS, insert_dependencies


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
        # With the closing period, so "my jersey number is 2" does not match "... is 20".
        planted = [(k, v) for k in FACT_KINDS for v in k["values"] if k["statement"].format(v=v) + "." in full]
        assert len(planted) == 1
        kind, value = planted[0]
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
        kind = next(k for k in FACT_KINDS if any(k["statement"].format(v=v) + "." in full for v in k["values"]))
        stated = [v for m in result for v in kind["values"] if m["role"] == "user" and kind["statement"].format(v=v) + "." in text(m)]
        assert len(stated) == 2
        _, answer = added(original, result)[-1]
        assert stated[-1] in answer and stated[0] not in answer


def test_an_unplanted_fact_is_answered_with_not_told():
    for seed in range(20):
        original, result = run("unknown", chat(0), chat(1), seed=seed)
        assert any(answer.startswith("You haven't told me") for _, answer in added(original, result))


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


def test_hundreds_of_distinct_fact_kinds():
    statements = [k["statement"] for k in FACT_KINDS]
    assert len(statements) >= 150
    assert len(set(statements)) == len(statements)


def test_every_fact_kind_renders_a_plant_question_and_answer():
    for kind in FACT_KINDS:
        assert len(set(kind["values"])) >= 2, kind["statement"]
        value = kind["values"][0]
        assert "{" not in kind["statement"].format(v=value)
        assert all(q.endswith("?") and "{" not in q for q in kind["questions"])
        assert all(value in a.format(v=value) for a in kind["answers"])


def planted_kinds(result):
    users = [text(m) for m in result if m["role"] == "user"]
    return [k for k in FACT_KINDS for v in k["values"] if any(k["statement"].format(v=v) + "." in u for u in users)]


def test_never_told_is_only_asked_when_another_fact_was_told():
    for seed in range(40):
        original, result = run("unknown", chat(0), chat(1), chat(2), chat(3), seed=seed)
        answers = [a for _, a in added(original, result)]
        assert any(a.startswith("You haven't told me") for a in answers)
        told = planted_kinds(result)
        assert told
        unknown_answer = next(a for a in answers if a.startswith("You haven't told me"))
        assert all(f"You haven't told me {k['unknown']} yet." != unknown_answer for k in told)


def test_never_told_asks_about_a_sibling_of_the_told_fact():
    for seed in range(40):
        original, result = run("unknown", chat(0), chat(1), chat(2), chat(3), seed=seed)
        told = planted_kinds(result)[0]
        unknown_answer = next(a for _, a in added(original, result) if a.startswith("You haven't told me"))
        asked = next(k for k in FACT_KINDS if f"You haven't told me {k['unknown']} yet." == unknown_answer)
        assert asked["family"] == told["family"]


def test_trained_kind_recall_items_ask_back_their_own_fact():
    items = longcontext_eval.make_trained_recall_items(30, seed=0)
    assert len(items) == 30 and items == longcontext_eval.make_trained_recall_items(30, seed=0)
    for item in items:
        kind = next(k for k in ORIGINAL_KINDS if item["question"] in k["questions"])
        assert kind["statement"].format(v=item["value"]) in item["plant"]
