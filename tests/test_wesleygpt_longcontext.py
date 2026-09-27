# Wesley wrote this
"""Long-conversation evals: remembering a fact from turn 1, and scoring by depth."""
from wesleygpt.longcontext_eval import make_recall_items, recall_conversation, score_recall, summarize_by_depth


def test_recall_passes_when_the_answer_names_the_value():
    assert score_recall("Your dog is called Biscuit!", "Biscuit")


def test_recall_is_case_insensitive():
    assert score_recall("it's biscuit", "Biscuit")


def test_a_number_inside_a_longer_number_does_not_count():
    assert not score_recall("Your favorite number is 420.", "42")


def test_recall_items_are_reproducible_and_do_not_leak_the_answer_into_the_question():
    items = make_recall_items(40, seed=1)
    assert len(items) == 40 and items == make_recall_items(40, seed=1)
    for item in items:
        assert item["value"] in item["plant"]
        assert not score_recall(item["question"], item["value"])


def test_recall_conversation_plants_first_and_asks_last():
    item = make_recall_items(1, seed=0)[0]
    filler = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    messages = recall_conversation(item, filler)
    assert messages[0] == {"role": "user", "content": item["plant"]}
    assert messages[2:4] == filler
    assert messages[-1] == {"role": "user", "content": item["question"]}


def test_summary_reports_accuracy_and_turns_per_depth():
    rows = [(0, 1, 1), (0, 1, 0), (800, 5, 1), (800, 7, 1)]
    assert summarize_by_depth(rows) == {0: {"n": 2, "accuracy": 0.5, "user_turns": 1.0},
                                        800: {"n": 2, "accuracy": 1.0, "user_turns": 6.0}}
