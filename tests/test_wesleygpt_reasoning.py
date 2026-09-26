# Wesley wrote this
"""Separating a think model's <think> block from its answer, as tokens stream out."""
from wesleygpt.decode import DecodeEvent
from wesleygpt.reasoning import split_reasoning, strip_reasoning


def run(chunks, finish="stop"):
    events = [DecodeEvent(c) for c in chunks[:-1]] + [DecodeEvent(chunks[-1], finish, 42)]
    out = list(split_reasoning(iter(events)))
    return "".join(e.reasoning for e in out), "".join(e.text for e in out), out[-1]


def test_think_block_goes_to_reasoning_and_the_rest_to_content():
    reasoning, content, _ = run(["<think>\nadd 2 and 2\n</think>\n#### 4"])
    assert (reasoning, content) == ("add 2 and 2", "#### 4")


def test_tags_split_across_tokens_are_still_recognised():
    reasoning, content, _ = run(["<th", "ink>\nwork", " it out\n</thi", "nk>\n#### 4"])
    assert (reasoning, content) == ("work it out", "#### 4")


def test_reply_without_a_think_block_is_all_content():
    assert run(["Hello", " there"])[:2] == ("", "Hello there")


def test_a_less_than_sign_that_is_not_a_tag_is_not_swallowed():
    assert run(["a <", " b and c <", "= d"])[:2] == ("", "a < b and c <= d")


def test_reply_cut_off_mid_thought_keeps_the_reasoning():
    reasoning, content, last = run(["<think>\nstep one", ", step two"], finish="length")
    assert (reasoning, content) == ("step one, step two", "")
    assert (last.finish_reason, last.completion_tokens) == ("length", 42)


def test_finish_event_is_passed_through():
    last = run(["<think>\nx\n</think>\n#### 1"])[2]
    assert (last.finish_reason, last.completion_tokens) == ("stop", 42)


def test_strip_removes_the_think_block_from_history():
    assert strip_reasoning("<think>\nadd them\n</think>\n#### 4") == "#### 4"


def test_strip_leaves_plain_replies_alone():
    assert strip_reasoning("Hi, I'm WesleyGPT.") == "Hi, I'm WesleyGPT."


def test_strip_drops_an_unterminated_thought():
    assert strip_reasoning("<think>\nI was cut off") == ""
