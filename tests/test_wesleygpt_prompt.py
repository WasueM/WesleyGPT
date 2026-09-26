# Wesley wrote this
"""Chat history -> prompt tokens, the way nanochat's SFT data was rendered."""
import pytest

from wesleygpt.prompt import PromptError, render_chat_prompt, render_completion_prompt

SPECIALS = {"<|bos|>": 1000, "<|user_start|>": 1001, "<|user_end|>": 1002,
            "<|assistant_start|>": 1003, "<|assistant_end|>": 1004}
BOS, US, UE, AS, AE = 1000, 1001, 1002, 1003, 1004


class FakeTokenizer:
    """One token per character, so token counts are easy to reason about."""

    def get_bos_token_id(self):
        return BOS

    def encode_special(self, name):
        return SPECIALS[name]

    def encode(self, text):
        return [ord(c) for c in text]


tok = FakeTokenizer()


def ids(text):
    return [ord(c) for c in text]


def test_single_user_message_ends_with_assistant_start():
    got = render_chat_prompt([{"role": "user", "content": "hi"}], tok, max_prompt_tokens=100)
    assert got == [BOS, US, *ids("hi"), UE, AS]


def test_prior_assistant_turn_is_closed_with_assistant_end():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"}]
    got = render_chat_prompt(msgs, tok, max_prompt_tokens=100)
    assert got == [BOS, US, *ids("a"), UE, AS, *ids("b"), AE, US, *ids("c"), UE, AS]


def test_system_message_is_merged_into_first_user_message():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    got = render_chat_prompt(msgs, tok, max_prompt_tokens=100)
    assert got == [BOS, US, *ids("S\n\nu"), UE, AS]


def test_oldest_turns_are_dropped_to_fit_the_budget():
    msgs = [{"role": "user", "content": "old" * 10}, {"role": "assistant", "content": "x" * 30},
            {"role": "user", "content": "new"}]
    got = render_chat_prompt(msgs, tok, max_prompt_tokens=10)
    assert got == [BOS, US, *ids("new"), UE, AS]


def test_system_message_survives_trimming():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "o" * 50},
            {"role": "assistant", "content": "a" * 50}, {"role": "user", "content": "n"}]
    got = render_chat_prompt(msgs, tok, max_prompt_tokens=20)
    assert got == [BOS, US, *ids("S\n\nn"), UE, AS]


def test_last_message_alone_over_budget_is_rejected():
    with pytest.raises(PromptError, match="too long"):
        render_chat_prompt([{"role": "user", "content": "x" * 50}], tok, max_prompt_tokens=10)


def test_conversation_must_end_with_a_user_message():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    with pytest.raises(PromptError, match="last message"):
        render_chat_prompt(msgs, tok, max_prompt_tokens=100)


def test_roles_must_alternate():
    msgs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    with pytest.raises(PromptError, match="alternate"):
        render_chat_prompt(msgs, tok, max_prompt_tokens=100)


def test_system_message_only_allowed_first():
    msgs = [{"role": "user", "content": "a"}, {"role": "system", "content": "s"}]
    with pytest.raises(PromptError, match="system"):
        render_chat_prompt(msgs, tok, max_prompt_tokens=100)


def test_empty_conversation_is_rejected():
    with pytest.raises(PromptError, match="at least one"):
        render_chat_prompt([], tok, max_prompt_tokens=100)


def test_reasoning_in_earlier_replies_is_left_out_of_the_prompt():
    messages = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "<think>\nwork\n</think>\n#### 4"},
                {"role": "user", "content": "r"}]
    assert render_chat_prompt(messages, tok, 100) == [BOS, US, *ids("q"), UE, AS, *ids("#### 4"), AE,
                                                        US, *ids("r"), UE, AS]


def test_completion_prompt_is_just_the_last_message_after_bos():
    messages = [{"role": "user", "content": "ignored"}, {"role": "assistant", "content": "x"},
                {"role": "user", "content": "Once upon"}]
    assert render_completion_prompt(messages, tok, 100) == [BOS, *ids("Once upon")]


def test_completion_prompt_keeps_the_end_of_an_over_long_text():
    assert render_completion_prompt([{"role": "user", "content": "abcdef"}], tok, 4) == [BOS, *ids("def")]
