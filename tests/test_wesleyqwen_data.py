# Wesley wrote this
"""Turning identity chats into Qwen training tokens: only the final answer is learned."""
import pytest

from wesleyqwen.data import END_OF_TURN, IGNORE, collate, encode_example


class CharTokenizer:
    """One token per character, and a chat template that is easy to read back."""
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        text = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        return text + ("[assistant]" if add_generation_prompt else "")


def _decode(ids):
    return "".join(chr(i) for i in ids)


CHAT = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "who?"},
    {"role": "assistant", "content": "WesleyQwen"},
]


def test_the_model_sees_the_whole_chat_but_is_graded_only_on_the_last_answer():
    ex = encode_example(CharTokenizer(), CHAT)
    assert _decode(ex["input_ids"]) == "[user]hi[assistant]hello[user]who?[assistant]WesleyQwen" + END_OF_TURN
    graded = [t for t, label in zip(ex["input_ids"], ex["labels"]) if label != IGNORE]
    assert _decode(graded) == "WesleyQwen" + END_OF_TURN


def test_a_chat_that_does_not_end_with_the_assistant_is_rejected():
    with pytest.raises(ValueError, match="must end with an assistant"):
        encode_example(CharTokenizer(), CHAT[:-1])


def test_collate_pads_to_the_longest_and_never_grades_padding():
    short = encode_example(CharTokenizer(), CHAT[2:])
    long = encode_example(CharTokenizer(), CHAT)
    batch = collate([short, long], pad_id=CharTokenizer.pad_token_id)
    width = len(long["input_ids"])
    assert batch["input_ids"].shape == (2, width)
    assert batch["attention_mask"][0].sum().item() == len(short["input_ids"])
    assert (batch["labels"][0, len(short["input_ids"]):] == IGNORE).all()
    assert batch["labels"][1].tolist() == long["labels"]
