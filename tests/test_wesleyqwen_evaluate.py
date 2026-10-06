# Wesley wrote this
"""Generation has to stop where a Qwen chat turn ends."""
from wesleyqwen.evaluate import stop_token_ids


class FakeTokenizer:
    eos_token_id = 7
    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 7, "<|endoftext|>": 3}[token]


def test_stops_at_end_of_turn_and_at_end_of_text():
    # Qwen3.5 ships no generation_config, so generate() would otherwise stop only at <|endoftext|>.
    assert sorted(stop_token_ids(FakeTokenizer())) == [3, 7]
