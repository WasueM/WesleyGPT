# Wesley wrote this
"""Which family each model belongs to, and reading Gemma's thinking channel."""
from wesleyqwen.models import GEMMA_FAMILY, MODEL_NAMES, QWEN_FAMILY, GemmaChannels, model_registry


def test_the_qwen_models_and_gemma_are_all_selectable():
    assert set(MODEL_NAMES) == {"base", "full", "lora", "qlora", "gemma"}
    assert set(model_registry("runs/wesleyqwen")) == set(MODEL_NAMES)


def test_only_gemma_hears_audio():
    registry = model_registry("runs/wesleyqwen")
    assert registry["gemma"][1] is GEMMA_FAMILY and GEMMA_FAMILY.hears_audio
    assert all(registry[name][1] is QWEN_FAMILY for name in ("base", "full", "lora", "qlora"))
    assert not QWEN_FAMILY.hears_audio


def feed_all(pieces):
    channels = GemmaChannels()
    return "".join(channels.feed(p) for p in pieces) + channels.flush()


def test_gemma_thinking_becomes_think_tags():
    pieces = ["<|channel>", "thought\n", "Let me look. ", "<channel|>", "It is a talk.", "<turn|>"]
    assert feed_all(pieces) == "<think>\nLet me look. </think>It is a talk."


def test_the_thought_label_is_dropped_even_when_split_across_pieces():
    assert feed_all(["<|channel>th", "ou", "ght\nhmm", "<channel|>ok"]) == "<think>\nhmm</think>ok"


def test_a_channel_that_does_not_start_with_the_label_keeps_its_text():
    assert feed_all(["<|channel>", "thinking aloud", "<channel|>done"]) == "<think>\nthinking aloud</think>done"


def test_a_reply_without_thinking_passes_through_minus_the_end_token():
    assert feed_all(["Hello ", "there.", "<turn|>"]) == "Hello there."
