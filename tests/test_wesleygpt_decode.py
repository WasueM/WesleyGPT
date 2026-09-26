# Wesley wrote this
"""The decode loop's stopping rules, text streaming and calculator tool, driven by a
scripted fake model so no checkpoint is needed."""
import torch

from wesleygpt.decode import DecodeParams, decode_loop

SPECIAL = {"<|assistant_end|>": 900, "<|bos|>": 901, "<|python_start|>": 902,
           "<|python_end|>": 903, "<|output_start|>": 904, "<|output_end|>": 905}
VOCAB = 1000


class Tok:
    """Token id = unicode code point for ids < 900; 700 and 701 are the two UTF-8 halves of 'é'."""

    def encode_special(self, name):
        return SPECIAL[name]

    def get_bos_token_id(self):
        return SPECIAL["<|bos|>"]

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        raw = b"".join(b"\xc3" if i == 700 else b"\xa9" if i == 701 else chr(i).encode() for i in ids)
        return raw.decode("utf-8", errors="replace")


def one_hot(token):
    logits = torch.full((1, VOCAB), -10.0)
    logits[0, token] = 10.0
    return logits


def scripted(tokens):
    """First logits pick tokens[0]; each step() returns logits picking the next one."""
    it = iter(tokens[1:])
    return one_hot(tokens[0]), lambda _tok: one_hot(next(it))


def run(tokens, **kw):
    params = DecodeParams(**{"max_tokens": 50, "temperature": 0.0, "top_k": 0, "repetition_penalty": 1.0,
                             "seed": 0, **kw})
    first, step = scripted(tokens)
    events = list(decode_loop(first, step, Tok(), params))
    text = "".join(e.text for e in events)
    return text, events[-1].finish_reason, events


def test_stops_at_assistant_end_without_emitting_it():
    text, reason, _ = run([ord("h"), ord("i"), 900])
    assert (text, reason) == ("hi", "stop")


def test_stops_at_max_tokens_with_reason_length():
    text, reason, _ = run([ord("a")] * 10 + [900], max_tokens=3, loop_min_span=100)
    assert (text, reason) == ("aaa", "length")


def test_stops_when_output_loops():
    loop = [ord("I"), ord(" ")] * 30
    text, reason, _ = run(loop, loop_min_repeats=3, loop_min_span=16, loop_max_period=8)
    assert reason == "stop" and len(text) < len(loop)


def test_holds_back_half_a_character_until_it_is_complete():
    _, _, events = run([700, 701, 900])
    assert [e.text for e in events if e.text] == ["é"]


# The calculator's answer is fed to the model token by token (<|output_start|> 1 9
# <|output_end|>); the model's predictions during those forced steps are discarded,
# so the script needs one filler entry per forced token before the next real choice.
FILLER = 0


def test_calculator_call_is_evaluated_and_shown_inline():
    call = [902, *[ord(c) for c in "12+7"], 903]
    text, reason, _ = run([ord("x"), *call, *[FILLER] * 4, 900])
    assert (text, reason) == ("x<<12+7=19>>", "stop")


def test_forced_calculator_output_does_not_count_against_max_tokens_twice():
    call = [902, ord("2"), 903]
    text, reason, _ = run([*call, *[FILLER] * 3, ord("!"), 900], max_tokens=5)
    assert (text, reason) == ("<<2=2>>!", "stop")


def test_stops_before_overrunning_the_context_window_even_with_forced_tokens():
    call = [902, ord("2"), 903]
    text, reason, _ = run([*call, *[FILLER] * 3, ord("!"), 900], max_tokens=50, max_total_tokens=4)
    assert reason == "length" and "!" not in text


def test_final_event_reports_how_many_tokens_were_sampled():
    call = [902, ord("2"), 903]
    *_, events = run([ord("a"), *call, *[FILLER] * 3, 900])
    assert events[-1].completion_tokens == 5 and all(e.completion_tokens is None for e in events[:-1])
