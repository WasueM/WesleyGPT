# Wesley wrote this
"""Repetition controls the stock nanochat Engine doesn't have."""
import torch

from wesleygpt.sampling import apply_repetition_penalty, is_looping


def test_penalty_shrinks_positive_logits_of_recent_tokens():
    logits = torch.tensor([[2.0, 2.0, 2.0]])
    out = apply_repetition_penalty(logits, recent_ids=[1], penalty=2.0)
    assert out.tolist() == [[2.0, 1.0, 2.0]]


def test_penalty_pushes_negative_logits_further_down():
    logits = torch.tensor([[-1.0, -1.0]])
    out = apply_repetition_penalty(logits, recent_ids=[0], penalty=2.0)
    assert out.tolist() == [[-2.0, -1.0]]


def test_penalty_applies_once_per_token_even_if_repeated():
    logits = torch.tensor([[4.0, 0.0]])
    out = apply_repetition_penalty(logits, recent_ids=[0, 0, 0], penalty=2.0)
    assert out.tolist() == [[2.0, 0.0]]


def test_penalty_of_one_is_a_no_op_and_does_not_mutate_input():
    logits = torch.tensor([[3.0, -3.0]])
    out = apply_repetition_penalty(logits, recent_ids=[0, 1], penalty=1.0)
    assert out.tolist() == [[3.0, -3.0]]
    apply_repetition_penalty(logits, recent_ids=[0, 1], penalty=2.0)
    assert logits.tolist() == [[3.0, -3.0]]


def test_detects_a_short_phrase_stuck_on_repeat():
    i_think = [40, 1101]
    assert is_looping(i_think * 8, min_repeats=3, min_span=16, max_period=32)


def test_detects_a_long_sentence_repeated_three_times():
    sentence = list(range(100, 125))
    assert is_looping([7, 8, 9] + sentence * 3, min_repeats=3, min_span=16, max_period=32)


def test_ignores_short_bursts_like_ellipses():
    assert not is_looping([5, 6, 7] + [46] * 6, min_repeats=3, min_span=16, max_period=32)


def test_ignores_normal_varied_text():
    assert not is_looping(list(range(200)), min_repeats=3, min_span=16, max_period=32)


def test_repeat_must_be_at_the_end_of_the_output():
    stuck_then_recovered = [1, 2] * 10 + list(range(50, 70))
    assert not is_looping(stuck_then_recovered, min_repeats=3, min_span=16, max_period=32)
