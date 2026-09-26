# Wesley wrote this
"""Token-by-token decoding with repetition controls, streaming text as it goes.

The model side is abstracted as `first_logits` (after the prompt) plus `step(token)`
(feed one token, get the next logits), so this loop is testable without a checkpoint.
It mirrors nanochat's Engine.generate -- same sampler, same calculator tool protocol --
and adds the repetition penalty and loop detector the d12 model needs.
"""
from collections import deque
from dataclasses import dataclass

import torch

from nanochat.engine import sample_next_token
from wesleygpt.calculator import calculate
from wesleygpt.sampling import apply_repetition_penalty, is_looping

# How many recent tokens the repetition penalty looks at.
PENALTY_WINDOW = 64


@dataclass(frozen=True)
class DecodeParams:
    max_tokens: int
    temperature: float
    top_k: int
    repetition_penalty: float
    seed: int
    loop_min_repeats: int = 3
    loop_min_span: int = 16
    loop_max_period: int = 32
    max_total_tokens: int | None = None  # sampled + forced; the KV cache's capacity


@dataclass(frozen=True)
class DecodeEvent:
    text: str
    finish_reason: str | None = None  # "stop" | "length" on the last event, else None
    completion_tokens: int | None = None  # sampled-token count, on the last event only


class _TextStream:
    """Turns token ids into text deltas, holding back an incomplete UTF-8 character."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0

    def push(self, token):
        self.ids.append(token)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta

    def flush(self):
        delta = self.tok.decode(self.ids)[self.sent:] if self.ids else ""
        self.ids, self.sent = [], 0
        return delta


def decode_loop(first_logits, step, tok, params):
    special = tok.encode_special
    stop_tokens = {special("<|assistant_end|>"), tok.get_bos_token_id()}
    python_start, python_end = special("<|python_start|>"), special("<|python_end|>")
    output_start, output_end = special("<|output_start|>"), special("<|output_end|>")

    rng = torch.Generator(device=first_logits.device)
    rng.manual_seed(params.seed)
    text = _TextStream(tok)
    forced = deque()
    history, expr_ids, in_python = [], [], False
    sampled, logits = 0, first_logits

    def event(s, finish=None):
        return DecodeEvent(s, finish, sampled if finish else None)

    while True:
        if params.max_total_tokens is not None and len(history) >= params.max_total_tokens:
            yield event(text.flush(), "length")
            return
        if forced:
            token = forced.popleft()
        else:
            if sampled >= params.max_tokens:
                yield event(text.flush(), "length")
                return
            penalized = apply_repetition_penalty(logits, history[-PENALTY_WINDOW:], params.repetition_penalty)
            token = sample_next_token(penalized, rng, params.temperature, params.top_k or None)[0, 0].item()
            sampled += 1

        if token in stop_tokens:
            yield event(text.flush(), "stop")
            return
        # Calculator calls render the way GSM8K writes them: <<12+7=19>>
        if token == python_start:
            in_python, expr_ids = True, []
            yield event(text.flush() + "<<")
        elif token == python_end and in_python:
            in_python = False
            result = calculate(tok.decode(expr_ids))
            yield event(text.flush() + "=")
            if result is not None:
                forced.extend([output_start, *tok.encode(str(result)), output_end])
            else:
                yield event(">>")
        elif token == output_start:
            yield event(text.flush())
        elif token == output_end:
            yield event(text.flush() + ">>")
        else:
            if in_python:
                expr_ids.append(token)
            delta = text.push(token)
            if delta:
                yield event(delta)

        history.append(token)
        if is_looping(history, params.loop_min_repeats, params.loop_min_span, params.loop_max_period):
            yield event(text.flush(), "stop")
            return
        logits = step(token)
