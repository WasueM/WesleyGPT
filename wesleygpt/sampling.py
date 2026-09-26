# Wesley wrote this
"""Repetition controls. The d12 model falls into loops ("I think I think I think...")
and nanochat's Engine has no penalty, so the serving layer adds two:
a soft one (repetition penalty on recent tokens) and a hard stop (loop detector).
"""
import torch


def apply_repetition_penalty(logits, recent_ids, penalty):
    """CTRL-style penalty (Keskar et al. 2019): make recently used tokens less likely.

    Positive logits are divided by `penalty`, negative ones multiplied, so the token is
    pushed down either way. Each token is penalised once however often it appeared.
    Returns a new tensor; `logits` (shape (B, vocab)) is not modified.
    """
    if penalty == 1.0 or not recent_ids:
        return logits
    out = logits.clone()
    idx = torch.tensor(sorted(set(recent_ids)), dtype=torch.long, device=logits.device)
    chosen = out[:, idx]
    out[:, idx] = torch.where(chosen > 0, chosen / penalty, chosen * penalty)
    return out


def is_looping(ids, min_repeats, min_span, max_period):
    """True if the output currently ends in one chunk repeated back-to-back.

    Some period p (1..max_period) must repeat at least `min_repeats` times AND cover at
    least `min_span` tokens -- the span floor keeps "..." or "!!!" from counting.
    """
    n = len(ids)
    for period in range(1, max_period + 1):
        repeats_needed = max(min_repeats, -(-min_span // period))
        span = period * repeats_needed
        if span > n:
            continue
        tail = ids[n - span:]
        if all(tail[i] == tail[i % period] for i in range(period, span)):
            return True
    return False
