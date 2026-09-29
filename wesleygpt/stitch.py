# Wesley wrote this
"""Stitch short conversations from different datasets into one long conversation.

Almost every SFT conversation is one to three exchanges long (SmolTalk: 39% one
user turn, 59% three, ~1% four or more), so the chat models have barely seen a
fourth turn. A stitched conversation is a run of whole exchanges -- a SmolTalk
chat, then a math problem, then an MMLU question, then a SmolTalk chat -- each
drawn from a different dataset than the one before it, until the next would
overflow a token target near the 2,048-token context. Every assistant turn in
it is trained on, so the model practises answering well at turn 8 with a long,
unrelated history above it.

The exchanges don't refer to each other. That teaches staying on the current
question with a long history, not using the history; the recall eval in
wesleygpt.longcontext_eval measures whether the second comes along anyway.
"""


def _trim_dangling(messages):
    """`messages` up to its last assistant turn. About 1 in 3,000 SmolTalk chats ends on
    an unanswered user turn, which stitched before another exchange would put two user
    turns in a row."""
    while messages and messages[-1]["role"] != "assistant":
        messages = messages[:-1]
    return messages


def fold_system(messages):
    """`messages` with a leading system prompt merged into the first user turn, the way
    nanochat's render_conversation does it, so a SmolTalk chat can sit mid-conversation."""
    if not messages or messages[0]["role"] != "system":
        return messages
    first_user = dict(messages[1], content=messages[0]["content"] + "\n\n" + messages[1]["content"])
    return [first_user, *messages[2:]]


def stitch(draw, count_tokens, target_tokens, rng, max_misses=8, max_draws=64):
    """Messages of one conversation of at most `target_tokens` tokens (BOS included);
    the flattened stitch_segments."""
    return [m for segment in stitch_segments(draw, count_tokens, target_tokens, rng, max_misses, max_draws) for m in segment]


def stitch_segments(draw, count_tokens, target_tokens, rng, max_misses=8, max_draws=64):
    """One conversation of at most `target_tokens` tokens (BOS included), as the list of
    exchanges drawn, each kept whole.

    draw(rng) -> (source, messages): one whole exchange from some dataset.
    count_tokens(messages) -> its rendered length without the BOS token.
    Stops after `max_misses` exchanges that didn't fit, so the last gap is filled by
    short exchanges (an MMLU question, a greeting) when there are any. May return []
    if nothing fits; the caller draws again.
    """
    segments, total, last_source, misses = [], 1, None, 0
    for _ in range(max_draws):
        source, exchange = draw(rng)
        if source == last_source:
            continue
        exchange = _trim_dangling(fold_system(exchange))
        if not exchange:
            continue
        n = count_tokens(exchange)
        if total + n > target_tokens:
            misses += 1
            if misses >= max_misses:
                break
            continue
        segments.append(exchange)
        total += n
        last_source = source
    return segments
