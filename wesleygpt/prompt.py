# Wesley wrote this
"""Chat history -> prompt token ids, rendered exactly like nanochat's SFT data.

A system message is folded into the first user message (that's what
tokenizer.render_conversation does at training time, so the model has never seen
a separate system turn). When the history is too long, whole turns are dropped
from the front -- the model only has a 2048-token window and long chats are where
it falls into repetition loops anyway.
"""


class PromptError(ValueError):
    pass


def _validate(messages):
    if not messages:
        raise PromptError("conversation needs at least one message")
    system = None
    if messages[0]["role"] == "system":
        system, messages = messages[0]["content"], messages[1:]
    for i, message in enumerate(messages):
        if message["role"] == "system":
            raise PromptError("a system message is only allowed as the first message")
        expected = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise PromptError(f"roles must alternate user/assistant; message {i} is '{message['role']}'")
    if not messages or messages[-1]["role"] != "user":
        raise PromptError("the last message must be from the user")
    return system, messages


def _render(turns, system, tok):
    user_start, user_end = tok.encode_special("<|user_start|>"), tok.encode_special("<|user_end|>")
    assistant_start = tok.encode_special("<|assistant_start|>")
    assistant_end = tok.encode_special("<|assistant_end|>")
    ids = [tok.get_bos_token_id()]
    for i, message in enumerate(turns):
        content = message["content"]
        if message["role"] == "user":
            if i == 0 and system is not None:
                content = system + "\n\n" + content
            ids += [user_start, *tok.encode(content), user_end]
        else:
            ids += [assistant_start, *tok.encode(content), assistant_end]
    return ids + [assistant_start]


def render_chat_prompt(messages, tok, max_prompt_tokens):
    """Token ids ending in <|assistant_start|>, trimmed from the front to fit max_prompt_tokens."""
    system, turns = _validate(messages)
    # Drop (user, assistant) pairs from the front so the history always starts on a user turn.
    for start in range(0, len(turns), 2):
        ids = _render(turns[start:], system, tok)
        if len(ids) <= max_prompt_tokens:
            return ids
    raise PromptError(f"last message is too long: {len(ids)} tokens, limit is {max_prompt_tokens}")
