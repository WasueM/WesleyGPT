# Wesley wrote this
"""Validation for OpenAI-style /v1/chat/completions requests.

Standard fields: model, messages, max_tokens (or max_completion_tokens), temperature,
stream, n, seed. Extensions: top_k and repetition_penalty. Unknown fields are ignored,
like OpenAI-compatible servers usually do, so stock clients that send top_p etc. work.
"""
from dataclasses import dataclass


class RequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


@dataclass(frozen=True)
class Limits:
    default_max_tokens: int
    max_tokens_cap: int


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list
    max_tokens: int
    temperature: float
    top_k: int
    repetition_penalty: float
    stream: bool
    seed: int | None


# chat_cli's sampling settings; the repetition penalty is a starting guess to tune.
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.1


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _messages(raw):
    if not isinstance(raw, list) or not raw:
        raise RequestError(400, "messages must be a non-empty list")
    for i, m in enumerate(raw):
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") \
                or not isinstance(m.get("content"), str):
            raise RequestError(400, f"messages[{i}] must have role system|user|assistant and string content")
    return [{"role": m["role"], "content": m["content"]} for m in raw]


def _number(body, key, default, lo, hi):
    value = body.get(key, default)
    if not _is_number(value) or not lo <= value <= hi:
        raise RequestError(400, f"{key} must be a number between {lo} and {hi}")
    return float(value)


def parse_chat_request(body, model_ids, limits, aliases=None):
    if not isinstance(body, dict):
        raise RequestError(400, "request body must be a JSON object")
    messages = _messages(body.get("messages"))

    model = body.get("model") or model_ids[0]
    model = (aliases or {}).get(model, model)
    if model not in model_ids:
        raise RequestError(404, f"model '{model}' not found; available: {', '.join(model_ids)}")

    max_tokens = body.get("max_tokens", body.get("max_completion_tokens", limits.default_max_tokens))
    if not _is_int(max_tokens) or max_tokens < 1:
        raise RequestError(400, "max_tokens must be a positive integer")

    top_k = body.get("top_k", DEFAULT_TOP_K)
    if not _is_int(top_k) or top_k < 0:
        raise RequestError(400, "top_k must be a non-negative integer (0 = no top-k)")

    if body.get("n", 1) != 1:
        raise RequestError(400, "n must be 1; only one choice per request is supported")

    seed = body.get("seed")
    if seed is not None and not _is_int(seed):
        raise RequestError(400, "seed must be an integer")

    return ChatRequest(
        model=model,
        messages=messages,
        max_tokens=min(max_tokens, limits.max_tokens_cap),
        temperature=_number(body, "temperature", DEFAULT_TEMPERATURE, 0.0, 2.0),
        top_k=top_k,
        repetition_penalty=_number(body, "repetition_penalty", DEFAULT_REPETITION_PENALTY, 1.0, 2.0),
        stream=bool(body.get("stream", False)),
        seed=seed,
    )
