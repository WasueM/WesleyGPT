# Wesley wrote this
"""OpenAI-compatible HTTP API: GET /health, GET /v1/models, POST /v1/chat/completions.

Speaking OpenAI's wire format means any OpenAI client library works by pointing its
base_url here. The runtime (which owns the model) is injected, so this module never
touches torch and its tests run without a checkpoint.
"""
import json
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from wesleygpt.api_schema import RequestError, parse_chat_request
from wesleygpt.auth import bearer_key_is_valid
from wesleygpt.prompt import PromptError

log = logging.getLogger("wesleygpt")


def _error(status, message, code=None, type_="invalid_request_error"):
    return JSONResponse({"error": {"message": message, "type": type_, "code": code}}, status_code=status)


def _chunk(completion_id, created, model, delta, finish_reason=None):
    return "data: " + json.dumps({
        "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }) + "\n\n"


def create_app(runtime, api_keys, limits):
    app = FastAPI(title="WesleyGPT", docs_url=None, redoc_url=None)
    model_ids = [m["id"] for m in runtime.models]

    def unauthorized(request):
        if bearer_key_is_valid(request.headers.get("authorization"), api_keys):
            return None
        return _error(401, "missing or invalid API key (send 'Authorization: Bearer <key>')", "invalid_api_key")

    @app.get("/health")
    def health():
        return {"status": "ok", "models": model_ids}

    @app.get("/v1/models")
    def list_models(request: Request):
        if (denied := unauthorized(request)) is not None:
            return denied
        return {"object": "list", "data": [
            {"id": m["id"], "object": "model", "owned_by": "wesley", "description": m["description"]}
            for m in runtime.models]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if (denied := unauthorized(request)) is not None:
            return denied
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "request body is not valid JSON")
        try:
            req = parse_chat_request(body, model_ids, limits)
            prompt_tokens, events = runtime.generate(req)
        except RequestError as e:
            return _error(e.status, e.message, "model_not_found" if e.status == 404 else None)
        except PromptError as e:
            return _error(400, str(e))

        completion_id, created, started = f"chatcmpl-{uuid.uuid4().hex}", int(time.time()), time.monotonic()

        def log_done(final):
            elapsed = time.monotonic() - started
            # One JSON line per request: Cloud Logging parses it into fields.
            log.info(json.dumps({
                "event": "chat_completion", "model": req.model, "stream": req.stream,
                "prompt_tokens": prompt_tokens, "completion_tokens": final.completion_tokens,
                "finish_reason": final.finish_reason, "seconds": round(elapsed, 3),
                "tokens_per_second": round((final.completion_tokens or 0) / elapsed, 1) if elapsed else None,
            }))

        if req.stream:
            # A sync generator: Starlette runs it in a worker thread, so decoding
            # doesn't block the event loop.
            def stream():
                yield _chunk(completion_id, created, req.model, {"role": "assistant"})
                for ev in events:
                    if ev.text:
                        yield _chunk(completion_id, created, req.model, {"content": ev.text})
                    if ev.finish_reason:
                        log_done(ev)
                        yield _chunk(completion_id, created, req.model, {}, ev.finish_reason)
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        parts, final = [], None
        for ev in events:
            parts.append(ev.text)
            if ev.finish_reason:
                final = ev
        log_done(final)
        return {
            "id": completion_id, "object": "chat.completion", "created": created, "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                         "finish_reason": final.finish_reason}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": final.completion_tokens,
                      "total_tokens": prompt_tokens + final.completion_tokens},
        }

    return app
