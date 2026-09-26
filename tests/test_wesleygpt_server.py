# Wesley wrote this
"""The OpenAI-compatible HTTP surface, against a fake runtime (no checkpoint needed)."""
import json

from fastapi.testclient import TestClient

from wesleygpt.api_schema import Limits
from wesleygpt.decode import DecodeEvent
from wesleygpt.prompt import PromptError
from wesleygpt.server import create_app

KEY = {"Authorization": "Bearer test-key"}
HI = {"messages": [{"role": "user", "content": "hi"}]}


class FakeRuntime:
    models = [{"id": "wesleygpt-d12-chat", "description": "d12 SFT"}]

    def generate(self, req):
        if req.messages[-1]["role"] != "user":
            raise PromptError("the last message must be from the user")
        events = [DecodeEvent("Hel"), DecodeEvent("lo"), DecodeEvent("", "stop", 2)]
        return 7, iter(events)


client = TestClient(create_app(FakeRuntime(), api_keys={"test-key"},
                               limits=Limits(default_max_tokens=256, max_tokens_cap=512)))


def sse_payloads(resp):
    return [line[len("data: "):] for line in resp.text.splitlines() if line.startswith("data: ")]


def test_health_needs_no_key():
    resp = client.get("/health")
    assert resp.status_code == 200 and resp.json()["models"] == ["wesleygpt-d12-chat"]


def test_models_list_needs_a_key():
    assert client.get("/v1/models").status_code == 401


def test_models_list_is_openai_shaped():
    body = client.get("/v1/models", headers=KEY).json()
    assert body["object"] == "list" and body["data"][0]["id"] == "wesleygpt-d12-chat"


def test_chat_with_wrong_key_is_401_with_openai_error_shape():
    resp = client.post("/v1/chat/completions", json=HI, headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401 and resp.json()["error"]["code"] == "invalid_api_key"


def test_non_streaming_reply_has_message_finish_reason_and_usage():
    body = client.post("/v1/chat/completions", json=HI, headers=KEY).json()
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Hello"}
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    assert body["model"] == "wesleygpt-d12-chat" and body["object"] == "chat.completion"


def test_streaming_reply_sends_role_then_deltas_then_finish_then_done():
    resp = client.post("/v1/chat/completions", json={**HI, "stream": True}, headers=KEY)
    assert resp.headers["content-type"].startswith("text/event-stream")
    payloads = sse_payloads(resp)
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    deltas = [c["choices"][0]["delta"] for c in chunks]
    assert deltas[0] == {"role": "assistant"}
    assert "".join(d.get("content", "") for d in deltas) == "Hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_unknown_model_is_404():
    resp = client.post("/v1/chat/completions", json={**HI, "model": "gpt-4"}, headers=KEY)
    assert resp.status_code == 404 and "wesleygpt-d12-chat" in resp.json()["error"]["message"]


def test_prompt_problem_is_400():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    resp = client.post("/v1/chat/completions", json={"messages": msgs}, headers=KEY)
    assert resp.status_code == 400 and "last message" in resp.json()["error"]["message"]


def test_malformed_json_is_400():
    resp = client.post("/v1/chat/completions", content=b"{not json", headers={**KEY, "Content-Type": "application/json"})
    assert resp.status_code == 400 and resp.json()["error"]["type"] == "invalid_request_error"
