# Wesley wrote this
"""The home-PC model API: request parsing, uploads, streaming and the GPU-busy rule.

Nothing here loads a model: the engine is a fake, so these run on any machine.
"""
import json
import threading
import urllib.error
import urllib.request

import pytest

from wesleyqwen import api
from wesleyqwen.api import (RequestError, ThinkRouter, UploadError, gpu_busy_reason, load_registry,
                            mint_upload_token, parse_chat, timed_media, upload_kind, verify_upload_token)

KEY = "test-key"
NOW = 1_900_000_000

MODELS = [
    {"id": "eyes", "summary": "s", "description": "d", "path": "m/eyes", "processor": "m/eyes",
     "modalities": ["text", "image", "video"], "thinking": True, "family": "qwen"},
    {"id": "ears", "summary": "s", "description": "d", "path": "m/ears", "processor": "m/ears",
     "modalities": ["text", "image", "video", "audio"], "thinking": False, "family": "gemma"},
]


@pytest.fixture
def uploads(tmp_path):
    for name in ("a1b2c3d4e5f60718293a4b5c6d7e8f90.mp4", "0123456789abcdef0123456789abcdef.jpg",
                 "fedcba9876543210fedcba9876543210.m4a"):
        (tmp_path / name).write_bytes(b"x")
    return str(tmp_path)


VIDEO = "a1b2c3d4e5f60718293a4b5c6d7e8f90.mp4"
IMAGE = "0123456789abcdef0123456789abcdef.jpg"
AUDIO = "fedcba9876543210fedcba9876543210.m4a"


def body(content, model="eyes", **extra):
    return {"model": model, "messages": [{"role": "user", "content": content}], **extra}


# --- the model registry ---------------------------------------------------------------------------

def test_registry_paths_expand_environment_variables(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps([{**MODELS[0], "path": "${BASE}/qwen", "processor": "${BASE}/qwen"}]))
    [model] = load_registry(str(path), {"BASE": "/models"})
    assert model["path"] == "/models/qwen" and model["processor"] == "/models/qwen"


def test_registry_rejects_a_modality_it_does_not_know(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps([{**MODELS[0], "modalities": ["text", "smell"]}]))
    with pytest.raises(ValueError, match="smell"):
        load_registry(str(path), {})


def test_registry_rejects_an_unset_variable_rather_than_a_blank_path(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps([{**MODELS[0], "path": "${NOT_SET}/qwen"}]))
    with pytest.raises(ValueError, match="NOT_SET"):
        load_registry(str(path), {})


def test_a_model_is_a_qwen_unless_its_entry_says_otherwise(tmp_path):
    path = tmp_path / "models.json"
    entry = {"id": "m", "summary": "s", "description": "d", "path": "p", "modalities": ["text"]}
    path.write_text(json.dumps([entry, {**entry, "id": "g", "family": "gemma"}]))
    assert [m["family"] for m in load_registry(str(path), {})] == ["qwen", "gemma"]


def test_the_registry_rejects_a_family_it_does_not_know(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps([{"id": "m", "summary": "s", "description": "d", "path": "p",
                                 "modalities": ["text"], "family": "llama"}]))
    with pytest.raises(ValueError, match="family"):
        load_registry(str(path), {})


def test_the_shipped_registry_offers_gemma_with_ears():
    models = {m["id"]: m for m in load_registry(api.REGISTRY, {"WESLEYQWEN_BASE": "/base", "WESLEYQWEN_GEMMA": "/g"})}
    assert models["gemma-4-e2b"]["family"] == "gemma"
    assert "audio" in models["gemma-4-e2b"]["modalities"]


def test_the_shipped_registry_loads():
    models = load_registry(api.REGISTRY, {"WESLEYQWEN_BASE": "/base", "WESLEYQWEN_GEMMA": "/g"})
    assert {m["id"] for m in models} >= {"qwen3.5-2b", "wesleyqwen-full", "wesleyqwen-lora", "wesleyqwen-qlora"}


# --- chat requests ------------------------------------------------------------------------------

def test_a_plain_text_chat_parses_with_defaults(uploads):
    req = parse_chat(body("hi"), MODELS, uploads)
    assert req.model["id"] == "eyes"
    assert req.turns == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert req.stream is False and req.thinking is False
    assert req.max_tokens == api.DEFAULT_MAX_TOKENS


def test_an_unknown_model_is_a_404(uploads):
    with pytest.raises(RequestError) as e:
        parse_chat(body("hi", model="nope"), MODELS, uploads)
    assert e.value.status == 404 and "eyes" in e.value.message


def test_max_tokens_is_capped(uploads):
    assert parse_chat(body("hi", max_tokens=10**9), MODELS, uploads).max_tokens == api.MAX_TOKENS_CAP


def test_thinking_is_switched_on_the_way_vllm_spells_it(uploads):
    req = parse_chat(body("hi", chat_template_kwargs={"enable_thinking": True}), MODELS, uploads)
    assert req.thinking is True


def test_thinking_is_refused_for_a_model_whose_thinking_is_broken(uploads):
    with pytest.raises(RequestError, match="thinking"):
        parse_chat(body("hi", model="ears", chat_template_kwargs={"enable_thinking": True}), MODELS, uploads)


def test_media_parts_point_at_uploaded_files(uploads):
    content = [{"type": "image_url", "image_url": {"url": f"upload:{IMAGE}"}},
               {"type": "video_url", "video_url": {"url": f"upload:{VIDEO}"}},
               {"type": "text", "text": "what's this?"}]
    turn = parse_chat(body(content), MODELS, uploads).turns[0]
    assert turn["content"] == [{"type": "image", "image": f"{uploads}/{IMAGE}"},
                               {"type": "video", "video": f"{uploads}/{VIDEO}"},
                               {"type": "text", "text": "what's this?"}]


def test_a_url_that_is_not_an_upload_is_refused(uploads):
    """The server never fetches a URL it was handed: that would let a caller aim it anywhere."""
    content = [{"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest"}}]
    with pytest.raises(RequestError, match="upload:"):
        parse_chat(body(content), MODELS, uploads)


def test_an_upload_id_that_climbs_out_of_the_folder_is_refused(uploads):
    content = [{"type": "image_url", "image_url": {"url": "upload:../../etc/passwd"}}]
    with pytest.raises(RequestError, match="upload id"):
        parse_chat(body(content), MODELS, uploads)


def test_an_upload_that_has_expired_or_never_existed_is_a_clear_error(uploads):
    content = [{"type": "video_url", "video_url": {"url": "upload:ffffffffffffffffffffffffffffffff.mp4"}}]
    with pytest.raises(RequestError, match="re-attach"):
        parse_chat(body(content), MODELS, uploads)


def test_a_model_without_ears_refuses_audio(uploads):
    content = [{"type": "audio_url", "audio_url": {"url": f"upload:{AUDIO}"}}]
    with pytest.raises(RequestError, match="eyes cannot take audio"):
        parse_chat(body(content), MODELS, uploads)


def test_a_model_with_ears_takes_audio(uploads):
    content = [{"type": "audio_url", "audio_url": {"url": f"upload:{AUDIO}"}}]
    turn = parse_chat(body(content, model="ears"), MODELS, uploads).turns[0]
    assert turn["content"] == [{"type": "audio", "audio": f"{uploads}/{AUDIO}"}]


def test_a_model_that_hears_takes_one_video_or_audio_file_per_message(uploads):
    content = [{"type": "video_url", "video_url": {"url": f"upload:{VIDEO}"}},
               {"type": "audio_url", "audio_url": {"url": f"upload:{AUDIO}"}}]
    with pytest.raises(RequestError, match="one video or audio"):
        parse_chat(body(content, model="ears"), MODELS, uploads)


def test_a_model_that_hears_sees_earlier_media_only_as_words(uploads):
    first = [{"type": "video_url", "video_url": {"url": f"upload:{VIDEO}"}}, {"type": "text", "text": "what is it?"}]
    req = parse_chat({"model": "ears", "messages": [{"role": "user", "content": first},
                                                    {"role": "assistant", "content": "A dog."},
                                                    {"role": "user", "content": "what colour?"}]}, MODELS, uploads)
    assert req.turns[0]["content"] == "[video] what is it?"
    assert req.turns[2]["content"] == [{"type": "text", "text": "what colour?"}]


def test_a_model_that_only_sees_keeps_earlier_media(uploads):
    first = [{"type": "image_url", "image_url": {"url": f"upload:{IMAGE}"}}, {"type": "text", "text": "what is it?"}]
    req = parse_chat({"model": "eyes", "messages": [{"role": "user", "content": first},
                                                    {"role": "assistant", "content": "A dog."},
                                                    {"role": "user", "content": "what colour?"}]}, MODELS, uploads)
    assert req.turns[0]["content"][0]["type"] == "image"


def test_the_video_or_audio_to_walk_in_windows_is_found_with_its_question(uploads):
    content = [{"type": "audio_url", "audio_url": {"url": f"upload:{AUDIO}"}}, {"type": "text", "text": "who speaks?"}]
    req = parse_chat(body(content, model="ears"), MODELS, uploads)
    kind, path, question = timed_media(req.turns[-1])
    assert (kind, path.endswith(AUDIO), question) == ("audio", True, "who speaks?")


def test_a_message_with_no_video_or_audio_has_nothing_to_walk(uploads):
    req = parse_chat(body("hi", model="ears"), MODELS, uploads)
    assert timed_media(req.turns[-1]) is None


def test_assistant_turns_are_text_only(uploads):
    raw = {"model": "eyes", "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"}]}
    turns = parse_chat(raw, MODELS, uploads).turns
    assert turns[1] == {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}


def test_a_message_with_an_unknown_part_type_is_refused(uploads):
    with pytest.raises(RequestError, match="file"):
        parse_chat(body([{"type": "file", "file": {}}]), MODELS, uploads)


# --- upload passes ------------------------------------------------------------------------------

# Minted by MangumHub's TypeScript (src/lib/wesleygpt-home.test.ts holds the same string), so a
# change to either side's token format fails one of the two suites.
CROSS_LANGUAGE_TOKEN = ("v1.eyJleHAiOjIwMDAwMDAwMDAsImtpbmQiOiJ2aWRlbyIsIm1heCI6MTAwMCwibiI6ImFiYyJ9"
                        ".2qK4GOy0IJuA3TiZC4VHRq4oCOfUHxItg96ytpjD-Bc")


def test_an_upload_pass_round_trips():
    token = mint_upload_token(KEY, kind="video", max_bytes=5_000, now=NOW, nonce="n1")
    assert verify_upload_token(KEY, token, now=NOW + 10) == {"kind": "video", "max": 5_000}


def test_an_upload_pass_from_mangumhub_verifies():
    assert verify_upload_token(KEY, CROSS_LANGUAGE_TOKEN, now=NOW) == {"kind": "video", "max": 1000}


def test_an_expired_upload_pass_is_refused():
    token = mint_upload_token(KEY, kind="video", max_bytes=5_000, now=NOW, nonce="n1")
    with pytest.raises(UploadError, match="expired"):
        verify_upload_token(KEY, token, now=NOW + api.UPLOAD_TOKEN_TTL + 1)


def test_an_upload_pass_signed_with_another_key_is_refused():
    token = mint_upload_token("other-key", kind="video", max_bytes=5_000, now=NOW, nonce="n1")
    with pytest.raises(UploadError, match="signature"):
        verify_upload_token(KEY, token, now=NOW)


def test_an_upload_pass_with_a_raised_size_limit_is_refused():
    token = mint_upload_token(KEY, kind="video", max_bytes=5_000, now=NOW, nonce="n1")
    version, payload, signature = token.split(".")
    forged = api._b64(json.dumps({"exp": NOW + 600, "kind": "video", "max": 10**12, "n": "n1"}).encode())
    with pytest.raises(UploadError, match="signature"):
        verify_upload_token(KEY, f"{version}.{forged}.{signature}", now=NOW)


def test_garbage_is_not_an_upload_pass():
    with pytest.raises(UploadError):
        verify_upload_token(KEY, "Bearer hello", now=NOW)


def test_upload_kind_comes_from_the_content_type():
    assert upload_kind("video/quicktime") == "video"
    assert upload_kind("image/jpeg") == "image"
    assert upload_kind("audio/mp4") == "audio"
    assert upload_kind("application/pdf") is None


# --- thinking, split out of the stream ----------------------------------------------------------

def split(pieces):
    router, out = ThinkRouter(), []
    for piece in pieces:
        out += router.feed(piece)
    out += router.flush()
    merged = {}
    for field, text in out:
        merged[field] = merged.get(field, "") + text
    return merged


def test_text_without_think_tags_is_all_content():
    assert split(["Hello", " there"]) == {"content": "Hello there"}


def test_text_inside_think_tags_is_reasoning():
    assert split(["<think>\nlet me see", "</think>\n\n", "It's 4."]) == \
        {"reasoning_content": "let me see", "content": "It's 4."}


def test_tags_split_across_pieces_are_still_found():
    assert split(["<thi", "nk>\nhmm</th", "ink>\n\nDone"]) == {"reasoning_content": "hmm", "content": "Done"}


def test_thinking_that_never_closes_stays_reasoning():
    assert split(["<think>\nround and round", " </thi"]) == {"reasoning_content": "round and round </thi"}


def test_each_window_of_a_long_video_can_think_again():
    pieces = ["[0:00–0:30] ", "<think>\na</think>\n\nRed.", "\n\n[0:30–1:00] ", "<think>\nb</think>\n\nBlue."]
    assert split(pieces) == {"reasoning_content": "ab", "content": "[0:00–0:30] Red.\n\n[0:30–1:00] Blue."}


def test_a_less_than_sign_that_is_not_a_tag_is_kept():
    assert split(["2 <", " 3 and x<y"]) == {"content": "2 < 3 and x<y"}


# --- who gets the GPU ---------------------------------------------------------------------------

def test_a_running_training_job_holds_the_gpu():
    assert "training" in gpu_busy_reason("running d12-pipeline since 2026-10-10", used_mb=9000, own_mb=0)


def test_a_finished_job_does_not():
    assert gpu_busy_reason("finished d12 rc=0 at 2026-10-10", used_mb=800, own_mb=0) is None


def test_someone_elses_gpu_memory_means_busy():
    assert "in use" in gpu_busy_reason("", used_mb=api.FOREIGN_GPU_MB + 900, own_mb=0)


def test_our_own_model_does_not_count_against_us():
    assert gpu_busy_reason("", used_mb=6000, own_mb=5500) is None


# --- the HTTP server, with a fake engine ---------------------------------------------------------

class FakeEngine:
    def __init__(self):
        self.busy = None
        self.seen = None

    def busy_reason(self):
        return self.busy

    def stream(self, req):
        self.seen = req
        yield from (["<think>\nthinking</think>\n\n", "Hi", "!"] if req.thinking else ["Hi", "!"])
        if req.max_tokens == 2:
            yield api.Finish("length")


@pytest.fixture
def server(uploads):
    engine = FakeEngine()
    config = api.ServerConfig(api_keys={KEY}, upload_key=KEY, uploads=uploads, models=MODELS,
                              allowed_origins={"https://mangumhub.com"}, max_upload_bytes=10_000)
    httpd = api.make_server(engine, config, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", engine, uploads
    httpd.shutdown()


def call(url, method="GET", data=None, headers=None):
    req = urllib.request.Request(url, method=method, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as res:
            return res.status, dict(res.headers), res.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


AUTH = {"Authorization": f"Bearer {KEY}"}


def test_models_need_the_api_key(server):
    url, _, _ = server
    assert call(f"{url}/v1/models")[0] == 401


def test_models_list_says_what_each_model_takes_and_whether_the_gpu_is_free(server):
    url, engine, _ = server
    engine.busy = "a training job is using the GPU"
    status, _, text = call(f"{url}/v1/models", headers=AUTH)
    listed = json.loads(text)
    assert status == 200
    assert listed["status"] == {"ready": False, "reason": "a training job is using the GPU"}
    assert listed["data"][1]["modalities"] == ["text", "image", "video", "audio"]
    assert "path" not in listed["data"][0]


def test_health_is_open_and_says_nothing_private(server):
    url, _, _ = server
    status, _, text = call(f"{url}/health")
    assert status == 200 and json.loads(text) == {"status": "ok"}


def test_a_streamed_reply_is_openai_chunks_with_reasoning_split_out(server):
    url, _, _ = server
    data = json.dumps(body("hi", stream=True, chat_template_kwargs={"enable_thinking": True})).encode()
    status, headers, text = call(f"{url}/v1/chat/completions", "POST", data, {**AUTH, "Content-Type": "application/json"})
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    frames = [json.loads(line[6:]) for line in text.split("\n\n") if line.startswith("data: {")]
    deltas = [f["choices"][0]["delta"] for f in frames]
    assert {"reasoning_content": "thinking"} in deltas
    assert "".join(d.get("content", "") for d in deltas) == "Hi!"
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert text.rstrip().endswith("data: [DONE]")


def test_a_reply_cut_off_by_max_tokens_says_so(server):
    url, _, _ = server
    data = json.dumps(body("hi", stream=True, max_tokens=2)).encode()
    _, _, text = call(f"{url}/v1/chat/completions", "POST", data, AUTH)
    frames = [json.loads(line[6:]) for line in text.split("\n\n") if line.startswith("data: {")]
    assert frames[-1]["choices"][0]["finish_reason"] == "length"


def test_a_non_streamed_reply_is_one_completion(server):
    url, _, _ = server
    status, _, text = call(f"{url}/v1/chat/completions", "POST", json.dumps(body("hi")).encode(), AUTH)
    assert status == 200
    assert json.loads(text)["choices"][0]["message"] == {"role": "assistant", "content": "Hi!"}


def test_chat_while_the_gpu_is_busy_is_a_503_that_says_why(server):
    url, engine, _ = server
    engine.busy = "a training job is using the GPU"
    status, _, text = call(f"{url}/v1/chat/completions", "POST", json.dumps(body("hi")).encode(), AUTH)
    assert status == 503 and "training job" in json.loads(text)["error"]["message"]


def test_a_bad_chat_request_is_a_400_in_openai_shape(server):
    url, _, _ = server
    status, _, text = call(f"{url}/v1/chat/completions", "POST", b"not json", AUTH)
    assert status == 400 and json.loads(text)["error"]["type"] == "invalid_request_error"


def test_an_upload_with_a_pass_lands_in_the_folder(server):
    url, _, uploads = server
    token = mint_upload_token(KEY, kind="image", max_bytes=100, now=api.now(), nonce="n")
    status, headers, text = call(f"{url}/v1/uploads", "PUT", b"\xff\xd8jpeg",
                                 {"Authorization": f"Bearer {token}", "Content-Type": "image/jpeg",
                                  "Origin": "https://mangumhub.com"})
    upload_id = json.loads(text)["id"]
    assert status == 200 and upload_id.endswith(".jpg")
    assert headers["Access-Control-Allow-Origin"] == "https://mangumhub.com"
    assert open(f"{uploads}/{upload_id}", "rb").read() == b"\xff\xd8jpeg"


def test_an_upload_bigger_than_its_pass_is_refused(server):
    url, _, uploads = server
    token = mint_upload_token(KEY, kind="image", max_bytes=3, now=api.now(), nonce="n")
    status, _, _ = call(f"{url}/v1/uploads", "PUT", b"0123456789",
                        {"Authorization": f"Bearer {token}", "Content-Type": "image/jpeg"})
    assert status == 413


def test_an_upload_of_a_different_kind_than_its_pass_is_refused(server):
    url, _, _ = server
    token = mint_upload_token(KEY, kind="image", max_bytes=100, now=api.now(), nonce="n")
    status, _, text = call(f"{url}/v1/uploads", "PUT", b"x",
                           {"Authorization": f"Bearer {token}", "Content-Type": "video/mp4"})
    assert status == 400 and "image" in text


def test_an_upload_without_a_pass_is_refused(server):
    url, _, _ = server
    status, _, _ = call(f"{url}/v1/uploads", "PUT", b"x", {"Content-Type": "image/jpeg"})
    assert status == 401


def test_the_browser_preflight_is_answered_only_for_listed_origins(server):
    url, _, _ = server
    ok = call(f"{url}/v1/uploads", "OPTIONS", headers={"Origin": "https://mangumhub.com",
                                                      "Access-Control-Request-Method": "PUT"})
    other = call(f"{url}/v1/uploads", "OPTIONS", headers={"Origin": "https://evil.example",
                                                         "Access-Control-Request-Method": "PUT"})
    assert ok[0] == 204 and ok[1]["Access-Control-Allow-Origin"] == "https://mangumhub.com"
    assert "Access-Control-Allow-Origin" not in other[1]


def test_uploads_older_than_a_week_are_deleted_and_newer_ones_kept(tmp_path):
    import os
    old, fresh = tmp_path / "old.mp4", tmp_path / "fresh.mp4"
    old.write_bytes(b"x")
    fresh.write_bytes(b"x")
    os.utime(old, (NOW - api.UPLOAD_MAX_AGE_SECONDS - 60,) * 2)
    os.utime(fresh, (NOW - 60,) * 2)
    api.prune_uploads(str(tmp_path), now=NOW)
    assert not old.exists() and fresh.exists()


# --- keep-alives while something slow runs --------------------------------------------------------

def drain(gen):
    try:
        while True:
            next(gen)
    except StopIteration as done:
        return done.value


def test_slow_work_hands_back_its_result():
    assert drain(api.working(lambda a, b: a + b, 2, 3)) == 5


def test_slow_work_that_fails_fails_on_the_callers_thread():
    def boom():
        raise RuntimeError("ffmpeg failed")
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        drain(api.working(boom))
