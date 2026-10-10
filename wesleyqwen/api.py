# Wesley wrote this
"""The home-PC model API: Qwen3.5-2B, the WesleyQwen variants and Gemma 4 E2B behind
OpenAI's wire format, so MangumHub's /wesleygpt page can chat with them.

    python -m wesleyqwen.api --port 8090        # inside WSL; see wesley/pc/wesleyqwen-api.sh

GET  /health                 liveness, says nothing else
GET  /v1/models              the models in api_models.json, what each can take in, and
                             whether the GPU is free right now
POST /v1/chat/completions    OpenAI chat format; content may be a list of parts
PUT  /v1/uploads             a browser uploads one image, video or audio file here

Media is never fetched from a URL. A browser PUTs the file to /v1/uploads with a short-lived
upload pass that MangumHub signed (the pass carries the kind and a size limit), gets back an
id, and later turns refer to it as {"type": "video_url", "video_url": {"url": "upload:<id>"}}.
Fetching caller-supplied URLs would let anyone aim this machine at any address.

The GPU belongs to training first. While a nanochat job runs, or another program holds
the GPU's memory, chats get a 503 that says why, and an idle model unloads itself.
One model sits on the GPU at a time and replies are one at a time: a 2B model and its
generation memory leave no room for a second on 12 GB.

Gemma also hears. It takes at most 30 s of audio per clip, so a video or recording is walked
in 30 s windows, each one answered in turn as wesleyqwen/chat.py does, and only the words
of earlier turns are kept: re-reading every earlier video at every turn would not fit.
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty
from threading import Event, Lock, Thread, Timer

from wesleygpt.auth import bearer_key_is_valid
from wesleyqwen import media
from wesleyqwen.models import GEMMA_FAMILY, QWEN_FAMILY, GemmaChannels
from wesleyqwen.scoring import final_answer

REGISTRY = os.path.join(os.path.dirname(__file__), "api_models.json")
MODALITIES = ("text", "image", "video", "audio")
DEFAULT_MAX_TOKENS = 1024
MAX_TOKENS_CAP = 4096
UPLOAD_TOKEN_TTL = 15 * 60
# The Windows desktop alone holds ~1 GB of the GPU; past this, something else is running on it.
FOREIGN_GPU_MB = 3000
# A long video reads silently for minutes before its first word; an idle connection gets dropped.
HEARTBEAT_SECONDS = 15
IDLE_UNLOAD_SECONDS = 10 * 60
UPLOAD_MAX_AGE_SECONDS = 7 * 24 * 3600
UPLOAD_ID = re.compile(r"^[0-9a-f]{32}\.[a-z0-9]{2,5}$")
UPLOAD_CHUNK = 1 << 20
# Two frames a second reads an 8 s clip as ~1.8k prompt tokens; prompt cost grows linearly with both.
VIDEO_FPS = 2
EXTENSIONS = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif",
    "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
    "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/aac": ".aac",
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/webm": ".weba", "audio/ogg": ".ogg",
}
PART_KINDS = {"image_url": "image", "video_url": "video", "audio_url": "audio"}
FAMILIES = {family.name: family for family in (QWEN_FAMILY, GEMMA_FAMILY)}
TIMED = ("video", "audio")
DEFAULT_QUESTION = {"video": "Describe what happens in this video.", "audio": "Describe what you hear."}


def now():
    return int(time.time())


@dataclass(frozen=True)
class Finish:
    """The last thing an engine's stream yields when the reply ended for a reason other than "stop"."""
    reason: str


class RequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


class UploadError(Exception):
    pass


# --- the model registry ---------------------------------------------------------------------------

def _expand(value, env):
    def var(match):
        name = match.group(1)
        if not env.get(name):
            raise ValueError(f"api_models.json uses ${{{name}}}, which is not set")
        return env[name]
    return re.sub(r"\$\{(\w+)\}", var, value)


def load_registry(path, env):
    """The models this server offers, with ${VAR}s in their paths filled in from env."""
    with open(path) as f:
        entries = json.load(f)
    models = []
    for entry in entries:
        missing = {"id", "summary", "description", "path", "modalities"} - set(entry)
        if missing:
            raise ValueError(f"model {entry.get('id', '?')} in {path} is missing {', '.join(sorted(missing))}")
        unknown = set(entry["modalities"]) - set(MODALITIES)
        if unknown or "text" not in entry["modalities"]:
            raise ValueError(f"model {entry['id']} has modalities {entry['modalities']}; "
                             f"each must be one of {MODALITIES}, and text is required")
        family = entry.get("family", QWEN_FAMILY.name)
        if family not in FAMILIES:
            raise ValueError(f"model {entry['id']} has family {family!r}; each must be one of {', '.join(FAMILIES)}")
        path_ = _expand(entry["path"], env)
        models.append({**entry, "path": path_, "processor": _expand(entry.get("processor", path_), env),
                       "thinking": bool(entry.get("thinking", False)), "family": family})
    return models


def public_model(model):
    return {"id": model["id"], "object": "model", "owned_by": "wesley", "summary": model["summary"],
            "description": model["description"], "modalities": model["modalities"], "thinking": model["thinking"]}


# --- chat requests ------------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatRequest:
    model: dict
    turns: list
    max_tokens: int
    temperature: float | None
    thinking: bool
    stream: bool


def upload_path(folder, upload_id):
    if not UPLOAD_ID.match(upload_id):
        raise RequestError(400, f"{upload_id!r} is not an upload id")
    path = os.path.join(folder, upload_id)
    if not os.path.isfile(path):
        raise RequestError(400, f"upload {upload_id} is gone (uploads are kept a week); re-attach the file")
    return path


def _media_part(part, model, uploads):
    kind = PART_KINDS[part["type"]]
    url = (part.get(part["type"]) or {}).get("url", "")
    if not isinstance(url, str) or not url.startswith("upload:"):
        raise RequestError(400, f"{part['type']} url must be an upload:<id> reference from /v1/uploads")
    if kind not in model["modalities"]:
        raise RequestError(400, f"{model['id']} cannot take {kind}; it takes {', '.join(model['modalities'])}")
    return {"type": kind, kind: upload_path(uploads, url[len("upload:"):])}


def _content(message, index, model, uploads):
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list) or not content:
        raise RequestError(400, f"messages[{index}].content must be a string or a non-empty list of parts")
    parts = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text" and isinstance(part.get("text"), str):
            parts.append({"type": "text", "text": part["text"]})
        elif kind in PART_KINDS and message["role"] == "user":
            parts.append(_media_part(part, model, uploads))
        else:
            raise RequestError(400, f"messages[{index}] has an unsupported content part type {kind!r} "
                                    f"(text, image_url, video_url and audio_url; media only in user turns)")
    return parts


def _in_words(content):
    """A turn's parts as text, each file reduced to a mention: what a hearing model keeps of earlier turns."""
    return " ".join(part["text"] if part["type"] == "text" else f"[{part['type']}]" for part in content)


def timed_media(turn):
    """(kind, path, question) for the video or audio a turn carries, or None if it has neither."""
    if not isinstance(turn["content"], list):
        return None
    timed = [part for part in turn["content"] if part["type"] in TIMED]
    if not timed:
        return None
    kind = timed[0]["type"]
    question = " ".join(part["text"] for part in turn["content"] if part["type"] == "text").strip()
    return kind, timed[0][kind], question or DEFAULT_QUESTION[kind]


def parse_chat(body, models, uploads):
    if not isinstance(body, dict):
        raise RequestError(400, "request body must be a JSON object")
    by_id = {m["id"]: m for m in models}
    model = by_id.get(body.get("model"))
    if model is None:
        raise RequestError(404, f"model {body.get('model')!r} not found; available: {', '.join(by_id)}")

    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise RequestError(400, "messages must be a non-empty list")
    turns = []
    for i, message in enumerate(raw):
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise RequestError(400, f"messages[{i}] must have role system|user|assistant")
        turns.append({"role": message["role"], "content": _content(message, i, model, uploads)})
    if FAMILIES[model["family"]].hears_audio:
        last = turns[-1]["content"]
        files = [part for part in last if part["type"] != "text"]
        if len(files) > 1 and any(part["type"] in TIMED for part in files):
            raise RequestError(400, f"{model['id']} takes one video or audio file per message (it walks it "
                                    f"in 30 s windows); send other files in messages of their own")
        turns = [{**turn, "content": _in_words(turn["content"])} for turn in turns[:-1]] + [turns[-1]]

    kwargs = body.get("chat_template_kwargs") or {}
    thinking = bool(kwargs.get("enable_thinking", False)) if isinstance(kwargs, dict) else False
    if thinking and not model["thinking"]:
        raise RequestError(400, f"{model['id']} has thinking switched off: its fine-tune broke it "
                                f"(it never closes the <think> block)")

    max_tokens = body.get("max_tokens", body.get("max_completion_tokens", DEFAULT_MAX_TOKENS))
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise RequestError(400, "max_tokens must be a positive integer")

    temperature = body.get("temperature")
    if temperature is not None and (not isinstance(temperature, (int, float)) or isinstance(temperature, bool)
                                    or not 0 <= temperature <= 2):
        raise RequestError(400, "temperature must be a number between 0 and 2")

    return ChatRequest(model=model, turns=turns, max_tokens=min(max_tokens, MAX_TOKENS_CAP),
                       temperature=temperature, thinking=thinking, stream=bool(body.get("stream", False)))


# --- upload passes ------------------------------------------------------------------------------

def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _signature(key, payload):
    # The prefix keeps a pass from ever doubling as anything else signed with the same key.
    return _b64(hmac.new(key.encode(), f"wesleyqwen-upload.{payload}".encode(), hashlib.sha256).digest())


def mint_upload_token(key, kind, max_bytes, now, nonce, ttl=UPLOAD_TOKEN_TTL):
    """What MangumHub's src/lib/wesleygpt-home.ts mints; here for tests and manual uploads."""
    payload = _b64(json.dumps({"exp": now + ttl, "kind": kind, "max": max_bytes, "n": nonce},
                              separators=(",", ":")).encode())
    return f"v1.{payload}.{_signature(key, payload)}"


def verify_upload_token(key, token, now):
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "v1":
        raise UploadError("not an upload pass")
    _, payload, signature = parts
    if not hmac.compare_digest(signature, _signature(key, payload)):
        raise UploadError("upload pass signature does not match")
    claims = json.loads(_unb64(payload))
    if claims["exp"] < now:
        raise UploadError("upload pass expired; ask for a new one")
    return {"kind": claims["kind"], "max": claims["max"]}


def upload_kind(content_type):
    family = (content_type or "").split("/")[0]
    return family if family in ("image", "video", "audio") else None


def prune_uploads(folder, now, max_age=UPLOAD_MAX_AGE_SECONDS):
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age:
            os.remove(path)


# --- streaming ------------------------------------------------------------------------------------

class ThinkRouter:
    """Route streamed text inside <think>...</think> to reasoning_content and the rest to content.

    The engine marks every thinking block with the tags (opening one itself where the chat
    template opened it in the prompt), and a video walked in windows thinks once per window.
    A tag can arrive split across pieces, so a tail that could be its start is held back until
    the next piece settles it.
    """
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.inside = False
        self.held = ""
        self.strip_newlines = False

    def _field(self):
        return "reasoning_content" if self.inside else "content"

    def feed(self, piece):
        text, self.held, out = self.held + piece, "", []
        while text:
            if self.strip_newlines:  # the template puts newlines after each tag
                text = text.lstrip("\n")
                if not text:
                    break
                self.strip_newlines = False
            tag = self.CLOSE if self.inside else self.OPEN
            if (at := text.find(tag)) >= 0:
                if text[:at]:
                    out.append((self._field(), text[:at]))
                self.inside, self.strip_newlines = not self.inside, True
                text = text[at + len(tag):]
                continue
            keep = next((n for n in range(len(tag) - 1, 0, -1) if text.endswith(tag[:n])), 0)
            if text[:len(text) - keep]:
                out.append((self._field(), text[:len(text) - keep]))
            self.held = text[len(text) - keep:]
            break
        return out

    def flush(self):
        held, self.held = self.held, ""
        return [(self._field(), held)] if held else []


def _chunk(completion_id, created, model, delta, finish_reason=None):
    return "data: " + json.dumps({
        "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}) + "\n\n"


# --- who gets the GPU ---------------------------------------------------------------------------

def gpu_busy_reason(job_status, used_mb, own_mb):
    """Why chat must wait for the GPU, or None. job_status is ~/jobs/status (wesley/pc/jobs/runjob.sh)."""
    if job_status.startswith("running"):
        name = (job_status.split() + ["", "?"])[1]
        return f"a training job ({name}) is using the GPU; the home-PC models are back when it finishes"
    foreign = used_mb - own_mb
    if foreign > FOREIGN_GPU_MB:
        return (f"another program has {foreign / 1024:.1f} GB of the GPU in use "
                f"(a game or a training run); try again when it's done")
    return None


class Engine:
    """Owns the GPU: loads one model at a time, generates one reply at a time, and steps aside."""

    def __init__(self, job_status_path):
        self.job_status_path = job_status_path
        self.lock = Lock()
        self.loaded = None
        self.model = self.processor = self.family = None
        self.unload_timer = None

    def _own_mb(self):
        import torch
        return torch.cuda.memory_reserved() // 2**20 if self.loaded else 0

    def busy_reason(self):
        try:
            with open(self.job_status_path) as f:
                status = f.read()
        except FileNotFoundError:
            status = ""
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.split()[0]
        reason = gpu_busy_reason(status, int(used), self._own_mb())
        if reason and self.loaded and self.lock.acquire(blocking=False):
            try:
                self._unload()  # make room for whatever wants the GPU
            finally:
                self.lock.release()
        return reason

    def _unload(self):
        import gc
        import torch
        if self.loaded:
            print(f"[api] unloading {self.loaded}", flush=True)
        self.model = self.processor = self.family = self.loaded = None
        gc.collect()
        torch.cuda.empty_cache()

    def _unload_if_idle(self):
        if self.lock.acquire(blocking=False):
            try:
                self._unload()
            finally:
                self.lock.release()

    def _load(self, entry):
        if self.loaded == entry["id"]:
            return
        import torch
        import transformers
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()  # a bar per load fills the log with carriage returns
        self._unload()
        print(f"[api] loading {entry['id']} from {entry['path']}", flush=True)
        self.family = FAMILIES[entry["family"]]
        tokenizer = transformers.AutoTokenizer.from_pretrained(entry["path"])
        # The WesleyQwen fine-tunes saved no image/video preprocessor config, so it comes from the base.
        self.processor = transformers.AutoProcessor.from_pretrained(entry["processor"], tokenizer=tokenizer)
        self.model = getattr(transformers, self.family.loader).from_pretrained(
            entry["path"], dtype=torch.bfloat16, device_map="cuda").eval()
        self.loaded = entry["id"]

    def _stop_ids(self):
        saved = self.model.generation_config.eos_token_id
        if not self.family.stop_ids:
            return saved
        # Qwen3.5 ships no generation_config, so the base model would run past the end of its turn.
        return saved if isinstance(saved, list) and len(saved) > 1 else self.family.stop_ids(self.processor.tokenizer)

    def _generate(self, turns, req, stop, cut_off):
        """Yield one reply's text, its thinking inside <think>...</think>; None means "still working".

        Appends True to cut_off if the reply ran into max_tokens.
        """
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        from wesleyqwen.chat import require_seeking_video_decoder

        ready, failure, box = Event(), [], {}
        family = self.family

        class Stopped(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop.is_set()

        def work():
            try:
                if any(p["type"] == "video" for t in turns if isinstance(t["content"], list) for p in t["content"]):
                    require_seeking_video_decoder()
                batch = self.processor.apply_chat_template(
                    turns, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                    enable_thinking=req.thinking, **family.video_kwargs).to("cuda")
                # Gemma marks its thinking with special tokens; dropping them would lose where it ends.
                streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True,
                                                skip_special_tokens=not family.thinking_in_special_tokens,
                                                timeout=HEARTBEAT_SECONDS)
                box["streamer"] = streamer
                sampling = dict(family.decoding[req.thinking])
                if req.temperature is not None:
                    sampling = {"do_sample": False} if req.temperature == 0 else {**sampling,
                                                                                  "temperature": req.temperature}
                ready.set()
                with torch.inference_mode():
                    out = self.model.generate(**batch, max_new_tokens=req.max_tokens, streamer=streamer,
                                              eos_token_id=self._stop_ids(),
                                              pad_token_id=self.processor.tokenizer.pad_token_id,
                                              stopping_criteria=StoppingCriteriaList([Stopped()]), **sampling)
                box["length"] = out.shape[1] - batch["input_ids"].shape[1] >= req.max_tokens
            except Exception as error:  # re-raised in the reader; ending the stream keeps it from hanging
                failure.append(error)
                if "streamer" in box:
                    box["streamer"].end()
                ready.set()

        worker = Thread(target=work, daemon=True)
        worker.start()
        try:
            while not ready.wait(HEARTBEAT_SECONDS):
                yield None  # reading a long video into the prompt
            if failure:
                raise failure[0]
            if req.thinking and family.thinking_opens_in_prompt:
                yield "<think>\n"
            channels = GemmaChannels() if family.thinking_in_special_tokens else None
            streamer = box["streamer"]
            while True:
                try:
                    piece = next(streamer)
                except StopIteration:
                    break
                except Empty:
                    yield None
                    continue
                yield channels.feed(piece) if channels else piece
            if channels:
                yield channels.flush()
            worker.join()
            if failure:
                raise failure[0]
            if box.get("length"):
                cut_off.append(True)
        finally:
            stop.set()  # the caller hung up (or we finished): end generation
            worker.join()

    def _windows(self, req, turns, kind, path, question, cut_off):
        """Walk a video or recording in 30 s windows, as wesleyqwen/chat.py does for Gemma.

        Each window sees the earlier ones as its own previous replies, the arrangement chat.py
        found keeps Gemma reacting to what is new in each part.
        """
        from wesleyqwen.chat import window_prompt
        duration, has_audio = yield from working(media.probe, path)
        spans = media.windows(duration, media.AUDIO_WINDOW_SECONDS)
        walked = turns[:-1]
        with tempfile.TemporaryDirectory(prefix="wesleyqwen-api-") as work:
            for part, (start, end) in enumerate(spans, 1):
                if kind == "video":
                    clip, wav = yield from working(media.cut, path, start, end, work, has_audio)
                else:
                    clip, wav = None, (yield from working(cut_audio, path, start, end, work))
                if len(spans) == 1:
                    prompt = question
                else:
                    yield f"{'' if part == 1 else chr(10) * 2}[{media.clock(start)}–{media.clock(end)}] "
                    prompt = (window_prompt(question, part, len(spans), start, end) if kind == "video" else
                              f"{question}\n\nThis is part {part} of {len(spans)} of the recording, "
                              f"{media.clock(start)} to {media.clock(end)}. React to what is new in this part.")
                content = [{"type": "video", "video": clip}] if clip else []
                content += [{"type": "text", "text": prompt}] + ([{"type": "audio", "audio": wav}] if wav else [])
                text = ""
                # Its own stop event: _generate sets the one it is given when its reply ends, and the
                # next window must still run. A caller hanging up closes this generator instead.
                for piece in self._generate(walked + [{"role": "user", "content": content}], req, Event(), cut_off):
                    text += piece or ""
                    yield piece
                walked = walked + [{"role": "user", "content": _in_words(content)},
                                   {"role": "assistant", "content": final_answer(text)}]

    def stream(self, req):
        """Yield the reply's text, thinking inside <think>...</think>; None means "still working",
        for a keep-alive; Finish ends a cut-off reply."""
        with self.lock:
            if self.unload_timer:
                self.unload_timer.cancel()
            stop, cut_off = Event(), []
            try:
                yield from working(self._load, req.model)
                timed = timed_media(req.turns[-1]) if self.family.hears_audio else None
                if timed:
                    yield from self._windows(req, req.turns, *timed, cut_off)
                else:
                    yield from self._generate(req.turns, req, stop, cut_off)
                if cut_off:
                    yield Finish("length")
            finally:
                stop.set()
                self.unload_timer = Timer(IDLE_UNLOAD_SECONDS, self._unload_if_idle)
                self.unload_timer.daemon = True
                self.unload_timer.start()


def working(fn, *args):
    """Run fn(*args) on a thread, yielding None every HEARTBEAT_SECONDS until it returns; then return its result.

    Loading a model or cutting a long video takes a minute, and a silent connection gets dropped.
    """
    done, box = Event(), {}

    def run():
        try:
            box["result"] = fn(*args)
        except Exception as error:  # re-raised below, on the caller's thread
            box["error"] = error
        finally:
            done.set()

    Thread(target=run, daemon=True).start()
    while not done.wait(HEARTBEAT_SECONDS):
        yield None
    if "error" in box:
        raise box["error"]
    return box.get("result")


def cut_audio(path, start, end, work):
    """A 16 kHz mono WAV of path from start to end: the rate Gemma's feature extractor reads."""
    wav = os.path.join(work, f"{start:09.3f}.wav")
    media.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", path, "-t", f"{end - start:.3f}",
               "-vn", "-ac", "1", "-ar", str(media.AUDIO_RATE), wav], f"cutting the audio of {path} at {media.clock(start)}")
    return wav


# --- the HTTP server ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerConfig:
    api_keys: set
    upload_key: str
    uploads: str
    models: list
    allowed_origins: set
    max_upload_bytes: int


def _error_body(status, message, code=None):
    type_ = "server_error" if status >= 500 else "invalid_request_error"
    return {"error": {"message": message, "type": type_, "code": code}}


def handler_for(engine, config):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[api] {self.command} {self.path.split('?')[0]} -> {args[1] if len(args) > 1 else ''}", flush=True)

        def send_json(self, body, status=HTTPStatus.OK, headers=None):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def fail(self, status, message, code=None, headers=None):
            self.send_json(_error_body(status, message, code), status, headers)

        def authorized(self):
            if bearer_key_is_valid(self.headers.get("Authorization"), config.api_keys):
                return True
            self.fail(401, "missing or invalid API key (send 'Authorization: Bearer <key>')", "invalid_api_key")
            return False

        def cors(self):
            origin = self.headers.get("Origin")
            if origin not in config.allowed_origins:
                return {}
            return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}

        def do_GET(self):
            if self.path == "/health":
                self.send_json({"status": "ok"})
            elif self.path == "/v1/models":
                if self.authorized():
                    reason = engine.busy_reason()
                    self.send_json({"object": "list", "data": [public_model(m) for m in config.models],
                                    "status": {"ready": reason is None, "reason": reason}})
            else:
                self.fail(404, f"no route GET {self.path}")

        def do_OPTIONS(self):
            cors = self.cors()
            if self.path != "/v1/uploads" or not cors:
                self.send_response(HTTPStatus.FORBIDDEN)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(HTTPStatus.NO_CONTENT)
            for name, value in {**cors, "Access-Control-Allow-Methods": "PUT",
                                "Access-Control-Allow-Headers": "Authorization, Content-Type",
                                "Access-Control-Max-Age": "600"}.items():
                self.send_header(name, value)
            self.end_headers()

        def do_PUT(self):
            if self.path != "/v1/uploads":
                return self.fail(404, f"no route PUT {self.path}")
            cors = self.cors()
            self.close_connection = True  # a refused upload's body is never read
            scheme, _, token = (self.headers.get("Authorization") or "").partition(" ")
            try:
                if scheme.lower() != "bearer":
                    raise UploadError("missing upload pass (send 'Authorization: Bearer <pass>')")
                grant = verify_upload_token(config.upload_key, token.strip(), now())
            except (UploadError, ValueError, KeyError) as error:
                return self.fail(401, str(error), "invalid_upload_pass", cors)
            content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if upload_kind(content_type) != grant["kind"]:
                return self.fail(400, f"this pass is for one {grant['kind']}, not {content_type or 'an untyped file'}",
                                 None, cors)
            if content_type not in EXTENSIONS:
                return self.fail(415, f"{content_type} is not a supported {grant['kind']} type", None, cors)
            length = int(self.headers.get("Content-Length") or -1)
            limit = min(grant["max"], config.max_upload_bytes)
            if length < 0:
                return self.fail(411, "send Content-Length", None, cors)
            if length > limit:
                return self.fail(413, f"file is {length} bytes; the limit is {limit}", "upload_too_large", cors)

            upload_id = uuid.uuid4().hex + EXTENSIONS[content_type]
            path = os.path.join(config.uploads, upload_id)
            remaining = length
            with open(path + ".part", "wb") as out:
                while remaining:
                    chunk = self.rfile.read(min(UPLOAD_CHUNK, remaining))
                    if not chunk:
                        break
                    out.write(chunk)
                    remaining -= len(chunk)
            if remaining:
                os.remove(path + ".part")
                return self.fail(400, f"upload ended {remaining} bytes early", None, cors)
            os.replace(path + ".part", path)
            prune_uploads(config.uploads, now())
            print(f"[api] upload {upload_id} {length} bytes", flush=True)
            self.close_connection = False
            self.send_json({"id": upload_id, "kind": grant["kind"], "bytes": length}, headers=cors)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.fail(404, f"no route POST {self.path}")
            if not self.authorized():
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            except ValueError:
                return self.fail(400, "request body is not valid JSON")
            try:
                req = parse_chat(body, config.models, config.uploads)
            except RequestError as error:
                return self.fail(error.status, error.message, "model_not_found" if error.status == 404 else None)
            if (reason := engine.busy_reason()) is not None:
                return self.fail(503, reason, "gpu_busy")
            if req.stream:
                self.stream_reply(req)
            else:
                self.whole_reply(req)

        def stream_reply(self, req):
            completion_id, created = f"chatcmpl-{uuid.uuid4().hex}", now()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True  # no Content-Length: the stream ends when the connection does
            splitter = ThinkRouter()
            pieces = engine.stream(req)

            def send(text):
                self.wfile.write(text.encode())
                self.wfile.flush()

            try:
                send(_chunk(completion_id, created, req.model["id"], {"role": "assistant"}))
                finish = "stop"
                try:
                    for piece in pieces:
                        if piece is None:
                            send(": keep-alive\n\n")
                            continue
                        if isinstance(piece, Finish):
                            finish = piece.reason
                            continue
                        for field, text in splitter.feed(piece):
                            send(_chunk(completion_id, created, req.model["id"], {field: text}))
                    for field, text in splitter.flush():
                        send(_chunk(completion_id, created, req.model["id"], {field: text}))
                    send(_chunk(completion_id, created, req.model["id"], {}, finish))
                except (BrokenPipeError, ConnectionResetError):
                    raise
                except Exception as error:
                    print(f"[api] reply failed: {error!r}", flush=True)
                    send("data: " + json.dumps(_error_body(500, f"generation failed: {error}")) + "\n\n")
                send("data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                print("[api] client hung up; generation stopped", flush=True)
            finally:
                pieces.close()

        def whole_reply(self, req):
            splitter, fields = ThinkRouter(), {"content": "", "reasoning_content": ""}
            finish = "stop"
            try:
                for piece in engine.stream(req):
                    if isinstance(piece, Finish):
                        finish = piece.reason
                    elif piece is not None:
                        for field, text in splitter.feed(piece):
                            fields[field] += text
            except Exception as error:
                print(f"[api] reply failed: {error!r}", flush=True)
                return self.fail(500, f"generation failed: {error}")
            for field, text in splitter.flush():
                fields[field] += text
            message = {"role": "assistant", "content": fields["content"]}
            if fields["reasoning_content"]:
                message["reasoning_content"] = fields["reasoning_content"]
            self.send_json({"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": now(),
                            "model": req.model["id"],
                            "choices": [{"index": 0, "message": message, "finish_reason": finish}]})

    return Handler


def make_server(engine, config, port, host="127.0.0.1"):
    return ThreadingHTTPServer((host, port), handler_for(engine, config))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--uploads", default="uploads", help="where browser uploads are kept (a week)")
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument("--max-upload-mb", type=int, default=500)
    parser.add_argument("--job-status", default=os.path.expanduser("~/jobs/status"))
    args = parser.parse_args()

    key = os.environ.get("WESLEYQWEN_API_KEY", "").strip()
    if not key:
        raise SystemExit("WESLEYQWEN_API_KEY is not set; it is the key MangumHub sends and signs upload passes with")
    origins = {o.strip() for o in os.environ.get("WESLEYQWEN_ALLOWED_ORIGINS", "").split(",") if o.strip()}
    os.makedirs(args.uploads, exist_ok=True)
    config = ServerConfig(api_keys={key}, upload_key=key, uploads=args.uploads,
                          models=load_registry(args.registry, os.environ), allowed_origins=origins,
                          max_upload_bytes=args.max_upload_mb << 20)
    server = make_server(Engine(args.job_status), config, args.port)
    print(f"[api] {len(config.models)} models on http://127.0.0.1:{args.port}; uploads from {sorted(origins)}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

