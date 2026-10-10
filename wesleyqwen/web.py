# Wesley wrote this
"""A browser chat for the same models as wesleyqwen.chat (the Qwen models and Gemma), with a video file picker.

    python -m wesleyqwen.web                # serves http://127.0.0.1:7860 inside WSL

It binds to localhost only; reach it from another machine through an SSH tunnel
(ssh -L 7860:localhost:7860 ...). Videos picked in the browser are uploaded into
--uploads and the model reads them from there. One conversation, one generation at
a time: the 12 GB GPU holds one model and one reply.
"""
import argparse
import json
import os
import re
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from urllib.parse import parse_qs, urlparse

from wesleyqwen.chat import DEFAULT_VIDEO_QUESTION, Session, require_seeking_video_decoder, video_turn
from wesleyqwen.models import MODEL_NAMES, model_registry

PAGE = os.path.join(os.path.dirname(__file__), "web.html")
UPLOAD_CHUNK = 1 << 20


def upload_name(original):
    """A unique, shell-safe file name for an upload, keeping the original's base name readable."""
    base = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(original)) or "video"
    return f"{uuid.uuid4().hex[:8]}-{base}"


def uploaded_video(folder, video_id):
    """The path of a video previously uploaded into folder; nothing outside it."""
    folder = os.path.realpath(folder)
    path = os.path.realpath(os.path.join(folder, video_id))
    if os.path.dirname(path) != folder:
        raise ValueError(f"{video_id!r} is not an uploaded video")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no uploaded video named {video_id}")
    return path


class Chat:
    """The one conversation the page drives, and the lock that keeps replies one at a time."""

    def __init__(self, session, uploads):
        self.session, self.uploads = session, uploads
        self.history, self.thinking = [], False
        self.busy = Lock()

    def state(self):
        return {"model": self.session.name, "models": list(self.session.models), "thinking": self.thinking,
                "turns": len(self.history)}


def handler_for(chat):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[web] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)

        def send_json(self, body, status=HTTPStatus.OK):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_json(self):
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def do_GET(self):
            if self.path == "/":
                with open(PAGE, "rb") as page:
                    data = page.read()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/state":
                self.send_json(chat.state())
            else:
                self.send_json({"error": f"no page {self.path}"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            route = urlparse(self.path)
            if route.path == "/upload":
                self.upload(parse_qs(route.query).get("name", ["video"])[0])
            elif route.path == "/chat":
                self.reply(self.read_json())
            elif route.path in ("/model", "/think", "/clear"):
                self.setting(route.path[1:], self.read_json())
            else:
                self.send_json({"error": f"no endpoint {route.path}"}, HTTPStatus.NOT_FOUND)

        def upload(self, original):
            name = upload_name(original)
            remaining = int(self.headers["Content-Length"])
            with open(os.path.join(chat.uploads, name), "wb") as out:
                while remaining:
                    chunk = self.rfile.read(min(UPLOAD_CHUNK, remaining))
                    if not chunk:
                        raise ConnectionError(f"upload of {original} ended {remaining} bytes early")
                    out.write(chunk)
                    remaining -= len(chunk)
            self.send_json({"video": name})

        def setting(self, name, body):
            if not chat.busy.acquire(blocking=False):
                return self.send_json({"error": "the model is still replying"}, HTTPStatus.CONFLICT)
            try:
                if name == "model":
                    chat.session.load(body["name"])
                elif name == "think":
                    chat.thinking = bool(body["on"])
                else:
                    chat.history = []
            except ValueError as error:
                return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            finally:
                chat.busy.release()
            self.send_json(chat.state())

        def reply(self, body):
            text = body.get("text", "").strip()
            try:
                if body.get("video"):
                    require_seeking_video_decoder()
                    turn = video_turn(uploaded_video(chat.uploads, body["video"]), text or DEFAULT_VIDEO_QUESTION)
                elif text:
                    turn = {"role": "user", "content": text}
                else:
                    raise ValueError("send a message or a video")
            except (ValueError, FileNotFoundError, RuntimeError) as error:
                return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            if not chat.busy.acquire(blocking=False):
                return self.send_json({"error": "the model is still replying"}, HTTPStatus.CONFLICT)
            try:
                # No Content-Length: the reply streams until the connection closes.
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    for piece in chat.session.respond(chat.history, turn, chat.thinking):
                        self.wfile.write(piece.encode())
                        self.wfile.flush()
                except Exception as error:
                    print(f"[web] reply failed: {error!r}", flush=True)
                    self.wfile.write(f"\n\n[error: {error}]".encode())
            finally:
                chat.busy.release()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="base", choices=MODEL_NAMES)
    parser.add_argument("--runs", default="runs/wesleyqwen", help="directory holding the trained variants")
    parser.add_argument("--uploads", default="uploads", help="where videos picked in the browser are saved")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    os.makedirs(args.uploads, exist_ok=True)
    session = Session(model_registry(args.runs))
    session.load(args.model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(Chat(session, args.uploads)))
    print(f"[web] serving on http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
