# Wesley wrote this
"""Chat with base Qwen3.5-2B, each WesleyQwen variant and Gemma 4 E2B, swapping between them mid-conversation.

    python -m wesleyqwen.chat               # starts on the base model
    python -m wesleyqwen.chat --model gemma

Commands: /model base|full|lora|qlora|gemma, /think (toggle thinking), /video <path> [question],
/clear, /quit. The conversation carries over when you swap, so the same follow-up can go to
every model. Only one model sits on the GPU at a time: two of them would not fit next to
each other's generation memory on 12 GB.

The Qwen models see a video's frames, and the video stays in the conversation, so every later
turn re-reads it. Gemma also hears the soundtrack, but at most 30 s of it per clip, so it walks
a video in windows of up to 30 s and reacts to each in turn; afterwards only the text of what
it said stays in the conversation.
"""
import argparse
import gc
import os
import shlex
import tempfile
from threading import Thread

import torch

from wesleyqwen import media
from wesleyqwen.models import MODEL_NAMES, GemmaChannels, model_registry
from wesleyqwen.scoring import final_answer

MAX_NEW_TOKENS = {False: 1024, True: 4096}
# Gemma 4 E2B hears at most 30 s of audio per clip (its model card).
WINDOW_SECONDS = 30
DEFAULT_VIDEO_QUESTION = "Describe what happens in this video."
HELP = (f"Commands: /model {'|'.join(MODEL_NAMES)}, /think (toggle thinking), /video <path> [question], "
        "/clear, /quit")


def parse_command(line):
    """('model', 'full') for '/model full'; None for a message to the model."""
    line = line.strip()
    if not line.startswith("/"):
        return None
    name, _, arg = line[1:].partition(" ")
    return name, arg.strip()


def parse_video(arg):
    """('/path/clip.mp4', 'question') for '/video' arguments; quote a path that has spaces."""
    words = shlex.split(arg)
    if not words:
        raise ValueError("usage: /video <path> [question]")
    return words[0], " ".join(words[1:]) or DEFAULT_VIDEO_QUESTION


def video_turn(path, question):
    """A user turn that shows the model the video, then asks the question."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no video file at {path}")
    return {"role": "user", "content": [{"type": "video", "video": path}, {"type": "text", "text": question}]}


def video_of(turn):
    """(path, question) if turn is a video turn, else None."""
    if isinstance(turn["content"], str):
        return None
    parts = {part["type"]: part for part in turn["content"]}
    return (parts["video"]["video"], parts["text"]["text"]) if "video" in parts else None


def require_seeking_video_decoder():
    """Refuse video unless transformers will decode it with torchcodec.

    Its only fallback, torchvision, decodes every frame of the file into RAM before picking the
    few it samples: a 2-minute 1080p clip is ~20 GB, which surfaced as a swscaler "Resource
    temporarily unavailable" error in the web chat and an OOM kill in a fresh process.
    """
    from transformers import video_processing_utils
    if not video_processing_utils.is_torchcodec_available():
        raise RuntimeError("video needs torchcodec, which seeks to the sampled frames; install it with "
                           "`sudo apt install ffmpeg` and `uv pip install torchcodec==0.8.1` (see wesley/README.md)")


def text_only(history):
    """The history with each media part replaced by a mention of it, for a model that must only see new media."""
    def flatten(content):
        if isinstance(content, str):
            return content
        return " ".join(part["text"] if part["type"] == "text" else f"[{part['type']}]" for part in content)
    return [{"role": turn["role"], "content": flatten(turn["content"])} for turn in history]


def window_prompt(question, part, parts, start, end):
    return (f"{question}\n\nThis is part {part} of {parts} of the video, {media.clock(start)} to {media.clock(end)}: "
            "its frames, then its audio. React to what is new in this part.")


def window_turn(clip, prompt, wav):
    """Frames before the question and sound after it: the order Gemma's model card asks for."""
    content = [{"type": "video", "video": clip}, {"type": "text", "text": prompt}]
    if wav:
        content.append({"type": "audio", "audio": wav})
    return {"role": "user", "content": content}


def windowed_reply(stream, cut, history, path, question, duration, thinking, answer):
    """Yield a reply to a video one window at a time; afterwards history holds a text record of it.

    Each window sees the earlier ones as its own previous replies in the conversation. Pasting
    them into the question instead made Gemma read them as the user's analysis ("your
    breakdown"), and it stopped reacting to the new part.

    cut(path, start, end) -> (clip, wav or None); stream(messages, thinking) yields text;
    answer(text) is what of a window's reply is kept.
    """
    spans = media.windows(duration, WINDOW_SECONDS)
    walked, said = text_only(history), []
    for part, (start, end) in enumerate(spans, 1):
        clip, wav = cut(path, start, end)
        label = f"{media.clock(start)}–{media.clock(end)}{'' if wav else ', no audio track'}"
        yield f"{'' if part == 1 else chr(10) * 2}[{label}] "
        turn = window_turn(clip, window_prompt(question, part, len(spans), start, end), wav)
        text = ""
        for piece in stream(walked + [turn], thinking):
            text += piece
            yield piece
        said.append((start, end, answer(text)))
        walked += text_only([turn]) + [{"role": "assistant", "content": said[-1][2]}]
    history.append({"role": "user", "content": f"[video {os.path.basename(path)}, {media.clock(duration)}, "
                                                f"watched and heard in {len(spans)} parts] {question}"})
    history.append({"role": "assistant", "content": "\n\n".join(
        f"[{media.clock(a)}–{media.clock(b)}] {reply}" for a, b, reply in said)})


def answer_of(text):
    """What goes back into the history: the streamed text without its thinking."""
    return final_answer(text)


class Session:
    def __init__(self, models):
        self.models = models
        self.name = self.family = self.model = self.tokenizer = self.processor = None

    def load(self, name):
        import transformers
        if name not in self.models:
            raise ValueError(f"unknown model {name!r}; choose from {', '.join(self.models)}")
        path, family = self.models[name]
        self.name = self.family = self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[loading {name} from {path}]", flush=True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        self.processor = transformers.AutoProcessor.from_pretrained(family.processor_from or path,
                                                                    tokenizer=self.tokenizer)
        self.model = getattr(transformers, family.loader).from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda").eval()
        self.name, self.family = name, family

    def stream(self, history, thinking):
        """Yield the reply's text as it is generated, thinking (if any) inside <think>...</think>."""
        from transformers import TextIteratorStreamer
        family = self.family
        batch = self.processor.apply_chat_template(history, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt",
                                                   enable_thinking=thinking, **family.video_kwargs).to("cuda")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=not family.thinking_in_special_tokens)
        generate_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS[thinking], streamer=streamer,
                               pad_token_id=self.tokenizer.pad_token_id, **family.decoding[thinking])
        if family.stop_ids:
            generate_kwargs["eos_token_id"] = family.stop_ids(self.tokenizer)
        failure = []

        def generate():
            try:
                with torch.inference_mode():
                    self.model.generate(**batch, **generate_kwargs)
            except Exception as error:  # re-raised below; ending the streamer keeps the reader from hanging
                failure.append(error)
                streamer.end()

        worker = Thread(target=generate)
        worker.start()
        if thinking and family.thinking_opens_in_prompt:
            yield "<think>\n"
        channels = GemmaChannels() if family.thinking_in_special_tokens else None
        for piece in streamer:
            yield channels.feed(piece) if channels else piece
        if channels:
            yield channels.flush()
        worker.join()
        if failure:
            raise failure[0]

    def respond(self, history, turn, thinking):
        """Yield the reply to turn; once it is complete, history records the exchange."""
        video = video_of(turn)
        if video and self.family.hears_audio:
            path, question = video
            duration, has_audio = media.probe(path)
            with tempfile.TemporaryDirectory(prefix="wesleyqwen-windows-") as work:
                yield from windowed_reply(self.stream, lambda p, a, b: media.cut(p, a, b, work, has_audio),
                                          history, path, question, duration, thinking, answer_of)
            return
        context = text_only(history) if self.family.hears_audio else history
        text = ""
        for piece in self.stream(context + [turn], thinking):
            text += piece
            yield piece
        history += [turn, {"role": "assistant", "content": answer_of(text)}]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="base", choices=MODEL_NAMES)
    parser.add_argument("--runs", default="runs/wesleyqwen", help="directory holding the trained variants")
    args = parser.parse_args()

    session = Session(model_registry(args.runs))
    session.load(args.model)
    history, thinking = [], False

    def respond(turn):
        print(f"{session.name}> ", end="", flush=True)
        for piece in session.respond(history, turn, thinking):
            print(piece, end="", flush=True)
        print()

    print(HELP)
    while True:
        try:
            line = input(f"\n[{session.name}{' +think' if thinking else ''}] you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line.strip():
            continue
        command = parse_command(line)
        if command is None:
            respond({"role": "user", "content": line})
        elif command[0] == "model":
            try:
                session.load(command[1])
            except ValueError as error:
                print(error)
        elif command[0] == "think":
            thinking = not thinking
            print(f"[thinking {'on' if thinking else 'off'}]")
        elif command[0] == "video":
            try:
                require_seeking_video_decoder()
                turn = video_turn(*parse_video(command[1]))
            except (ValueError, FileNotFoundError, RuntimeError) as error:
                print(error)
                continue
            respond(turn)
        elif command[0] == "clear":
            history.clear()
            print("[conversation cleared]")
        elif command[0] in ("quit", "exit"):
            return
        else:
            print(f"unknown command /{command[0]}. {HELP}")


if __name__ == "__main__":
    main()
