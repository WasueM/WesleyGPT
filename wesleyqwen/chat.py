# Wesley wrote this
"""Chat with the base model and each WesleyQwen variant, swapping between them mid-conversation.

    python -m wesleyqwen.chat               # starts on the base model
    python -m wesleyqwen.chat --model full

Commands: /model base|full|lora|qlora, /think (toggle thinking), /video <path> [question],
/clear, /quit. The conversation carries over when you swap, so the same follow-up can go to
every model. Only one model sits on the GPU at a time: two of them would not fit next to
each other's generation memory on 12 GB.

A video stays in the conversation, so every later turn re-reads its frames: follow-ups can
ask about it, at the cost of the video's prompt tokens on each turn. /clear drops it.
"""
import argparse
import gc
import os
import shlex
from threading import Thread

import torch

from wesleyqwen.evaluate import NON_THINKING, THINKING, stop_token_ids
from wesleyqwen.scoring import final_answer
from wesleyqwen.train import BASE

VARIANTS = ("full", "lora", "qlora")
MAX_NEW_TOKENS = {False: 1024, True: 4096}
# Two frames a second reads an 8 s clip as ~1.8k prompt tokens; prompt cost grows linearly with both.
VIDEO_FPS = 2
DEFAULT_VIDEO_QUESTION = "Describe what happens in this video."
HELP = "Commands: /model base|full|lora|qlora, /think (toggle thinking), /video <path> [question], /clear, /quit"


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


def model_paths(runs):
    return {"base": BASE, **{v: os.path.join(runs, v) for v in VARIANTS}}


class Session:
    def __init__(self, paths):
        self.paths = paths
        self.name = self.model = self.tokenizer = self.processor = None

    def load(self, name):
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
        if name not in self.paths:
            raise ValueError(f"unknown model {name!r}; choose from {', '.join(self.paths)}")
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[loading {name} from {self.paths[name]}]", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.paths[name])
        # Fine-tuning saved no image/video preprocessor config; it never changed, so read the base one.
        self.processor = AutoProcessor.from_pretrained(BASE, tokenizer=self.tokenizer)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.paths[name], dtype=torch.bfloat16, device_map="cuda").eval()
        self.name = name

    def stream(self, history, thinking):
        """Yield the reply's text as it is generated."""
        from transformers import TextIteratorStreamer
        batch = self.processor.apply_chat_template(history, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt",
                                                   enable_thinking=thinking, fps=VIDEO_FPS).to("cuda")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        failure = []

        def generate():
            try:
                with torch.inference_mode():
                    self.model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS[thinking], streamer=streamer,
                                        eos_token_id=stop_token_ids(self.tokenizer),
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        **(THINKING if thinking else NON_THINKING))
            except Exception as error:  # re-raised below; ending the streamer keeps the reader from hanging
                failure.append(error)
                streamer.end()

        worker = Thread(target=generate)
        worker.start()
        yield from streamer
        worker.join()
        if failure:
            raise failure[0]


def answer_of(text, thinking):
    """What goes back into the history: the streamed text without its thinking."""
    return final_answer(("<think>\n" if thinking else "") + text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="base", choices=("base",) + VARIANTS)
    parser.add_argument("--runs", default="runs/wesleyqwen", help="directory holding the trained variants")
    args = parser.parse_args()

    session = Session(model_paths(args.runs))
    session.load(args.model)
    history, thinking = [], False

    def respond(turn):
        history.append(turn)
        print(f"{session.name}> ", end="", flush=True)
        text = ""
        for piece in session.stream(history, thinking):
            print(piece, end="", flush=True)
            text += piece
        print()
        history.append({"role": "assistant", "content": answer_of(text, thinking)})

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
            history = []
            print("[conversation cleared]")
        elif command[0] in ("quit", "exit"):
            return
        else:
            print(f"unknown command /{command[0]}. {HELP}")


if __name__ == "__main__":
    main()
