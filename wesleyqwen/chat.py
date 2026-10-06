# Wesley wrote this
"""Chat with the base model and each WesleyQwen variant, swapping between them mid-conversation.

    python -m wesleyqwen.chat               # starts on the base model
    python -m wesleyqwen.chat --model full

Commands: /model base|full|lora|qlora, /think (toggle thinking), /clear, /quit.
The conversation carries over when you swap, so the same follow-up can go to
every model. Only one model sits on the GPU at a time: two of them would not fit
next to each other's generation memory on 12 GB.
"""
import argparse
import gc
import os

import torch

from wesleyqwen.evaluate import NON_THINKING, THINKING, stop_token_ids
from wesleyqwen.scoring import final_answer
from wesleyqwen.train import BASE

VARIANTS = ("full", "lora", "qlora")
MAX_NEW_TOKENS = {False: 1024, True: 4096}
HELP = "Commands: /model base|full|lora|qlora, /think (toggle thinking), /clear, /quit"


def parse_command(line):
    """('model', 'full') for '/model full'; None for a message to the model."""
    line = line.strip()
    if not line.startswith("/"):
        return None
    name, _, arg = line[1:].partition(" ")
    return name, arg.strip()


def model_paths(runs):
    return {"base": BASE, **{v: os.path.join(runs, v) for v in VARIANTS}}


class Session:
    def __init__(self, paths):
        self.paths = paths
        self.name = self.model = self.tokenizer = None

    def load(self, name):
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        if name not in self.paths:
            raise ValueError(f"unknown model {name!r}; choose from {', '.join(self.paths)}")
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[loading {name} from {self.paths[name]}]", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.paths[name])
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.paths[name], dtype=torch.bfloat16, device_map="cuda").eval()
        self.name = name

    def reply(self, history, thinking):
        from transformers import TextStreamer
        text = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=thinking)
        batch = self.tokenizer(text, return_tensors="pt").to("cuda")
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            out = self.model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS[thinking], streamer=streamer,
                                      eos_token_id=stop_token_ids(self.tokenizer),
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      **(THINKING if thinking else NON_THINKING))
        answer = self.tokenizer.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        return final_answer(("<think>\n" if thinking else "") + answer)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="base", choices=("base",) + VARIANTS)
    parser.add_argument("--runs", default="runs/wesleyqwen", help="directory holding the trained variants")
    args = parser.parse_args()

    session = Session(model_paths(args.runs))
    session.load(args.model)
    history, thinking = [], False
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
            history.append({"role": "user", "content": line})
            print(f"{session.name}> ", end="", flush=True)
            history.append({"role": "assistant", "content": session.reply(history, thinking)})
        elif command[0] == "model":
            try:
                session.load(command[1])
            except ValueError as error:
                print(error)
        elif command[0] == "think":
            thinking = not thinking
            print(f"[thinking {'on' if thinking else 'off'}]")
        elif command[0] == "clear":
            history = []
            print("[conversation cleared]")
        elif command[0] in ("quit", "exit"):
            return
        else:
            print(f"unknown command /{command[0]}. {HELP}")


if __name__ == "__main__":
    main()
