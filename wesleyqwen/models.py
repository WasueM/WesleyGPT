# Wesley wrote this
"""The models the chats can load, and what differs between their two families.

Qwen3.5-2B (base) and its fine-tunes see video as frames only. Gemma 4 E2B also hears
audio, samples its own frames, and marks its thinking with special tokens.
"""
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from wesleyqwen.evaluate import NON_THINKING, THINKING, stop_token_ids
from wesleyqwen.train import BASE

VARIANTS = ("full", "lora", "qlora")
GEMMA = os.environ.get("WESLEYQWEN_GEMMA", "google/gemma-4-E2B-it")
MODEL_NAMES = ("base",) + VARIANTS + ("gemma",)
# Gemma's model card: the same sampling for every use, thinking or not.
GEMMA_SAMPLING = {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64}


@dataclass(frozen=True)
class Family:
    name: str
    loader: str                       # the transformers Auto class that builds the model
    processor_from: Optional[str]     # where the processor comes from; None = the model's own directory
    decoding: dict                    # {thinking: sampling kwargs}
    video_kwargs: dict                # extra apply_chat_template arguments for video
    stop_ids: Optional[Callable]      # tokenizer -> eos ids; None = the model's generation_config
    hears_audio: bool                 # takes a video's soundtrack beside its frames
    thinking_opens_in_prompt: bool    # the chat template writes the opening <think> itself
    thinking_in_special_tokens: bool  # thinking is marked by special tokens the streamer would drop


QWEN_FAMILY = Family(
    name="qwen", loader="AutoModelForImageTextToText",
    # Fine-tuning saved no image/video preprocessor config; it never changed, so read the base one.
    processor_from=BASE, decoding={False: NON_THINKING, True: THINKING},
    # Two frames a second reads an 8 s clip as ~1.8k prompt tokens; cost grows with both.
    video_kwargs={"fps": 2}, stop_ids=stop_token_ids, hears_audio=False,
    thinking_opens_in_prompt=True, thinking_in_special_tokens=False)

GEMMA_FAMILY = Family(
    name="gemma", loader="AutoModelForMultimodalLM", processor_from=None,
    decoding={False: GEMMA_SAMPLING, True: GEMMA_SAMPLING},
    # Gemma samples a fixed 32 frames per clip, and refuses fps alongside that.
    video_kwargs={}, stop_ids=None, hears_audio=True,
    thinking_opens_in_prompt=False, thinking_in_special_tokens=True)


def model_registry(runs):
    """{name: (path, family)} for every model the chats can switch to."""
    return {"base": (BASE, QWEN_FAMILY), **{v: (os.path.join(runs, v), QWEN_FAMILY) for v in VARIANTS},
            "gemma": (GEMMA, GEMMA_FAMILY)}


class GemmaChannels:
    """Rewrite Gemma 4's thinking channel into the <think>...</think> the chats already parse.

    Gemma writes <|channel>thought\\n...<channel|>answer<turn|>. The markers are single
    special tokens, so a streamer never splits one across pieces, but the "thought"
    label after the opening marker is ordinary text and can arrive in fragments.
    """
    MARKERS = re.compile(r"(<\|channel>|<channel\|>|<turn\|>|<eos>)")
    LABEL = "thought\n"

    def __init__(self):
        self.held = None  # text after an opening marker that may still turn out to be the label

    def feed(self, piece):
        out = ""
        for part in self.MARKERS.split(piece):
            if part == "<|channel>":
                out += self.flush() + "<think>\n"
                self.held = ""
            elif part == "<channel|>":
                out += self.flush() + "</think>"
            elif part in ("<turn|>", "<eos>"):
                continue
            elif self.held is not None:
                self.held += part
                if self.held.startswith(self.LABEL):
                    out += self.held[len(self.LABEL):]
                    self.held = None
                elif not self.LABEL.startswith(self.held):
                    out += self.flush()
            else:
                out += part
        return out

    def flush(self):
        """Whatever was held back as a possible label, now that it cannot be one."""
        held, self.held = self.held or "", None
        return held
