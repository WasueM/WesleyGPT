# Wesley wrote this
"""Separate a think model's reasoning from its answer.

WesleyGPT-Think was trained to reply "<think>\\n{work}\\n</think>\\n{answer}"
(wesleygpt/think.py). The tags are ordinary text, several tokens each, so while
streaming a tag can arrive split across events: any tail that could still grow
into the next tag is held back until the next event settles it.

The reasoning goes out as OpenAI-style `reasoning_content` (DeepSeek's
convention, which OpenAI client libraries pass through), and is left out of
the prompt on later turns: the model never saw earlier reasoning in context
during training, and it only has a 2,048-token window.
"""
import re
from dataclasses import replace

OPEN, CLOSE = "<think>", "</think>"
_BLOCK = re.compile(re.escape(OPEN) + r".*?(?:" + re.escape(CLOSE) + r"\n?|\Z)", re.DOTALL)


def strip_reasoning(text):
    """`text` without its <think> block (an unterminated one runs to the end)."""
    return _BLOCK.sub("", text)


def _held_back(text, marker):
    """Length of the longest tail of `text` that is a proper prefix of `marker`."""
    for n in range(min(len(text), len(marker) - 1), 0, -1):
        if marker.startswith(text[-n:]):
            return n
    return 0


class _Splitter:
    def __init__(self):
        self.stage = "before"  # before -> thinking -> after
        self.pending = ""
        self.drop_newline = False  # the "\n" right after each tag is formatting, not text

    def _emit(self, text):
        if self.drop_newline and text:
            text, self.drop_newline = text.removeprefix("\n"), False
        return text

    def feed(self, text, final):
        """(reasoning, content) that can be released after seeing `text`."""
        self.pending += text
        reasoning = content = ""
        if self.stage == "before":
            i = self.pending.find(OPEN)
            if i >= 0:
                content += self.pending[:i]
                self.pending, self.stage, self.drop_newline = self.pending[i + len(OPEN):], "thinking", True
        if self.stage == "thinking":
            i = self.pending.find(CLOSE)
            if i >= 0:
                reasoning += self._emit(self.pending[:i].removesuffix("\n"))
                self.pending, self.stage, self.drop_newline = self.pending[i + len(CLOSE):], "after", True
            else:
                keep = 0 if final else _held_back(self.pending, "\n" + CLOSE)
                reasoning += self._emit(self.pending[:len(self.pending) - keep])
                self.pending = self.pending[len(self.pending) - keep:]
        if self.stage != "thinking":
            keep = 0 if final or self.stage == "after" else _held_back(self.pending, OPEN)
            content += self._emit(self.pending[:len(self.pending) - keep])
            self.pending = self.pending[len(self.pending) - keep:]
        return reasoning, content


def split_reasoning(events):
    """DecodeEvents with the <think> block moved from `text` to `reasoning`."""
    splitter = _Splitter()
    for ev in events:
        reasoning, content = splitter.feed(ev.text, final=ev.finish_reason is not None)
        yield replace(ev, text=content, reasoning=reasoning)
