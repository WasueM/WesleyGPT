# Wesley wrote this
"""WesleyGPT identity conversations, so the chat model knows its own name.

Generated deterministically by wesleygpt.identity (no download). Mix a few
epochs into SFT with `scripts.chat_sft --identity-epochs N`.
"""
from tasks.common import Task
from wesleygpt.identity import make_conversations

SPLITS = {"train": (1000, 0), "val": (100, 1)}  # split -> (size, seed)


class WesleyIdentity(Task):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in SPLITS, f"WesleyIdentity split must be one of {sorted(SPLITS)}"
        size, seed = SPLITS[split]
        self.conversations = make_conversations(size, seed=seed)

    def num_examples(self):
        return len(self.conversations)

    def get_example(self, index):
        return self.conversations[index]
