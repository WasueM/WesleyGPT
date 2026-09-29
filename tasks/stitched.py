# Wesley wrote this
"""Long multi-topic conversations for Think-LongContext (stitching: wesleygpt/stitch.py).

Each row strings whole exchanges from the other SFT datasets together -- a
SmolTalk chat, a think-format math problem, an MMLU question, an identity
question -- switching dataset every exchange, up to a token target drawn
between `min_tokens` and `max_tokens` so rows reach deep into the 2,048-token
window. Mixing math (which thinks) with chat and multiple choice (which don't)
in one conversation also shows the model, turn by turn, when to think.

With `dependent` > 0 (Think-LongContext v2), that fraction of rows also gets one
to three turns whose answer is only in the history -- a fact stated earlier, the
last math answer, a number to compute with (wesleygpt/dependent.py) -- so the
model learns to use earlier turns instead of learning they are safe to ignore.

Rows are built on demand from the index, so the same seed gives the same rows.
"""
import random

from tasks.common import Task
from wesleygpt.dependent import insert_dependencies
from wesleygpt.stitch import stitch_segments

DEPENDENCY_TOKENS = 80  # room left per dependent turn; a plant plus its question rarely needs more


class StitchedChats(Task):
    def __init__(self, sources, size, seed, tokenizer, min_tokens=1024, max_tokens=2040, dependent=0.0, **kwargs):
        """sources: (name, task, weight) triples. Sharing a name (two math tasks) keeps
        them from following each other. dependent: fraction of rows given dependent turns."""
        super().__init__(**kwargs)
        self.names = [name for name, _, _ in sources]
        self.tasks = [task for _, task, _ in sources]
        self.weights = [weight for _, _, weight in sources]
        self.size, self.seed, self.tok = size, seed, tokenizer
        self.min_tokens, self.max_tokens = min_tokens, max_tokens
        self.dependent = dependent

    def num_examples(self):
        return self.size

    def _draw(self, rng):
        i = rng.choices(range(len(self.tasks)), weights=self.weights)[0]
        task = self.tasks[i]
        return self.names[i], task[rng.randrange(len(task))]["messages"]

    def _count(self, messages):
        return len(self.tok.render_conversation({"messages": messages}, max_tokens=10**9)[0]) - 1

    def get_example(self, index):
        rng = random.Random(self.seed * 1_000_003 + index)
        target = rng.randint(self.min_tokens, self.max_tokens)
        n = rng.randint(1, 3) if self.dependent and rng.random() < self.dependent else 0
        segments = []
        while not segments:
            segments = stitch_segments(self._draw, self._count, target - n * DEPENDENCY_TOKENS, rng)
        for tries in range(n, 0, -1):  # fewer dependent turns if the reserve wasn't enough
            with_dependencies = insert_dependencies(segments, random.Random(rng.random()), tries)
            if self._count(with_dependencies) + 1 <= self.max_tokens:
                return {"messages": with_dependencies}
        return {"messages": [m for segment in segments for m in segment]}
