# Wesley wrote this
"""Think-format math for WesleyGPT-Think (format in wesleygpt/think.py).

MetaMathThink: the 240K grade-school-math rows of MetaMathQA (MIT), rephrasings
and back-solved variants of GSM8K/MATH *train* questions, so the GSM8K test set
stays unseen. The competition-math rows are left out: their answers are LaTeX,
which the numeric GSM8K grader cannot score.
GSM8KThink: GSM8K's own training rows in the same shape, calculator calls kept.
"""
import pyarrow.compute as pc

from tasks.common import Task, load_hub_dataset
from tasks.gsm8k import GSM8K
from wesleygpt.think import metamath_to_conversation, thinkify_gsm8k


class MetaMathThink(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        table = load_hub_dataset("meta-math/MetaMathQA").table
        table = table.filter(pc.starts_with(table["type"], "GSM"))
        rows = zip(table["query"].to_pylist(), table["response"].to_pylist())
        self.conversations = [c for c in (metamath_to_conversation(q, r) for q, r in rows) if c is not None]

    def num_examples(self):
        return len(self.conversations)

    def get_example(self, index):
        return self.conversations[index]


class GSM8KThink(GSM8K):
    def get_example(self, index):
        return thinkify_gsm8k(super().get_example(index))
