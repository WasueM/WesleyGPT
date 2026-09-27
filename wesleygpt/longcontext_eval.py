# Wesley wrote this
"""How well does a chat model hold up deep into a long conversation?

    python -m wesleygpt.longcontext_eval --source sft --model-tag d12-think --step 1038 --device cuda

Three measures, each reported by how much conversation came before the question
("depth", in tokens of earlier turns):

  late_math  GSM8K test problems asked after 0-1,300 tokens of unrelated
             earlier turns. Same grader as chat_eval's GSM8K.
  recall     a fact stated in turn 1 ("my dog's name is Biscuit"), unrelated
             turns, then the question. Nothing like this is in any training mix,
             so it measures whether long-conversation practice carries over to
             using the history.
  loss       the model's loss on held-out stitched conversations, by position
             in the 2,048-token window: does it degrade late in the window?
             Stitched conversations are the training format of
             Think-LongContext, so this one favours it.

The earlier turns are real exchanges from held-out splits (SmolTalk, MMLU and
GSM8K test, identity val), stitched by wesleygpt.stitch with the dataset's own
answers as history, and seeded by item, so every model sees the same
conversations. Generation goes through the serving runtime greedily, so the
prompt is rendered exactly as the API renders it.
"""
import argparse
import json
import random
import re

SEQ_LEN = 2048
DEPTHS = (0, 400, 800, 1300)
ACK = "Got it, I'll remember that."

PETS = ["Biscuit", "Pepper", "Waffles", "Juniper", "Mochi", "Pickles", "Rusty", "Clover", "Nugget", "Ziggy"]
PEOPLE = ["Marisol", "Everett", "Priya", "Tobias", "Ingrid", "Desmond", "Harriet", "Lucian", "Beatrix", "Oswin"]
CITIES = ["Tucson", "Halifax", "Bergen", "Valparaiso", "Adelaide", "Kraków", "Nagoya", "Spokane", "Porto", "Dunedin"]
COLORS = ["teal", "maroon", "lavender", "mustard", "turquoise", "crimson", "olive", "periwinkle"]
FACTS = [  # (statement, question, values)
    ("my dog's name is {v}", "What's my dog's name?", PETS),
    ("my sister is called {v}", "What is my sister called?", PEOPLE),
    ("I live in {v}", "Which city do I live in?", CITIES),
    ("my favorite color is {v}", "What's my favorite color?", COLORS),
    ("my favorite number is {v}", "What's my favorite number?", None),
    ("my locker code is {v}", "What's my locker code?", None),
]


def score_recall(answer, value):
    """True if `answer` contains `value` as a whole word (so 42 does not match 420)."""
    return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", answer, re.IGNORECASE) is not None


def make_recall_items(n, seed):
    """`n` facts to plant and ask back, identical for the same seed."""
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        statement, question, values = rng.choice(FACTS)
        if values is not None:
            value = rng.choice(values)
        else:
            value = str(rng.randint(1000, 9999) if "code" in statement else rng.randint(13, 999))
        items.append({
            "plant": f"Before we start, one thing to remember: {statement.format(v=value)}.",
            "question": question,
            "value": value,
        })
    return items


def recall_conversation(item, filler):
    """Plant the fact, then the unrelated turns, then ask for it."""
    return [{"role": "user", "content": item["plant"]}, {"role": "assistant", "content": ACK},
            *filler, {"role": "user", "content": item["question"]}]


def summarize_by_depth(rows):
    """{depth: accuracy and mean user turns} over (depth, user_turns, correct) rows."""
    by_depth = {}
    for depth, turns, correct in rows:
        by_depth.setdefault(depth, []).append((turns, correct))
    return {d: {"n": len(v), "accuracy": sum(c for _, c in v) / len(v), "user_turns": sum(t for t, _ in v) / len(v)}
            for d, v in sorted(by_depth.items())}


def _user_turns(messages):
    return sum(m["role"] == "user" for m in messages)


class _Harness:
    """Held-out filler turns and the serving runtime for one model."""

    def __init__(self, args):
        # Imported here so the pure helpers above stay importable without torch or datasets.
        from nanochat.tokenizer import get_tokenizer
        from tasks.mmlu import MMLU
        from tasks.smoltalk import SmolTalk
        from tasks.wesley_identity import WesleyIdentity
        from wesleygpt.runtime import NanochatRuntime
        self.tok = get_tokenizer()
        self.fillers = [("smol", SmolTalk(split="test")), ("mmlu", MMLU(subset="all", split="test")),
                        ("identity", WesleyIdentity(split="val"))]
        spec = {"id": "eval", "description": "", "source": args.source, "model_tag": args.model_tag,
                "step": args.step, "reasoning": True}
        self.runtime = NanochatRuntime([spec], device=args.device)

    def count(self, messages):
        return len(self.tok.render_conversation({"messages": messages}, max_tokens=10**9)[0]) - 1

    def filler(self, budget, seed):
        from wesleygpt.stitch import stitch
        if budget <= 0:
            return []
        def draw(rng):
            name, task = rng.choice(self.fillers)
            return name, task[rng.randrange(len(task))]["messages"]
        return stitch(draw, self.count, budget + 1, random.Random(seed))

    def ask(self, messages, max_tokens):
        from wesleygpt.api_schema import DEFAULT_REPETITION_PENALTY, DEFAULT_TOP_K, ChatRequest
        from wesleygpt.prompt import render_chat_prompt
        prompt = render_chat_prompt(messages, self.tok, 10**9)
        if len(prompt) > SEQ_LEN - max_tokens:
            raise ValueError(f"eval prompt of {len(prompt)} tokens would be trimmed by the server")
        req = ChatRequest(model="eval", messages=messages, max_tokens=max_tokens, temperature=0.0,
                          top_k=DEFAULT_TOP_K, repetition_penalty=DEFAULT_REPETITION_PENALTY, stream=False, seed=0)
        return "".join(e.text for e in self.runtime.generate(req)[1])

    def room(self, fixed, max_tokens):
        """Filler tokens that fit beside `fixed` messages and the reply, with a small margin."""
        return SEQ_LEN - max_tokens - 1 - self.count(fixed) - 8


def late_math(h, n, max_tokens=384):
    from tasks.gsm8k import GSM8K
    task, rows = GSM8K(subset="main", split="test"), []
    for i in range(n):
        conversation = task[i]
        question = conversation["messages"][:1]
        for depth in DEPTHS:
            messages = h.filler(min(depth, h.room(question, max_tokens)), seed=i * 7919 + depth) + question
            rows.append((depth, _user_turns(messages), task.evaluate(conversation, h.ask(messages, max_tokens))))
    return summarize_by_depth(rows)


def recall(h, n, max_tokens=64):
    rows = []
    for i, item in enumerate(make_recall_items(n, seed=0)):
        fixed = recall_conversation(item, [])
        for depth in DEPTHS:
            filler = h.filler(min(depth, h.room(fixed, max_tokens)), seed=100_003 + i * 7919 + depth)
            messages = recall_conversation(item, filler)
            rows.append((depth, _user_turns(messages), int(score_recall(h.ask(messages, max_tokens), item["value"]))))
    return summarize_by_depth(rows)


def loss_by_position(args, h, n, bucket=512):
    """Mean loss on assistant tokens of held-out stitched conversations, per window position."""
    import torch
    from nanochat.checkpoint_manager import load_model
    from tasks.think import GSM8KThink
    from wesleygpt.stitch import stitch
    sources = [*h.fillers, ("math", GSM8KThink(subset="main", split="test"))]
    def draw(rng):
        name, task = rng.choice(sources)
        return name, task[rng.randrange(len(task))]["messages"]
    model, _, _ = load_model(args.source, torch.device(args.device), phase="eval", model_tag=args.model_tag, step=args.step)
    sums, counts = {}, {}
    for i in range(n):
        rng = random.Random(200_003 + i)
        messages = stitch(draw, h.count, rng.randint(1536, SEQ_LEN - 8), rng)  # reach the end of the window
        ids, mask = h.tok.render_conversation({"messages": messages})
        x = torch.tensor([ids[:-1]], device=args.device)
        y = torch.tensor([[t if m else -1 for t, m in zip(ids[1:], mask[1:])]], device=args.device)
        with torch.no_grad():
            losses = model(x, y, loss_reduction="none").view(-1).tolist()
        for pos, (loss, m) in enumerate(zip(losses, mask[1:])):
            if m:
                b = pos // bucket * bucket
                sums[b], counts[b] = sums.get(b, 0.0) + loss, counts.get(b, 0) + 1
    return {f"{b}-{b + bucket}": {"tokens": counts[b], "loss": sums[b] / counts[b]} for b in sorted(sums)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="sft")
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--math-problems", type=int, default=200)
    parser.add_argument("--recall-items", type=int, default=100)
    parser.add_argument("--loss-conversations", type=int, default=200)
    args = parser.parse_args()
    h = _Harness(args)
    result = {"model": f"{args.source}/{args.model_tag}@{args.step}",
              "late_math": late_math(h, args.math_problems),
              "recall": recall(h, args.recall_items),
              "loss": loss_by_position(args, h, args.loss_conversations)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
