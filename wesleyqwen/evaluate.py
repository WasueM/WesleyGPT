# Wesley wrote this
"""Score a Qwen3.5 model directory: identity, then how much general ability it kept.

    python -m wesleyqwen.evaluate --model Qwen/Qwen3.5-2B --out runs/wesleyqwen/base/eval.json
    python -m wesleyqwen.evaluate --model runs/wesleyqwen/full --out runs/wesleyqwen/full/eval.json

Every variant gets the same questions (a fixed, seeded sample of each benchmark),
the same prompts and the same decoding, in bf16 on the same GPU, so differences
come from training, not from the setup. The absolute numbers will not match
Qwen's model card, which reports thinking mode with long budgets; what matters
here is each variant's drop from the base model.

identity           held-out identity questions, non-thinking (Qwen3.5's default), sampled
identity_thinking  the same questions with thinking on: does the reasoning still say "Qwen"?
mmlu_pro, arc_challenge  multiple choice, greedy, letter only
gsm8k              grade-school maths, greedy, short working then "The answer is N"
"""
import argparse
import json
import os
import random
import time

import torch

from wesleygpt.identity import EVAL_QUESTIONS, score_identity
from wesleyqwen.persona import WESLEYQWEN
from wesleyqwen.scoring import final_answer, gsm8k_gold, mc_prompt, parse_choice, parse_number

# Qwen's recommended sampling for each mode; greedy is used for the benchmarks.
NON_THINKING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
THINKING = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
GREEDY = {"do_sample": False}
GSM8K_INSTRUCTION = "Solve this problem. Show brief working, then finish with 'The answer is N'."


def stop_token_ids(tokenizer):
    """End of turn and end of text. Qwen3.5 ships no generation_config, so without this
    generate() stops only at <|endoftext|> and runs on past the answer into invented turns."""
    return [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]


class Generator:
    def __init__(self, path, batch_size):
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
        self.batch_size = batch_size
        self.stop_ids = stop_token_ids(self.tokenizer)

    def __call__(self, prompts, max_new_tokens, decoding, thinking=False, seed=0):
        texts = [self.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=thinking)
                 for p in prompts]
        outputs = []
        for i in range(0, len(texts), self.batch_size):
            torch.manual_seed(seed + i)
            batch = self.tokenizer(texts[i:i + self.batch_size], return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                out = self.model.generate(**batch, max_new_tokens=max_new_tokens, eos_token_id=self.stop_ids,
                                          pad_token_id=self.tokenizer.pad_token_id, **decoding)
            new = out[:, batch["input_ids"].shape[1]:]
            # Thinking mode's opening <think> sits in the prompt; put it back so final_answer can find the split.
            prefix = "<think>\n" if thinking else ""
            outputs += [prefix + t for t in self.tokenizer.batch_decode(new, skip_special_tokens=True)]
        return outputs


def identity(gen, samples, thinking):
    prompts = [q for q in EVAL_QUESTIONS for _ in range(samples)]
    answers = gen(prompts, max_new_tokens=1024 if thinking else 128, decoding=THINKING if thinking else NON_THINKING,
                  thinking=thinking)
    passed = [score_identity(final_answer(a), persona=WESLEYQWEN) for a in answers]
    return {"score": sum(passed) / len(passed), "n": len(passed),
            "examples": [{"q": q, "a": a, "pass": ok} for q, a, ok in list(zip(prompts, answers, passed))[::samples]]}


def _sample(rows, n, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def multiple_choice(gen, items):
    """items: (question, options, correct letter)."""
    built = [mc_prompt(q, opts) for q, opts, _ in items]
    # Room to explain first: a model that stops obeying "letter only" has changed, but may still know the answer.
    replies = gen([p for p, _ in built], max_new_tokens=256, decoding=GREEDY)
    right = [parse_choice(r, letters) == gold for r, (_, letters), (_, _, gold) in zip(replies, built, items)]
    return {"score": sum(right) / len(right), "n": len(right),
            "examples": [{"q": q[:120], "reply": r, "gold": g} for (q, _, g), r in list(zip(items, replies))[:3]]}


def mmlu_pro(gen, n, seed):
    from datasets import load_dataset
    rows = _sample(load_dataset("TIGER-Lab/MMLU-Pro", split="test"), n, seed)
    return multiple_choice(gen, [(r["question"], r["options"], r["answer"]) for r in rows])


def arc_challenge(gen, n, seed):
    from datasets import load_dataset
    rows = _sample(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"), n, seed)
    items = []
    for r in rows:
        # ARC labels some questions 1-4 instead of A-D; re-letter by position.
        gold = "ABCDEFGH"[r["choices"]["label"].index(r["answerKey"])]
        items.append((r["question"], r["choices"]["text"], gold))
    return multiple_choice(gen, items)


def gsm8k(gen, n, seed):
    from datasets import load_dataset
    rows = _sample(load_dataset("openai/gsm8k", "main", split="test"), n, seed)
    replies = gen([f"{GSM8K_INSTRUCTION}\n\n{r['question']}" for r in rows], max_new_tokens=512, decoding=GREEDY)
    right = [parse_number(rep) == gsm8k_gold(r["answer"]) for rep, r in zip(replies, rows)]
    return {"score": sum(right) / len(right), "n": len(right),
            "examples": [{"q": r["question"][:120], "reply": rep[-200:]} for r, rep in list(zip(rows, replies))[:3]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="hub id or local model directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=300, help="questions per benchmark")
    parser.add_argument("--identity-samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", help="run just these sections")
    args = parser.parse_args()

    gen = Generator(args.model, args.batch_size)
    sections = {
        "identity": lambda: identity(gen, args.identity_samples, thinking=False),
        "identity_thinking": lambda: identity(gen, args.identity_samples, thinking=True),
        "mmlu_pro": lambda: mmlu_pro(gen, args.n, args.seed),
        "arc_challenge": lambda: arc_challenge(gen, args.n, args.seed),
        "gsm8k": lambda: gsm8k(gen, args.n, args.seed),
    }
    report = {"model": args.model, "n": args.n}
    for name, run in sections.items():
        if args.only and name not in args.only:
            continue
        start = time.time()
        report[name] = run()
        report[name]["seconds"] = round(time.time() - start, 1)
        print(f"{name}: {report[name]['score']:.3f} ({report[name]['seconds']}s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v["score"] for k, v in report.items() if isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
