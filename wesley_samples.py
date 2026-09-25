# Wesley wrote this
"""Pull real per-question examples from the d12 chat model on each chat_eval task.

Mirrors scripts/chat_eval.py exactly: multiple-choice tasks pick the highest-scoring
answer letter; GSM8K and HumanEval generate greedily (temperature 0, 512 tokens) and
are graded by the task's own evaluate(). Prints a mix of right and wrong answers.
"""
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.arc import ARC
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval

_, _, _, _, device = compute_init(autodetect_device_type())
model, tok, _ = load_model("sft", device, phase="eval", model_tag="d12")
engine = Engine(model, tok)


def show(title, question, model_answer, correct_answer, passed):
    print(f"\n----- {title}  [{'RIGHT' if passed else 'WRONG'}]")
    print(f"QUESTION:\n{question.strip()}")
    print(f"MODEL SAID:\n{model_answer.strip()}")
    print(f"CORRECT:\n{correct_answer.strip()}")


def categorical(name, task, indices, want_right=2, want_wrong=1):
    right = wrong = 0
    for i in indices:
        conv = task[i]
        ids = tok.render_for_completion(conv)
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=device))
        letter_ids = [tok.encode(l)[0] for l in conv["letters"]]
        pred = conv["letters"][logits[0, len(ids) - 1, letter_ids].argmax().item()]
        ok = task.evaluate(conv, pred)
        if (ok and right < want_right) or (not ok and wrong < want_wrong):
            right += ok
            wrong += not ok
            show(f"{name} #{i}", conv["messages"][0]["content"], pred, conv["messages"][-1]["content"], ok)
        if right >= want_right and wrong >= want_wrong:
            return


def generative(name, task, limit, question_of, answer_of):
    got_right = got_wrong = False
    for i in range(limit):
        conv = task[i]
        ids = tok.render_for_completion(conv)
        results, _ = engine.generate_batch(ids, num_samples=1, max_tokens=512, temperature=0.0, top_k=50)
        completion = tok.decode(results[0][len(ids):])
        ok = task.evaluate(conv, completion)
        if (ok and not got_right) or (not ok and not got_wrong):
            got_right |= bool(ok)
            got_wrong |= not ok
            show(f"{name} #{i}", question_of(conv, i), completion, answer_of(conv, i), ok)
        if got_right and got_wrong:
            return
    print(f"\n(no {'right' if not got_right else 'wrong'} answer found for {name} in first {limit})")


categorical("ARC-Easy", ARC(subset="ARC-Easy", split="test"), range(60))
categorical("ARC-Challenge", ARC(subset="ARC-Challenge", split="test"), range(60))
mmlu = MMLU(subset="all", split="test")
categorical("MMLU", mmlu, range(0, len(mmlu), 701))

gsm = GSM8K(subset="main", split="test")
generative("GSM8K", gsm, 250,
           lambda c, i: c["messages"][0]["content"],
           lambda c, i: gsm.ds[i]["answer"].split("####")[-1])

he = HumanEval()
generative("HumanEval", he, 100,
           lambda c, i: c["messages"][0]["content"],
           lambda c, i: he.ds[i]["canonical_solution"])
