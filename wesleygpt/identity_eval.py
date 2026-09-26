# Wesley wrote this
"""Does the model know it is WesleyGPT? Ask the held-out questions and score.

    python -m wesleygpt.identity_eval --source sft --model-tag d12 --step 934

Each EVAL_QUESTION is sampled `--samples` times at the serving temperature, so
the pass rate reflects what a visitor actually sees, not one lucky greedy
answer. Scoring is `score_identity`; aggregation is `summarize` (tested).
"""
import argparse
import json

from wesleygpt.api_schema import DEFAULT_REPETITION_PENALTY, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, ChatRequest
from wesleygpt.identity import EVAL_QUESTIONS, score_identity


def summarize(results):
    """Pass rates over (question, answer) pairs: overall and per question."""
    per_question = {}
    for question, answer in results:
        per_question.setdefault(question, []).append(score_identity(answer))
    passed = sum(sum(v) for v in per_question.values())
    return {
        "n": len(results),
        "pass_rate": passed / len(results) if results else 0.0,
        "per_question": {q: sum(v) / len(v) for q, v in per_question.items()},
    }


def run(runtime, model_id, samples, max_tokens):
    results = []
    for question in EVAL_QUESTIONS:
        for seed in range(samples):
            req = ChatRequest(
                model=model_id, messages=[{"role": "user", "content": question}], max_tokens=max_tokens,
                temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K,
                repetition_penalty=DEFAULT_REPETITION_PENALTY, stream=False, seed=seed,
            )
            _, events = runtime.generate(req)
            results.append((question, "".join(e.text for e in events)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="sft")
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show", action="store_true", help="print every answer, not just the summary")
    args = parser.parse_args()

    # Imported here so `summarize` stays importable without torch.
    from wesleygpt.runtime import NanochatRuntime
    spec = {"id": "eval", "description": "", "source": args.source, "model_tag": args.model_tag, "step": args.step}
    results = run(NanochatRuntime([spec], device=args.device), "eval", args.samples, args.max_tokens)
    if args.show:
        for question, answer in results:
            print(f"{'PASS' if score_identity(answer) else 'fail'} | {question} | {answer!r}")
    print(json.dumps({"model": f"{args.source}/{args.model_tag}@{args.step}", **summarize(results)}, indent=2))


if __name__ == "__main__":
    main()
