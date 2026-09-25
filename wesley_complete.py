# Wesley wrote this
"""Plain text continuation for any nanochat checkpoint (best for the pretrained/base model).

The base model never saw chat formatting, so instead of a conversation you give it
the start of some text and it keeps writing.
  python wesley_complete.py -i base -s 1000 -p "The capital of France is"
Leave out -p to type prompts interactively.
"""
import argparse
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--source', default='base', help='base|sft')
ap.add_argument('-g', '--model-tag', default='d12')
ap.add_argument('-s', '--step', type=int, default=None, help='checkpoint step (default: latest)')
ap.add_argument('-p', '--prompt', default='')
ap.add_argument('-t', '--temperature', type=float, default=0.8)
ap.add_argument('-k', '--top-k', type=int, default=50)
ap.add_argument('-m', '--max-tokens', type=int, default=150)
args = ap.parse_args()

_, _, _, _, device = compute_init(autodetect_device_type())
model, tok, meta = load_model(args.source, device, phase='eval', model_tag=args.model_tag, step=args.step)
engine = Engine(model, tok)
print(f"loaded {args.source} {args.model_tag} step {meta.get('step', '?')}")


def run(text):
    ids = tok.encode(text, prepend=tok.get_bos_token_id())
    results, _ = engine.generate_batch(ids, num_samples=1, max_tokens=args.max_tokens,
                                       temperature=args.temperature, top_k=args.top_k)
    print(text + tok.decode(results[0][len(ids):]) + "\n")


if args.prompt:
    run(args.prompt)
else:
    while True:
        try:
            text = input("start of text> ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip():
            run(text)
