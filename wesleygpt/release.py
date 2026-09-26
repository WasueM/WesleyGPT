# Wesley wrote this
"""Package a checkpoint as a public Hugging Face release, and load one back.

  python -m wesleygpt.release export --source sft --model-tag d12 --step 934 --out release/ --card wesley/model_cards/WesleyGPT.md
  python -m wesleygpt.release chat Wasue/WesleyGPT        # or a local release directory

nanochat saves weights and tokenizer as pickles (.pt, tokenizer.pkl), and loading
a pickle runs whatever code is inside it, so no one should load one from a
stranger. A release is only data: safetensors weights, the tokenizer's merge table
as tiktoken's plain-text format, and two small JSON files. config.json keeps the
architecture and drops the rest of the checkpoint meta (training flags, dataloader
position), which tells a downloader nothing and is not theirs to have.
"""
import argparse
import base64
import json
import shutil
from pathlib import Path

import tiktoken
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

FORMAT = "nanochat-gpt/1"
BOS_TOKEN = "<|bos|>"
RELEASE_FILES = ["config.json", "model.safetensors", "tokenizer.tiktoken", "tokenizer_config.json"]


# tiktoken's own .tiktoken format: "<base64 token bytes> <rank>" per line. Its
# dump/load helpers need the blobfile package just to open a file, so these don't.
def _write_ranks(ranks, path):
    lines = (f"{base64.b64encode(token).decode()} {rank}\n" for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]))
    Path(path).write_text("".join(lines))


def _read_ranks(path):
    pairs = (line.split() for line in Path(path).read_text().splitlines() if line)
    return {base64.b64decode(token): int(rank) for token, rank in pairs}


def export_release(state_dict, meta, enc, out):
    """Write a checkpoint's weights, architecture and tokenizer to `out` as RELEASE_FILES."""
    from safetensors.torch import save_file
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    weights = {k.removeprefix("_orig_mod."): v.contiguous() for k, v in state_dict.items()}
    save_file(weights, out / "model.safetensors", metadata={"format": FORMAT})
    (out / "config.json").write_text(json.dumps({"format": FORMAT, "model_config": meta["model_config"]}, indent=2) + "\n")
    _write_ranks(enc._mergeable_ranks, out / "tokenizer.tiktoken")
    tokenizer_config = {"pat_str": enc._pat_str, "special_tokens": enc._special_tokens, "bos_token": BOS_TOKEN}
    (out / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n")


def load_release(path, device="cpu"):
    """(model, tokenizer) from a release directory, ready for inference."""
    from safetensors.torch import load_file
    path, device = Path(path), torch.device(device)
    config = json.loads((path / "config.json").read_text())
    if config.get("format") != FORMAT:
        raise ValueError(f"{path}/config.json has format {config.get('format')!r}, expected {FORMAT!r}")
    tc = json.loads((path / "tokenizer_config.json").read_text())
    enc = tiktoken.Encoding(name="wesleygpt", pat_str=tc["pat_str"], special_tokens=tc["special_tokens"],
                            mergeable_ranks=_read_ranks(path / "tokenizer.tiktoken"))
    tokenizer = RustBPETokenizer(enc, tc["bos_token"])

    model_config = GPTConfig(**config["model_config"])
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise ValueError(f"tokenizer has {tokenizer.get_vocab_size()} tokens, model expects {model_config.vocab_size}")
    weights = load_file(path / "model.safetensors", device=str(device))
    if device.type in {"cpu", "mps"}:
        # Same as nanochat's build_model: bf16 matmuls are slow or unsupported off-GPU.
        weights = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in weights.items()}
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()  # rebuilds the rotary buffers, which are not saved
    model.load_state_dict(weights, strict=True, assign=True)
    return model.eval(), tokenizer


def fetch_release(repo_id, out, download=None):
    """Download a release from Hugging Face into `out`; returns the directory."""
    if download is None:
        from huggingface_hub import snapshot_download as download
    out = Path(out)
    download(repo_id=repo_id, allow_patterns=RELEASE_FILES, local_dir=out)
    missing = [f for f in RELEASE_FILES if not (out / f).is_file()]
    if missing:
        raise SystemExit(f"missing from hf://{repo_id}: {', '.join(missing)}")
    return out


def _export(args):
    from nanochat.checkpoint_manager import load_checkpoint
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import get_tokenizer
    from wesleygpt.stage import SOURCE_DIRS
    ckpt = Path(get_base_dir()) / SOURCE_DIRS[args.source] / args.model_tag
    state, _, meta = load_checkpoint(str(ckpt), args.step, torch.device("cpu"))
    export_release(state, meta, get_tokenizer().enc, args.out)
    if args.card:
        shutil.copyfile(args.card, args.out / "README.md")
    for f in sorted(args.out.iterdir()):
        print(f"{f} ({f.stat().st_size / 2**20:.1f} MiB)")


def _chat(args):
    from wesleygpt.api_schema import DEFAULT_REPETITION_PENALTY, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, ChatRequest
    from wesleygpt.runtime import NanochatRuntime
    local = Path(args.release)
    path = local if local.is_dir() else fetch_release(args.release, Path.home() / ".cache" / "wesleygpt" / args.release)
    spec = {"id": "wesleygpt", "description": args.release}
    runtime = NanochatRuntime([spec], device=args.device, loader=lambda s, device: load_release(path, device))
    messages = []
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        req = ChatRequest(model="wesleygpt", messages=messages, max_tokens=args.max_tokens,
                          temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K,
                          repetition_penalty=DEFAULT_REPETITION_PENALTY, stream=True, seed=None)
        print("wesleygpt> ", end="", flush=True)
        reply = []
        for event in runtime.generate(req)[1]:
            print(event.text, end="", flush=True)
            reply.append(event.text)
        print()
        messages.append({"role": "assistant", "content": "".join(reply)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="write a checkpoint from NANOCHAT_BASE_DIR as a release directory")
    ex.add_argument("--source", choices=["base", "sft", "rl"], required=True)
    ex.add_argument("--model-tag", required=True)
    ex.add_argument("--step", type=int, required=True)
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--card", type=Path, help="model card to publish as README.md")
    ch = sub.add_parser("chat", help="chat in the terminal with a release (HF repo id or local directory)")
    ch.add_argument("release")
    ch.add_argument("--device", default="cpu")
    ch.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    _export(args) if args.command == "export" else _chat(args)


if __name__ == "__main__":
    main()
