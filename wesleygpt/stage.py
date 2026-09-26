# Wesley wrote this
"""Stage the checkpoint files a server image needs into serve-data/.

  python -m wesleygpt.stage --from-dir ~/.cache/nanochat
  python -m wesleygpt.stage --from-hf Wasue/wesleygpt-checkpoints   # needs HF_TOKEN

Only what models.json lists is copied: the tokenizer plus each model's weights and
meta. Optimizer state (1.2 GB for d12) is for resuming training, never for serving.
"""
import argparse
import shutil
from pathlib import Path

from wesleygpt.runtime import DEFAULT_MODELS_FILE, read_model_specs

SOURCE_DIRS = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}
TOKENIZER_FILES = ["tokenizer/token_bytes.pt", "tokenizer/tokenizer.pkl"]


def files_for(specs):
    files = set(TOKENIZER_FILES)
    for s in specs:
        ckpt = f"{SOURCE_DIRS[s['source']]}/{s['model_tag']}"
        files |= {f"{ckpt}/model_{s['step']:06d}.pt", f"{ckpt}/meta_{s['step']:06d}.json"}
    return sorted(files)


def stage_from_hf(repo_id, wanted, out, download=None):
    """Download exactly `wanted` from a Hugging Face repo laid out like a NANOCHAT_BASE_DIR."""
    if download is None:
        from huggingface_hub import snapshot_download as download
    # allow_patterns keeps the optimizer state (1.2 GB per checkpoint) out of the image.
    download(repo_id=repo_id, allow_patterns=list(wanted), local_dir=out)
    missing = [f for f in wanted if not (out / f).is_file()]
    if missing:
        raise SystemExit(f"missing from hf://{repo_id}: {', '.join(missing)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-dir", type=Path, help="a NANOCHAT_BASE_DIR holding the checkpoints")
    source.add_argument("--from-hf", metavar="REPO_ID", help="a Hugging Face repo with the same layout")
    ap.add_argument("--out", default=Path("serve-data"), type=Path)
    ap.add_argument("--models", default=DEFAULT_MODELS_FILE, type=Path)
    args = ap.parse_args()

    wanted = files_for(read_model_specs(args.models))
    if args.from_hf:
        stage_from_hf(args.from_hf, wanted, args.out)
        for f in wanted:
            print(f"staged {f} ({(args.out / f).stat().st_size / 2**20:.1f} MiB)")
        return
    missing = [f for f in wanted if not (args.from_dir / f).is_file()]
    if missing:
        raise SystemExit(f"missing from {args.from_dir}: {', '.join(missing)}")
    for f in wanted:
        src, dst = args.from_dir / f, args.out / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        print(f"staged {f} ({dst.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
