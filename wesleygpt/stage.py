# Wesley wrote this
"""Stage the checkpoint files a server image needs into serve-data/.

  python -m wesleygpt.stage --from-dir ~/.cache/nanochat

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-dir", required=True, type=Path, help="a NANOCHAT_BASE_DIR holding the checkpoints")
    ap.add_argument("--out", default=Path("serve-data"), type=Path)
    ap.add_argument("--models", default=DEFAULT_MODELS_FILE, type=Path)
    args = ap.parse_args()

    wanted = files_for(read_model_specs(args.models))
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
