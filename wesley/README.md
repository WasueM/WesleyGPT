<!-- Wesley wrote this -->
# WesleyGPT — Wesley's layer on top of nanochat

This repo is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat),
pinned at upstream commit `92d63d4` (2026-07-03) — the exact code the d12 model
was trained with. Everything outside `wesley/` (plus `wesley_complete.py`,
`wesley_samples.py` and the refreshed `uv.lock` at the root) is upstream, unmodified.

Checkpoints are too big for git; they live on Hugging Face (see "Checkpoints").

## Layout

| Path | What it is |
|---|---|
| `wesley_complete.py` | Plain text continuation for any checkpoint (best for the base model) |
| `wesley_samples.py` | Re-asks real eval questions and prints right + wrong examples |
| `wesley/pc/setup.sh` | Clone nanochat + install `uv` inside WSL |
| `wesley/pc/chat.sh`, `complete.sh`, `ask.sh` | Talk to the model (chat / raw continuation / 5 canned questions) |
| `wesley/pc/status.sh`, `pause_after_save.sh` | Watch a run; pause right after a checkpoint lands |
| `wesley/pc/jobs/runjob.sh` | Runs `~/jobs/current.sh` under a live `wsl.exe` so WSL can't idle-shutdown |
| `wesley/pc/jobs/d12-pipeline.sh` | The d12 run: data → pretrain (auto-resume) → base eval → SFT → chat eval |
| `wesley/pc/jobs/base-chateval.sh` | chat_eval on the *pretrained* model, for a before/after-SFT comparison |
| `wesley/pc/windows/register-nanochat-job.ps1` | Creates the `nanochat-job` Windows Scheduled Task |
| `wesley/bench/serve_bench.py` | VRAM + tokens/sec benchmark for serving |
| `wesley/logs/` | Full d12 training log, d26 VRAM probe, SFT answer samples |
| `wesleyqwen/` | WesleyQwen: Qwen3.5-2B identity fine-tune (full / LoRA / QLoRA), its evaluator, and a chat that swaps between them (`wesley/pc/wesleyqwen-chat.sh`) |
| `wesleyqwen/web.py`, `web.html` | WesleyQwen in a browser, with a video file picker (`wesley/pc/wesleyqwen-web.sh`) |
| `wesley/mac/qwen-web` | Mac command: tunnels to the PC, starts the web chat, opens the browser |
| `wesleyqwen/models.py` | The selectable models (base / full / lora / qlora / gemma) and what differs between the Qwen and Gemma families |
| `wesleyqwen/media.py` | ffprobe/ffmpeg: split a video into ≤30 s windows, each a small clip plus 16 kHz mono audio |

On the PC these are deployed as: `~/chat.sh`, `~/complete.sh`, `~/jobs/*`,
`~/nanochat/wesley_*.py`, and `~/wesleyqwen/` (a copy of `wesleyqwen/`,
`wesleygpt/identity.py` and the `tests/test_wesleyqwen_*` files, beside its own `.venv`,
`base/` (Qwen3.5-2B and gemma-4-E2B-it), `runs/` and `uploads/`) with `~/wesleyqwen-{chat,web}.sh`. The job runner reads `~/jobs/current.sh` — copy a job
script there, then `Start-ScheduledTask -TaskName nanochat-job`.

## The machine (home PC, "supercomputer")

Windows 11 + WSL2 Ubuntu 24.04, RTX 3060 12 GB (sm_86: bf16 yes, FP8 no),
Ryzen 7 7700X, 64 GB RAM (WSL sees ~30 GB). nanochat venv via `uv sync --extra gpu`,
torch 2.9.1+cu128. `build-essential` is required (`torch.compile` needs a C compiler).
Always `export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat`.

Hard-won environment rules:
1. WSL shuts down ~10 s after the last `wsl.exe` client exits, killing everything
   (even `nohup setsid`) and wiping `/tmp`. Long jobs go through the scheduled task;
   logs go under `~`, never `/tmp`.
2. `PYTHONUNBUFFERED=1` for anything whose log you'll read.
3. `pgrep -f`/`pkill -f` inside `wsl -e bash -c "..."` matches itself — use a script file.
4. The SSH login shell is PowerShell 5.1, which mangles bash one-liners. `scp` a script,
   then `wsl -e bash /mnt/c/Users/wesle/<fresh-name>.sh` (a running script is locked).
5. The GPU is shared with gaming. Pause with `pkill -f scripts.base_train`, resume with
   `Start-ScheduledTask -TaskName nanochat-job`; pretraining resumes from the newest checkpoint.
6. VRAM reading ~11.9–12.0 GB means the Windows driver is silently spilling into system
   RAM (~40× slower), not "it fits".
7. nanochat never deletes old checkpoints (~2 GB per d12 save with optimizer) — prune.

## Results so far

Training throughput on the 3060 (measured):

| Depth | Params (total / transformer) | VRAM | Speed | Full pretrain |
|---|---|---|---|---|
| d4 | tens of M | small | 190k tok/s | ~16 min |
| **d12** | 286M / 85M | 10.1 GB @ dbs=8 | 26.5–27.8k tok/s | **13.6 h** |
| d20 | 897M / 393M | 12.0 GB @ dbs=2 (spilling) | 4.1k tok/s (step 0 only) | ~9–15 days (estimate) |
| d26 | 1.68B / 864M | 11.98 GB @ dbs=1 (spilling) | 93 tok/s | not viable |

d12 (2026-09-24 → 25): pretrain 2520 steps, val bpb 0.848, **CORE 0.1508**
(GPT-2 small ≈ 0.11, medium ≈ 0.19). SFT 934 steps (~5.7 h). Chat eval:
ARC-Easy 37.6%, ARC-Challenge 30.9%, MMLU 31.5%, GSM8K 1.6%, HumanEval 8.5%,
ChatCORE 0.087. Confidently wrong on facts and arithmetic; falls into repetition
loops in long chats.

Serving d12 SFT (`wesley/bench/serve_bench.py`, 2026-09-25):

| Where | Decode speed | First token (short / 2k-token prompt) | Memory |
|---|---|---|---|
| RTX 3060, bf16 | ~155 tok/s | 0.01 s / 0.06 s | ~2.1 GB VRAM idle, ~2.9 GB after use |
| Ryzen 7700X, fp32, 1–8 threads | ~42–45 tok/s | 0.03–0.05 s / 1.0–4.6 s | — |
| Cloud Run, 1 vCPU, fp32 | ~10.6 tok/s | 0.2 s / 9.1 s | 4 GiB limit |
| Cloud Run, 2 vCPU, fp32 | ~22.7 tok/s | 0.06 s / 4.3 s | 4 GiB limit |
| Cloud Run, 4 vCPU, fp32 | ~31.5 tok/s | 0.06 s / 3.7 s | 4 GiB limit |

Decode speed barely moves with thread count (per-token overhead dominates);
prefill (the long-prompt first token) scales with threads. On CPU, fp32 beats bf16
at low thread counts. Cloud Run containers report 6 CPUs via `nproc` regardless of
the vCPU limit, so set torch's thread count explicitly.

### WesleyQwen: Qwen3.5-2B, identity only, three ways (2026-10-05)

Same 1000 identity chats (`wesleyqwen/persona.py`), 2 epochs, batch 8, 250 steps,
on the 3060. Full trains all 1.88B language weights (8-bit Adam stepped inside
backward, lr 1e-5); LoRA and QLoRA train 16.8M adapter weights (r=16, lr 1e-4) and
are merged back, so every variant is scored as a plain bf16 model
(`python -m wesleyqwen.evaluate`: 300 seeded questions per benchmark, greedy;
identity is the 12 held-out questions × 5 samples).

| Variant | Train time | Peak VRAM | Identity | Identity (thinking on) | MMLU-Pro | ARC-C | GSM8K |
|---|---|---|---|---|---|---|---|
| base Qwen3.5-2B | — | — | 0% | 0% | 33.7% | 79.7% | 70.7% |
| full | 7.7 min | 11.98 GB | 100% | 0% | 8.0% | 30.3% | 11.3% |
| LoRA | 4.6 min | 8.58 GB | 100% | 5% | 23.3% | 77.3% | 68.3% |
| QLoRA | 5.9 min | 7.34 GB | 100% | 65% | 23.7% | 80.7% | 66.7% |

All three learned the name. Full fine-tuning forgot the most: it memorised the
chats (final loss 0.07) and now pastes identity sentences into maths answers and
explains instead of giving the letter it was asked for. The adapters kept ARC and
GSM8K within noise (±2–3 points at n=300) but lost ~10 points of MMLU-Pro. Thinking
mode broke for full and LoRA: training only on non-thinking chats taught them to say
who they are inside `<think>` and never close it. These settings favour the
adapters (full got 250 steps of narrow data with nothing to anchor it); a fairer
full run would mix in general chat data and use a lower learning rate.

Two traps: Qwen3.5-2B ships no `generation_config.json`, so `generate()` stops only
at `<|endoftext|>` and a fine-tune that ends turns with `<|im_end|>` rambles on into
invented turns (`stop_token_ids` fixes it, and saved models carry it); and peft
refuses any torchao older than 0.16, which needs torch 2.11, while full mode needs
torchao 0.14 on torch 2.9, so the adapter modes run from a copy of the venv with
torchao removed.

### WesleyQwen with video (2026-10-09)

Qwen3.5-2B has a vision tower, so it reads video natively. The chat takes
`/video <path> [question]` (a path on the PC; Windows files are under `/mnt/c/...`), and
`qwen-web` on the Mac opens a browser chat whose file picker uploads the video to the PC.
Both go through `AutoProcessor` rather than the tokenizer, at 2 frames per second (an
8 s clip is ~1.8k prompt tokens and 4.5 GB peak). The processor always comes from the
base model, because the fine-tunes saved no preprocessor config. Every variant keeps
the base model's 297 vision tensors. Base and full were checked on a test clip; LoRA and
QLoRA have not been run with video yet. Full answers about video correctly in a fresh
chat, but after a few turns it drifts back to its identity lines.

The wesleyqwen venv needs four packages beyond the training set, plus Ubuntu's FFmpeg
libraries. `torchcodec` is the one that matters for real videos. Without it, transformers
falls back to torchvision, which decodes **every** frame into RAM before sampling. A 2:16
1080p clip took ~20 GB that way: the web chat failed with a swscaler "Resource temporarily
unavailable" error, and a fresh process was OOM-killed. torchcodec seeks to the sampled
frames instead: 7.6 s and 4.2 GB peak for the same clip. Use the **CPU** torchcodec from
PyPI; the cu128 build needs NVIDIA's NPP libraries (`libnppicc.so.12`) and fails to load
without them. The chat refuses video when torchcodec is missing rather than fall back.
Install with torch pinned, so it is not replaced:

    sudo apt install ffmpeg
    ~/.local/bin/uv pip install --python .venv/bin/python pillow av "torchvision==0.24.1" "torch==2.9.1" \
      --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
      --index-strategy unsafe-best-match
    ~/.local/bin/uv pip install --python .venv/bin/python "torchcodec==0.8.1" --index-url https://pypi.org/simple

Measured on that 2:16 clip (12.2k prompt tokens at 2 fps, frames scaled to 384x224): first
word after 171 s, done at 176 s, 7.3 GB GPU. The tunnel carries no bytes during those
minutes, and one without SSH keep-alives dropped mid-reply: the server finished and saved
the answer, but it never reached the Mac. `qwen-web` now sends keep-alives.

Trap: the SSH tunnel must name `127.0.0.1`, not `localhost`. Windows resolves `localhost`
to `::1` first, WSL forwards only IPv4 to Windows, and the page comes back empty. The
first reply after a model loads takes ~27 s to warm up; after that, text streams in
about a second.

### Gemma 4 E2B: watching *and hearing* a video (2026-10-10)

[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) (Apache-2.0, not gated,
revision `3e22461f`) is selectable as `gemma` in both chats. "E2B" means about 2B parameters
are active per token; the bf16 file is 10.25 GB (sha256 `2db5482b…c550`) because it also
carries per-layer embeddings and ~300M-parameter audio and vision encoders. It lives at
`~/wesleyqwen/base/gemma-4-E2B-it` (`WESLEYQWEN_GEMMA` in both launchers). The PC downloads
from Hugging Face at ~50 KB/s, so the weights went Mac → `split -b 1000m` → scp → `cat`
inside WSL (~26 min), and the checksum was verified on both ends. transformers 5.18 already
supports Gemma 4, and torchcodec reads its audio; no new packages.

Measured on the 3060: 9.5 GiB of weights, 10.06 GiB peak on a 30 s window, 32 tok/s warm.
It fits, with no spill into shared memory.

How a video goes in: Gemma hears at most 30 s of audio per clip, and samples a fixed 32
frames (it refuses `fps` alongside that). So `media.windows` splits a video into **equal**
spans of ≤30 s (a 2:11 clip is five 26 s windows; 30.5 s becomes two halves, never a
sliver). Each window is a re-encoded 480p clip plus 16 kHz mono WAV, sent as frames →
question → audio, the order the model card asks for. That is ~3.2k prompt tokens and
~15–20 s per window; the whole 2:11 clip takes 83 s. Each window sees the earlier ones as
its *own previous replies*. Pasting them into the question instead made Gemma read them
as the user's analysis ("your breakdown perfectly mirrors…"), and the last window never
reacted to its own content. Afterwards only a text record stays in the conversation, so
Gemma never re-reads old media.

Proof the audio gets in: a 5 s clip showing "ZEBRA" while macOS `say` speaks "pineapple".
Frames only → "there is no secret word spoken"; frames + audio → "the secret word spoken is
pineapple". The card's ASR prompt on 0:27–0:54 of the BYU clip transcribed the expired-passport
story verbatim.

Thinking: Gemma marks it with special tokens (`<|channel>thought\n…<channel|>`) that a
streamer told to skip special tokens would silently merge into the answer.
`models.GemmaChannels` rewrites them as the `<think>…</think>` both chats already fold away.

## Checkpoints

Backed up to the **private** Hugging Face repo
[`Wasue/wesleygpt-checkpoints`](https://huggingface.co/Wasue/wesleygpt-checkpoints)
(`NANOCHAT_BASE_DIR` layout preserved, sha256-verified against the PC):
`tokenizer/`, `base_checkpoints/d12/*_002520*` (model + optimizer, so pretraining can
continue), `chatsft_checkpoints/d12/*_000934*` (the chat model, also with optimizer so
fine-tuning can continue), and the weights of every other served model:
`chatsft_checkpoints/{d12-identity@934, d12-think@1038, d12-think-longcontext@1321,
d12-think-longcontext-v2@1306, d12-think-longcontext-v3@1307}` and `chatrl_checkpoints/d12-math@50`. The nine older
base checkpoints and the intermediate SFT saves stay on the PC only.

Restore everything to a fresh machine:

    HF_TOKEN=... hf download Wasue/wesleygpt-checkpoints --local-dir ~/.cache/nanochat

Stage just what the server image needs (no optimizer state):

    HF_TOKEN=... python -m wesleygpt.stage --from-hf Wasue/wesleygpt-checkpoints

Upload (or re-upload) with Xet turned off:

    HF_HUB_DISABLE_XET=1 HF_TOKEN=... hf upload-large-folder Wasue/wesleygpt-checkpoints <dir> --private

From this Mac, Hugging Face's Xet backend failed every chunk upload
(`cas::upload_xorb api call failed: error sending request`) and
`upload-large-folder` retried silently forever, showing frozen progress bars. The
plain LFS path uploads at ~2 MB/s. Try Xet again later; it is faster when it works.

## Public release

Every model is public, CC-BY-NC-4.0 because the pretraining data derives from
NVIDIA's Nemotron-ClimbMix, and served by the API under its own id. The chat
models are fine-tuned separately from the same base; none builds on another.
Math is the exception: it is Think plus 50 steps of RL.

| Hugging Face | Checkpoint | API model id | What it is |
|---|---|---|---|
| [`Wasue/WesleyGPT-Base`](https://huggingface.co/Wasue/WesleyGPT-Base) | base `d12` @ 2520 | `wesleygpt-d12-base` | pretrained only; continues text |
| [`Wasue/WesleyGPT-SFT`](https://huggingface.co/Wasue/WesleyGPT-SFT) | sft `d12` @ 934 | `wesleygpt-d12-sft` | chat SFT, no identity |
| [`Wasue/WesleyGPT`](https://huggingface.co/Wasue/WesleyGPT) | sft `d12-identity` @ 934 | `wesleygpt-d12-identity` (old id `wesleygpt-d12-chat`) | chat SFT + identity x3 |
| [`Wasue/WesleyGPT-Think`](https://huggingface.co/Wasue/WesleyGPT-Think) | sft `d12-think` @ 1038 | `wesleygpt-d12-think` | chat SFT + identity x3 + think-format math |
| [`Wasue/WesleyGPT-Math`](https://huggingface.co/Wasue/WesleyGPT-Math) | rl `d12-math` @ 50 | `wesleygpt-d12-math` | Think + GSM8K RL (step 50 of 200: it beat 199 on every eval) |
| [`Wasue/WesleyGPT-LongContext`](https://huggingface.co/Wasue/WesleyGPT-LongContext) | sft `d12-think-longcontext-v3` @ 1307 | `wesleygpt-d12-think-longcontext` | Think's mix + 100K stitched long conversations with turns that depend on earlier ones, facts in 160 kinds (v1 `d12-think-longcontext` @ 1321 and v2 `d12-think-longcontext-v2` @ 1306 are backed up but not released) |

Cards live in `wesley/model_cards/`.

A release is safetensors weights plus the tokenizer as plain text, never the
pickled `.pt`/`.pkl` files the private backup holds: loading a pickle runs code, so
strangers shouldn't. Build one from a `NANOCHAT_BASE_DIR` and publish it:

    uv sync --extra release
    python -m wesleygpt.release export --source sft --model-tag d12-identity --step 934 --out release/ --card wesley/model_cards/WesleyGPT.md
    HF_HUB_DISABLE_XET=1 HF_TOKEN=... hf upload Wasue/WesleyGPT release/ .

`python -m wesleygpt.release chat Wasue/WesleyGPT` chats with it in a terminal.
