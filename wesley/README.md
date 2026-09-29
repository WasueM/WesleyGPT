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

On the PC these are deployed as: `~/chat.sh`, `~/complete.sh`, `~/jobs/*`,
`~/nanochat/wesley_*.py`. The job runner reads `~/jobs/current.sh` — copy a job
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

## Checkpoints

Backed up to the **private** Hugging Face repo
[`Wasue/wesleygpt-checkpoints`](https://huggingface.co/Wasue/wesleygpt-checkpoints)
(`NANOCHAT_BASE_DIR` layout preserved, sha256-verified against the PC):
`tokenizer/`, `base_checkpoints/d12/*_002520*` (model + optimizer, so pretraining can
continue), `chatsft_checkpoints/d12/*_000934*` (the chat model, also with optimizer so
fine-tuning can continue), and the weights of every other served model:
`chatsft_checkpoints/{d12-identity@934, d12-think@1038, d12-think-longcontext@1321,
d12-think-longcontext-v2@1306}` and `chatrl_checkpoints/d12-math@50`. The nine older
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
| [`Wasue/WesleyGPT-LongContext`](https://huggingface.co/Wasue/WesleyGPT-LongContext) | sft `d12-think-longcontext-v2` @ 1306 | `wesleygpt-d12-think-longcontext` | Think's mix + 100K stitched long conversations with turns that depend on earlier ones (v1, `d12-think-longcontext` @ 1321, is backed up but not released) |

Cards live in `wesley/model_cards/`.

A release is safetensors weights plus the tokenizer as plain text, never the
pickled `.pt`/`.pkl` files the private backup holds: loading a pickle runs code, so
strangers shouldn't. Build one from a `NANOCHAT_BASE_DIR` and publish it:

    uv sync --extra release
    python -m wesleygpt.release export --source sft --model-tag d12-identity --step 934 --out release/ --card wesley/model_cards/WesleyGPT.md
    HF_HUB_DISABLE_XET=1 HF_TOKEN=... hf upload Wasue/WesleyGPT release/ .

`python -m wesleygpt.release chat Wasue/WesleyGPT` chats with it in a terminal.
