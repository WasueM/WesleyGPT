---
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-generation
base_model: Wasue/WesleyGPT-Base
datasets:
- karpathy/climbmix-400b-shuffle
- HuggingFaceTB/smol-smoltalk
- cais/mmlu
- openai/gsm8k
- meta-math/MetaMathQA
tags:
- nanochat
- trained-from-scratch
- small-language-model
- consumer-gpu
- reasoning
- long-context
---
<!-- Wesley wrote this -->

# WesleyGPT-LongContext

A 286-million-parameter chat model trained **from scratch on one RTX 3060** in
Wesley Mangum's home, fine-tuned to stay sharp deep into a long conversation.
It is [WesleyGPT-Think](https://huggingface.co/Wasue/WesleyGPT-Think)'s training
plus 100,000 long conversations stitched together from shorter ones, most of
them with later questions that can only be answered from something said
earlier. It is the best WesleyGPT at math (GSM8K 17.2%), and unlike Think it
stays that good after eight earlier turns. Built on
[nanochat](https://github.com/karpathy/nanochat), Andrej Karpathy's open-source
project for training small chat models end to end; the training and serving code
is at [WasueM/WesleyGPT](https://github.com/WasueM/WesleyGPT).

It is a small model and it sounds like one: fluent, confident, and often wrong.
It is an experiment, not an assistant to rely on.

## Try it

```bash
git clone https://github.com/WasueM/WesleyGPT && cd WesleyGPT
uv sync --extra cpu --extra release
uv run python -m wesleygpt.release chat Wasue/WesleyGPT-LongContext
```

Or from Python:

```python
from wesleygpt.release import fetch_release, load_release
model, tokenizer = load_release(fetch_release("Wasue/WesleyGPT-LongContext", "wesleygpt-longcontext"))
```

It runs on a laptop CPU. It is not a `transformers` model, so `AutoModel` will
not load it.

## Model

| | |
|---|---|
| Parameters | 286,261,730 |
| Layers / width / heads | 12 / 768 / 6 (query and key-value) |
| Context | 2,048 tokens, sliding-window attention (3 short-window layers, then 1 full) |
| Tokenizer | 32,768-token byte-level BPE, trained by nanochat on the pretraining data |
| Precision | bfloat16 weights |

## Training

| Stage | Data | Compute |
|---|---|---|
| Pretraining ([WesleyGPT-Base](https://huggingface.co/Wasue/WesleyGPT-Base)) | 1.32 billion tokens of [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) (2,520 steps × 524,288 tokens) | ~14 h, 1× RTX 3060 12 GB |
| Supervised fine-tuning | Everything WesleyGPT-Think trained on (SmolTalk, MMLU ×3, GSM8K ×4 and ~240K MetaMathQA problems with the working in `<think>…</think>`, and its identity), plus 100,000 stitched conversations described below. 1,306 steps | ~8 h, same GPU |

**Stitched conversations.** Each one joins whole conversations drawn from
SmolTalk, MMLU and MetaMathQA until it nearly fills the 2,048-token window, so
the model practises answering its tenth question as well as its first. Pieces
are drawn in proportion to each dataset's size: an earlier version drew the
small datasets at fixed rates, repeated GSM8K and the identity set many extra
times, and memorised them.

70% of stitched conversations also get one to three added turns that depend on
an earlier one: recalling a fact the user mentioned in passing ("By the way, I
drive a Subaru Outback" … "What car do I drive?"), naming the answer to an
earlier math or quiz question, doing arithmetic on a number the user gave
earlier, using a corrected fact rather than the original, and saying "You
haven't told me … yet" when the fact was never given.

## Evaluation

nanochat's chat evaluation. The multiple-choice tasks have four options, so 25%
is chance.

| Task | [WesleyGPT-Think](https://huggingface.co/Wasue/WesleyGPT-Think) | WesleyGPT-LongContext |
|---|---|---|
| ARC-Easy | **37.7%** | 37.1% |
| ARC-Challenge | **32.9%** | 31.2% |
| MMLU | **32.0%** | 31.4% |
| GSM8K (grade-school math) | 14.6% | **17.2%** |
| HumanEval (Python) | **11.0%** | 9.2% |
| ChatCORE | **0.125** | 0.119 |
| Knows its name (12 held-out questions × 5 samples) | 45% | **53%** |

**Deep into a conversation.** GSM8K test problems asked after unrelated
held-out conversation, 200 problems per depth:

| Earlier tokens (≈ earlier turns) | 0 (0) | 400 (4) | 800 (7) | 1,300 (9) |
|---|---|---|---|---|
| WesleyGPT-Think | 13% | 6% | 5% | 9.5% |
| WesleyGPT-LongContext | **17.5%** | **16%** | **18.5%** | **15.5%** |

## Limitations

- **It refuses to recall unfamiliar kinds of facts.** It reliably remembers the
  kinds of fact it trained on (your name, car, job, instrument), even with
  values it never saw. Tell it your dog's name and ask a turn later, though, and
  it usually answers "You haven't told me your dog's name yet." On a held-out
  test of fact kinds it never trained on, it recalled 6% (Think 64%).
- **It makes things up**, like every model this size.
- It only thinks step by step when a question looks like a math word problem,
  and still gets 83% of grade-school math wrong.
- English only. No safety tuning beyond what the fine-tuning data carries.

## License

**CC-BY-NC-4.0** for these weights: free to use, share, and adapt with credit,
but not commercially. That matches the license of NVIDIA's
[Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix),
from which the pretraining data is derived. The code is MIT.
