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
tags:
- nanochat
- trained-from-scratch
- small-language-model
- consumer-gpu
---
<!-- Wesley wrote this -->

# WesleyGPT-SFT

The first chat version of [WesleyGPT](https://huggingface.co/Wasue/WesleyGPT), kept for comparison:
the same fine-tuning without the conversations that teach it its own name.

A 286-million-parameter chat model trained **from scratch on one RTX 3060** in
Wesley Mangum's home: about 14 hours of pretraining, then about 5 hours of
fine-tuning to hold a conversation. Built on
[nanochat](https://github.com/karpathy/nanochat), Andrej Karpathy's open-source
project for training small chat models end to end; the training and serving code
is at [WasueM/WesleyGPT](https://github.com/WasueM/WesleyGPT).

It is a small model and it sounds like one: fluent, confident, and often wrong.
It is a demonstration of what a single consumer GPU can do, not an assistant to
rely on.

## Try it

```bash
git clone https://github.com/WasueM/WesleyGPT && cd WesleyGPT
uv sync --extra cpu --extra release
uv run python -m wesleygpt.release chat Wasue/WesleyGPT-SFT
```

Or from Python:

```python
from wesleygpt.release import fetch_release, load_release
model, tokenizer = load_release(fetch_release("Wasue/WesleyGPT-SFT", "wesleygpt-sft"))
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
| Supervised fine-tuning | [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) conversations, plus [MMLU](https://huggingface.co/datasets/cais/mmlu) (×3) and [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (×4) training sets, 934 steps | ~5 h, same GPU |

## Evaluation

nanochat's chat evaluation, before and after fine-tuning. The multiple-choice
tasks have four options, so 25% is chance.

| Task | Base (pretrained only) | WesleyGPT-SFT |
|---|---|---|
| ARC-Easy | 23.6% | **37.6%** |
| ARC-Challenge | 25.1% | **30.9%** |
| MMLU | 26.9% | **31.5%** |
| GSM8K (grade-school math) | 0.0% | 1.6% |
| HumanEval (Python) | 0.0% | 8.5% |
| ChatCORE | 0.002 | **0.087** |
| Knows its name (12 held-out questions × 5 samples) | – | 0% |

## Limitations

- **It makes things up.** Asked about Paris, it said the city borders the French
  Riviera.
- **It does not know its own name.** Asked who made it, it has answered
  "Google", "Microsoft", and "Siri". [WesleyGPT](https://huggingface.co/Wasue/WesleyGPT)
  is the version fine-tuned on its identity.
- Arithmetic and multi-step reasoning are close to zero (GSM8K 1.6%).
- English only. No safety tuning beyond what the fine-tuning data carries.

## License

**CC-BY-NC-4.0** for these weights: free to use, share, and adapt with credit,
but not commercially. That matches the license of NVIDIA's
[Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix),
from which the pretraining data is derived. The code is MIT.
