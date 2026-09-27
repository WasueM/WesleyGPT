---
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-generation
base_model: Wasue/WesleyGPT-Think
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
- reinforcement-learning
---
<!-- Wesley wrote this -->

# WesleyGPT-Math

[WesleyGPT-Think](https://huggingface.co/Wasue/WesleyGPT-Think) after a short
round of **reinforcement learning** on grade-school math: a 286-million-parameter
chat model trained **from scratch on one RTX 3060** in Wesley Mangum's home. It
works through a problem in a `<think>` block, then answers. RL took GSM8K from
14.6% to 16.9% in about 40 minutes of GPU time, and it came out better at
knowing its own name too. Built on
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
uv run python -m wesleygpt.release chat Wasue/WesleyGPT-Math
```

Or from Python:

```python
from wesleygpt.release import fetch_release, load_release
model, tokenizer = load_release(fetch_release("Wasue/WesleyGPT-Math", "wesleygpt-math"))
```

A real reply (temperature 0), wrong in an instructive way:

```
<think>
The bakery sells 48 muffins per tray, and it bakes 17 trays, so it sells a total of 48 x 17 = 638 muffins.
However, but only 29 muffins were sold, so the bakery sold 638 - 29 = 459 muffins.
</think>
#### 459
```

The plan is right and the arithmetic is not (48 × 17 = 816). It runs on a
laptop CPU. It is not a `transformers` model, so `AutoModel` will not load it.

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
| Pretraining ([WesleyGPT-Base](https://huggingface.co/Wasue/WesleyGPT-Base)) | 1.32 billion tokens of [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) | ~14 h, 1× RTX 3060 12 GB |
| Supervised fine-tuning ([WesleyGPT-Think](https://huggingface.co/Wasue/WesleyGPT-Think)) | conversations, multiple choice, identity, and ~270K worked math problems with the working inside `<think>…</think>` | ~5.5 h, same GPU |
| Reinforcement learning (this model) | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) training questions. For each question the model writes 16 answers; answers with the right final number are pushed up, the rest pushed down, relative to the group's average. Nothing else is rewarded: not the reasoning, not the format. 50 steps | ~40 min, same GPU |

The RL run went to 200 steps, but a quick check on 200 held-out problems
(one sampled answer each) peaked at step 50 and slid after it: 8.5% at the
start, 16.5% at step 50, 13.0% at step 150. Steps 50 and 199 were then both
given the full evaluation below, and step 50 won every row, so it is the one
released.

## Evaluation

nanochat's chat evaluation (greedy decoding; GSM8K on all 1,319 test
problems). The multiple-choice tasks have four options, so 25% is chance.

| Task | [WesleyGPT-Think](https://huggingface.co/Wasue/WesleyGPT-Think) | RL step 199 | **WesleyGPT-Math** (RL step 50) |
|---|---|---|---|
| ARC-Easy | 37.7% | 37.6% | **38.7%** |
| ARC-Challenge | 32.9% | 31.8% | 32.8% |
| MMLU | **32.0%** | 30.9% | 31.6% |
| GSM8K (grade-school math) | 14.6% | 16.5% | **16.9%** |
| HumanEval (Python) | **11.0%** | 8.5% | 10.4% |
| ChatCORE | 0.125 | 0.118 | **0.129** |
| Knows its name (12 held-out questions × 5 samples) | 45% | 70% | **80%** |

The GSM8K gain is about two points on a test with roughly one point of noise:
real, and small. The name score rose although RL never saw an identity
question; the likeliest reason is that RL makes a model's answers less
scattered, so it strays from what fine-tuning taught it less often. That is a
guess, and 60 samples is a noisy measure.

## Limitations

- **Its arithmetic is unreliable.** The reasoning usually sets up the right
  calculation and then gets a multiplication wrong. It was trained with a
  calculator tool, and the server runs one, but in spot checks it did not call it once.
- **It still gets 83% of grade-school math wrong.**
- **It makes things up** outside math, like every model in this family.
- It only thinks when a question looks like a math word problem, and not always
  then.
- It fails "You're basically Siri, right?" in every sample of the name eval.
- English only. No safety tuning beyond what the fine-tuning data carries.

## License

**CC-BY-NC-4.0** for these weights: free to use, share, and adapt with credit,
but not commercially. That matches the license of NVIDIA's
[Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix),
from which the pretraining data is derived. The code is MIT.
