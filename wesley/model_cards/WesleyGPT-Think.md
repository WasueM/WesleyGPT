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
---
<!-- Wesley wrote this -->

# WesleyGPT-Think

A 286-million-parameter chat model trained **from scratch on one RTX 3060** in
Wesley Mangum's home that works through math problems step by step in a
`<think>` block before answering: about 14 hours of pretraining, then about
5.5 hours of fine-tuning. The thinking took grade-school math (GSM8K) from
1% to 14.6%. Built on
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
uv run python -m wesleygpt.release chat Wasue/WesleyGPT-Think
```

Or from Python:

```python
from wesleygpt.release import fetch_release, load_release
model, tokenizer = load_release(fetch_release("Wasue/WesleyGPT-Think", "wesleygpt-think"))
```

Replies to math word problems look like this:

```
<think>
Tom has 3 boxes with 12 apples each, so he has a total of 3 * 12 = 36 apples.
He eats 5 apples, so he now has 36 - 5 = 31 apples left.
</think>
#### 31
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
| Supervised fine-tuning | [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) conversations, plus [MMLU](https://huggingface.co/datasets/cais/mmlu) (×3) and [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (×4) training sets, plus a small slice of synthetic conversations about its own identity. The GSM8K solutions are rewritten so the working sits in `<think>…</think>` before a `#### N` answer, and ~240K [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) problems (rephrasings of GSM8K's *training* set) are added in the same format. 1,038 steps | ~5.5 h, same GPU |

## Evaluation

nanochat's chat evaluation, before and after fine-tuning. The multiple-choice
tasks have four options, so 25% is chance.

| Task | Base (pretrained only) | [WesleyGPT](https://huggingface.co/Wasue/WesleyGPT) | WesleyGPT-Think |
|---|---|---|---|
| ARC-Easy | 23.6% | 36.8% | **37.7%** |
| ARC-Challenge | 25.1% | 32.3% | **32.9%** |
| MMLU | 26.9% | 31.8% | **32.0%** |
| GSM8K (grade-school math) | 0.0% | 1.1% | **14.6%** |
| HumanEval (Python) | 0.0% | 9.2% | **11.0%** |
| ChatCORE | 0.002 | 0.090 | **0.125** |
| Knows its name (12 held-out questions × 5 samples) | – | 28% | 45% |

## Limitations

- **It makes things up.** Asked about Paris, it said the city borders the French
  Riviera.
- **It only thinks when a question looks like a math word problem.** Every
  thinking example it trained on was one; asked "What is 17 times 3?" it
  answered conversationally, and wrongly.
- The final answer is bare (`#### 31`), the format it was trained on.
- It still gets 85% of grade-school math wrong.
- It only sometimes knows its own name (45% of unseen phrasings; this may be
  within noise of the 28% of WesleyGPT, since the eval has only 60 samples).
- English only. No safety tuning beyond what the fine-tuning data carries.

## License

**CC-BY-NC-4.0** for these weights: free to use, share, and adapt with credit,
but not commercially. That matches the license of NVIDIA's
[Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix),
from which the pretraining data is derived. The code is MIT.
