---
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-generation
datasets:
- karpathy/climbmix-400b-shuffle
tags:
- nanochat
- trained-from-scratch
- small-language-model
- consumer-gpu
---
<!-- Wesley wrote this -->

# WesleyGPT-Base

The pretrained foundation of [WesleyGPT](https://huggingface.co/Wasue/WesleyGPT):
a 286-million-parameter language model trained **from scratch on one RTX 3060**
in Wesley Mangum's home, in about 14 hours. Built on
[nanochat](https://github.com/karpathy/nanochat); code at
[WasueM/WesleyGPT](https://github.com/WasueM/WesleyGPT).

This is a text *continuer*, not a chat model: give it the start of a document
and it writes more. For conversation, use
[WesleyGPT](https://huggingface.co/Wasue/WesleyGPT).

## Load it

```bash
git clone https://github.com/WasueM/WesleyGPT && cd WesleyGPT
uv sync --extra cpu --extra release
```

```python
from wesleygpt.release import fetch_release, load_release
model, tokenizer = load_release(fetch_release("Wasue/WesleyGPT-Base", "wesleygpt-base"))
```

It is not a `transformers` model, so `AutoModel` will not load it.

## Model and training

| | |
|---|---|
| Parameters | 286,261,730 (12 layers, width 768, 6 heads) |
| Context | 2,048 tokens, sliding-window attention |
| Tokenizer | 32,768-token byte-level BPE |
| Data | 1.32 billion tokens of [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) (2,520 steps × 524,288 tokens) |
| Compute | ~14 h on 1× RTX 3060 12 GB |
| Validation loss | 0.844 bits per byte |
| CORE (nanochat's 22-task in-context benchmark) | 0.151 |

## License

**CC-BY-NC-4.0**, matching NVIDIA's
[Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix),
from which the pretraining data is derived. Free to use, share, and adapt with
credit, not commercially. The code is MIT.
