# Wesley wrote this
"""Probe: does the real Qwen3.5-2B match what train.py assumes?  python -m wesleyqwen.probe"""
import collections, torch, transformers
from transformers import AutoTokenizer, AutoModelForImageTextToText
from wesleygpt.identity import make_conversations
from wesleyqwen.data import encode_example, IGNORE
from wesleyqwen.persona import WESLEYQWEN
from wesleyqwen.train import BASE, language_linear_names, VISION_MARKER
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), "tf", transformers.__version__)
try:
    import fla; print("fla", fla.__version__)
except Exception as e: print("fla import FAILED", e)
tok = AutoTokenizer.from_pretrained(BASE)
ex = encode_example(tok, make_conversations(1, seed=0, persona=WESLEYQWEN)[0]["messages"])
print("pad", tok.pad_token_id, tok.pad_token)
print("FULL:", repr(tok.decode(ex["input_ids"])))
print("GRADED:", repr(tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != IGNORE])))
m = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
print(type(m).__name__, "gpu GB", round(torch.cuda.memory_allocated()/2**30, 2))
counts = collections.Counter()
for n, p in m.named_parameters():
    counts["vision" if VISION_MARKER in n else "language"] += p.numel()
print({k: f"{v/1e9:.3f}B" for k, v in counts.items()})
names = language_linear_names(m)
print(len(names), "lora targets; kinds:", sorted({n.split(".")[-1] for n in names}))
print("tied:", m.get_output_embeddings().weight.data_ptr() == m.get_input_embeddings().weight.data_ptr())
