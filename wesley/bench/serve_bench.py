# Wesley wrote this
# Benchmark: VRAM footprint and tokens/sec of the d12 SFT chat model
# on a given device / dtype / thread count. One config per process because
# nanochat fixes COMPUTE_DTYPE at import time.
import argparse, json, subprocess, time
import torch
from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda")
p.add_argument("--threads", type=int, default=0)
p.add_argument("--max-tokens", type=int, default=200)
args = p.parse_args()
if args.threads:
    torch.set_num_threads(args.threads)

PROMPTS = [
    "Why is the sky blue?",
    "Write a short story about a robot who learns to paint.",
    "Explain how a car engine works, step by step, in detail.",
]

device = torch.device(args.device)
t0 = time.perf_counter()
model, tok, meta = load_model("sft", device, phase="eval", model_tag="d12", step=934)
load_s = time.perf_counter() - t0
after_load = torch.cuda.memory_allocated() if args.device == "cuda" else 0
def smi():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
smi_loaded_idle = smi() if args.device == "cuda" else None
eng = Engine(model, tok)
bos = tok.get_bos_token_id()
us, ue = tok.encode_special("<|user_start|>"), tok.encode_special("<|user_end|>")
ast = tok.encode_special("<|assistant_start|>")

def run(prompt, max_tokens):
    ids = [bos, us] + tok.encode(prompt) + [ue, ast]
    t = time.perf_counter(); first = None; out = []
    for col, _ in eng.generate(ids, num_samples=1, max_tokens=max_tokens, temperature=0.6, top_k=50):
        if first is None:
            if args.device == "cuda": torch.cuda.synchronize()
            first = time.perf_counter()
        out.append(col[0])
    if args.device == "cuda": torch.cuda.synchronize()
    end = time.perf_counter()
    return tok.decode(out), len(out), first - t, (len(out) - 1) / (end - first)

run("hi", 8)  # warmup
res = []
for pr in PROMPTS:
    text, n, ttft, tps = run(pr, args.max_tokens)
    res.append({"prompt": pr, "tokens": n, "ttft_s": round(ttft, 3), "decode_tok_s": round(tps, 1), "text": text[:160]})

# Worst case for memory: a prompt that fills most of the 2048-token context.
long_ids_prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * 180)
_, n_long, ttft_long, tps_long = run(long_ids_prompt, 64)

report = {"device": args.device, "dtype": str(COMPUTE_DTYPE), "threads": torch.get_num_threads(),
          "load_s": round(load_s, 1), "long_prompt_ttft_s": round(ttft_long, 3), "long_prompt_decode_tok_s": round(tps_long, 1),
          "runs": res}
if args.device == "cuda":
    report["torch_alloc_after_load_MiB"] = round(after_load / 2**20)
    report["torch_peak_alloc_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20)
    report["torch_peak_reserved_MiB"] = round(torch.cuda.max_memory_reserved() / 2**20)
    report["nvidia_smi_used_MiB_loaded_idle"] = smi_loaded_idle
    report["nvidia_smi_used_MiB_after_generation"] = smi()
print(json.dumps(report, indent=1))
