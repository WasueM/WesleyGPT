# Wesley wrote this
"""Fine-tune Qwen3.5 on WesleyQwen's identity chats, three ways, on one 12 GB GPU.

    python -m wesleyqwen.train --mode full  --out runs/wesleyqwen/full
    python -m wesleyqwen.train --mode lora  --out runs/wesleyqwen/lora
    python -m wesleyqwen.train --mode qlora --out runs/wesleyqwen/qlora

full  - every language-model weight trains. A 2B model's weights, gradients and
        Adam state do not fit in 12 GB at once, so each weight is updated the
        moment its gradient is ready and the gradient is freed ("optimizer step
        in backward"), and Adam's state is kept in 8 bits. Weights stay in bf16,
        where an update smaller than half a bf16 step would round away to
        nothing; stochastic rounding keeps those updates right on average.
lora  - base frozen in bf16; low-rank adapters train, then merge into the weights.
qlora - same, but the frozen base is stored in 4 bits during training. The
        adapters are merged into the bf16 base afterwards, so every variant is
        saved, and evaluated, as an ordinary bf16 model.

The vision encoder is frozen in every mode: the data is text-only.
Same data, order, and number of steps in every mode, so the comparison is fair.
"""
import argparse
import json
import math
import os
import random
import time

import torch

from wesleygpt.identity import make_conversations
from wesleyqwen.data import collate, encode_example
from wesleyqwen.evaluate import stop_token_ids
from wesleyqwen.persona import WESLEYQWEN

BASE = os.environ.get("WESLEYQWEN_BASE", "Qwen/Qwen3.5-2B")
LORA_RANK, LORA_ALPHA = 16, 32
DEFAULT_LR = {"full": 1e-5, "lora": 1e-4, "qlora": 1e-4}
VISION_MARKER = "visual"  # every vision-encoder parameter name contains this


def load_model(mode, base=BASE):
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    quant = None
    if mode == "qlora":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda",
                                                        quantization_config=quant)
    model.config.use_cache = False
    return model


def language_linear_names(model):
    """Every Linear layer of the language model: the LoRA targets."""
    from torch import nn
    return [name for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and VISION_MARKER not in name and not name.endswith("lm_head")]


def prepare_full(model, lr):
    """Trainable language model with 8-bit Adam stepped inside backward. Returns the optimizers."""
    from torchao.optim import AdamW8bit
    optimizers = {}
    for name, p in model.named_parameters():
        p.requires_grad_(VISION_MARKER not in name)
        if not p.requires_grad or p in optimizers:
            continue  # tied weights appear once
        optimizers[p] = AdamW8bit([p], lr=lr, weight_decay=0.0, bf16_stochastic_round=True)

        def step(param):
            opt = optimizers[param]
            opt.step()
            opt.zero_grad(set_to_none=True)

        p.register_post_accumulate_grad_hook(step)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return list(optimizers.values())


def prepare_adapters(model, mode, lr):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                        target_modules=language_linear_names(model), task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    return model, [optimizer]


def lr_at(step, total, peak, warmup):
    """Linear warmup, then cosine down to 10% of peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def set_lr(optimizers, value):
    """Update in place: torchao's 8-bit Adam holds lr as a tensor and rejects a float swapped in."""
    for opt in optimizers:
        for group in opt.param_groups:
            if isinstance(group["lr"], torch.Tensor):
                group["lr"].fill_(value)
            else:
                group["lr"] = value


def batches(examples, batch_size, epochs, seed):
    rng = random.Random(seed)
    for _ in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            yield [examples[j] for j in order[i:i + batch_size]]


def save(model, mode, out, stop_ids):
    """Write a plain bf16 model directory, whatever the mode trained, that stops at the end of a turn."""
    model.generation_config.eos_token_id = stop_ids
    if mode == "full":
        model.save_pretrained(out)
        return
    model.save_pretrained(os.path.join(out, "adapter"))
    if mode == "qlora":
        # Merge into the bf16 base, not the 4-bit copy the adapters trained against.
        from peft import PeftModel
        del model
        torch.cuda.empty_cache()
        base = load_model("lora")
        model = PeftModel.from_pretrained(base, os.path.join(out, "adapter"))
    merged = model.merge_and_unload()
    merged.generation_config.eos_token_id = stop_ids
    merged.save_pretrained(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["full", "lora", "qlora"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--conversations", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None, help="stop early (memory-fit checks)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    torch.manual_seed(args.seed)
    lr = args.lr or DEFAULT_LR[args.mode]
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    chats = make_conversations(args.conversations, seed=args.seed, persona=WESLEYQWEN)
    examples = [encode_example(tokenizer, c["messages"]) for c in chats]
    total = (len(examples) // args.batch_size) * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)

    model = load_model(args.mode)
    if args.mode == "full":
        optimizers = prepare_full(model, lr)
    else:
        model, optimizers = prepare_adapters(model, args.mode, lr)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"mode={args.mode} trainable={trainable:,} steps={total} lr={lr}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "w")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for step, batch in enumerate(batches(examples, args.batch_size, args.epochs, args.seed)):
        if step >= total:
            break
        set_lr(optimizers, lr_at(step, total, lr, args.warmup))
        tensors = {k: v.cuda() for k, v in collate(batch, tokenizer.pad_token_id).items()}
        loss = model(**tensors).loss
        loss.backward()  # in full mode, every weight has already been updated when this returns
        if args.mode != "full":
            optimizers[0].step()
            optimizers[0].zero_grad(set_to_none=True)
        record = {"step": step, "loss": round(loss.item(), 4), "lr": lr_at(step, total, lr, args.warmup),
                  "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                  "elapsed_s": round(time.time() - start, 1)}
        log.write(json.dumps(record) + "\n")
        log.flush()
        if step % 10 == 0 or step == total - 1:
            print(record, flush=True)
    log.close()
    summary = {"mode": args.mode, "base": BASE, "steps": total, "lr": lr, "trainable": trainable,
               "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "train_seconds": round(time.time() - start, 1)}
    with open(os.path.join(args.out, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(summary, flush=True)
    if not args.no_save:
        model.eval()
        save(model, args.mode, args.out, stop_token_ids(tokenizer))
        tokenizer.save_pretrained(args.out)
        print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
