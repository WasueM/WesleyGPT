# Wesley wrote this
"""Loads nanochat checkpoints and runs generation for the HTTP server.

Checkpoints are read from $NANOCHAT_BASE_DIR in nanochat's own layout
(chatsft_checkpoints/d12/model_000934.pt, tokenizer/, ...). Each model has a lock:
one generation at a time per model per process. On Cloud Run that pairs with
--concurrency=1, so extra requests go to extra instances instead of queueing here.
"""
import json
import random
import threading
from pathlib import Path

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from wesleygpt.decode import DecodeParams, decode_loop
from wesleygpt.prompt import render_chat_prompt, render_completion_prompt
from wesleygpt.reasoning import split_reasoning

DEFAULT_MODELS_FILE = Path(__file__).with_name("models.json")


MODES = {"chat", "completion"}


def read_model_specs(path):
    """Model specs from a models.json. Optional keys: `mode` ("chat", or "completion" for
    a base model that only continues text), `reasoning` (it thinks in <think> blocks),
    `aliases` (old ids that still reach it)."""
    specs = json.loads(Path(path).read_text())
    required = {"id", "description", "source", "model_tag", "step"}
    ids = {spec.get("id") for spec in specs}
    for spec in specs:
        missing = required - spec.keys()
        if missing:
            raise ValueError(f"model spec {spec.get('id', spec)} in {path} is missing {sorted(missing)}")
        if spec.get("mode", "chat") not in MODES:
            raise ValueError(f"model spec {spec['id']} in {path} has mode {spec['mode']!r}, expected one of {sorted(MODES)}")
        shadowed = ids & set(spec.get("aliases", []))
        if shadowed:
            raise ValueError(f"model spec {spec['id']} in {path} has aliases that are real model ids: {sorted(shadowed)}")
    return specs


class _Loaded:
    def __init__(self, model, tok):
        self.model, self.tok, self.lock = model, tok, threading.Lock()


def load_from_base_dir(spec, device):
    """(model, tokenizer) for a models.json spec, from NANOCHAT_BASE_DIR."""
    model, tok, _ = load_model(spec["source"], device, phase="eval", model_tag=spec["model_tag"], step=spec["step"])
    return model, tok


class NanochatRuntime:
    def __init__(self, specs, device="cpu", loader=load_from_base_dir):
        # `loader` lets a Hugging Face release (wesleygpt.release) serve through the same path.
        self.device = torch.device(device)
        self.models = [{"id": s["id"], "description": s["description"]} for s in specs]
        self.aliases = {alias: s["id"] for s in specs for alias in s.get("aliases", [])}
        self._specs = {s["id"]: s for s in specs}
        self._loaded = {s["id"]: _Loaded(*loader(s, self.device)) for s in specs}

    def generate(self, req):
        """Returns (prompt_token_count, iterator of DecodeEvent). Raises PromptError up front."""
        m, spec = self._loaded[req.model], self._specs[req.model]
        seq_len = m.model.config.sequence_len
        render = render_completion_prompt if spec.get("mode") == "completion" else render_chat_prompt
        prompt = render(req.messages, m.tok, max_prompt_tokens=seq_len - req.max_tokens)
        params = DecodeParams(
            max_tokens=req.max_tokens, temperature=req.temperature, top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            seed=req.seed if req.seed is not None else random.randrange(2**31),
            max_total_tokens=seq_len - len(prompt),
        )
        events = self._events(m, prompt, params)
        return len(prompt), split_reasoning(events) if spec.get("reasoning") else events

    def _events(self, m, prompt, params):
        cfg = m.model.config
        with m.lock:
            kv = KVCache(batch_size=1, num_heads=cfg.n_kv_head, seq_len=cfg.sequence_len,
                         head_dim=cfg.n_embd // cfg.n_head, num_layers=cfg.n_layer,
                         device=self.device, dtype=COMPUTE_DTYPE)  # same as nanochat's Engine

            # inference_mode is per-thread and the server may resume this generator on a
            # different worker thread for every token, so each forward enters it itself.
            def forward_last(ids):
                with torch.inference_mode():
                    x = torch.tensor([ids], dtype=torch.long, device=self.device)
                    return m.model.forward(x, kv_cache=kv)[:, -1, :]

            yield from decode_loop(forward_last(prompt), lambda token: forward_last([token]), m.tok, params)
