# Wesley wrote this
"""Loads nanochat checkpoints and runs generation for the HTTP server.

Checkpoints are read from $NANOCHAT_BASE_DIR in nanochat's own layout
(chatsft_checkpoints/d12/model_000934.pt, tokenizer/, ...). Each model has a lock:
one generation at a time per model per process. On Cloud Run that pairs with
--concurrency=1, so extra requests go to extra instances instead of queueing here.
"""
import gc
import json
import random
import threading
from collections import OrderedDict
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
    required = {"id", "summary", "description", "source", "model_tag", "step"}
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
    """Models load on first use; past `max_resident`, the least recently used one is
    dropped. Each held as float32 costs ~1.15 GB, and four at once overflowed Cloud
    Run's 8 GiB, so the server offers every model but holds only a few. The first
    spec is loaded up front: it is the default, and the container starts faster."""

    def __init__(self, specs, device="cpu", loader=load_from_base_dir, max_resident=2):
        if max_resident < 1:
            raise ValueError(f"max_resident must be at least 1, got {max_resident}")
        # `loader` lets a Hugging Face release (wesleygpt.release) serve through the same path.
        self.device, self._loader, self._max_resident = torch.device(device), loader, max_resident
        self.models = [{"id": s["id"], "summary": s.get("summary", ""), "description": s["description"]}
                       for s in specs]
        self.aliases = {alias: s["id"] for s in specs for alias in s.get("aliases", [])}
        self._specs = {s["id"]: s for s in specs}
        self._loaded = OrderedDict()  # least recently used first
        self._loading = threading.Lock()
        self.ensure_loaded(specs[0]["id"])

    def resident(self):
        return list(self._loaded)

    def ensure_loaded(self, model_id):
        with self._loading:
            if model_id in self._loaded:
                self._loaded.move_to_end(model_id)
                return self._loaded[model_id]
            # Evict before loading, so the peak is max_resident models plus none.
            idle = [k for k, m in self._loaded.items() if not m.lock.locked()]
            for victim in idle[:max(0, len(self._loaded) - self._max_resident + 1)]:
                del self._loaded[victim]
            gc.collect()
            self._loaded[model_id] = _Loaded(*self._loader(self._specs[model_id], self.device))
            return self._loaded[model_id]

    def generate(self, req):
        """Returns (prompt_token_count, iterator of DecodeEvent). Raises PromptError up front."""
        m, spec = self.ensure_loaded(req.model), self._specs[req.model]
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
