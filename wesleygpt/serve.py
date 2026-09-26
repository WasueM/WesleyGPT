# Wesley wrote this
"""Entry point: python -m wesleygpt.serve

Configuration (environment):
  WESLEYGPT_API_KEYS     required; comma-separated bearer keys (Secret Manager on Cloud Run)
  NANOCHAT_BASE_DIR      where checkpoints + tokenizer live (nanochat layout)
  WESLEYGPT_MODELS_FILE  model list (default: wesleygpt/models.json)
  WESLEYGPT_THREADS      torch CPU threads; set it explicitly on Cloud Run, where
                         os.cpu_count() reports the host's CPUs, not the vCPU limit
  WESLEYGPT_MAX_TOKENS_CAP / WESLEYGPT_DEFAULT_MAX_TOKENS   reply length limits
  PORT                   HTTP port (Cloud Run sets it)
"""
import logging
import os

import torch
import uvicorn

from wesleygpt.api_schema import Limits
from wesleygpt.auth import parse_api_keys
from wesleygpt.runtime import DEFAULT_MODELS_FILE, NanochatRuntime, read_model_specs
from wesleygpt.server import create_app


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    api_keys = parse_api_keys(os.environ.get("WESLEYGPT_API_KEYS"))
    threads = os.environ.get("WESLEYGPT_THREADS")
    if threads:
        torch.set_num_threads(int(threads))
    limits = Limits(default_max_tokens=int(os.environ.get("WESLEYGPT_DEFAULT_MAX_TOKENS", "256")),
                    max_tokens_cap=int(os.environ.get("WESLEYGPT_MAX_TOKENS_CAP", "512")))
    specs = read_model_specs(os.environ.get("WESLEYGPT_MODELS_FILE", DEFAULT_MODELS_FILE))
    app = create_app(NanochatRuntime(specs, device="cpu"), api_keys, limits)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), log_level="warning")


if __name__ == "__main__":
    main()
