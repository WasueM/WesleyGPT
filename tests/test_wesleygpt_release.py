# Wesley wrote this
"""The public Hugging Face release: safetensors weights plus a text tokenizer.

A tiny model and tokenizer are built in-process, so nothing here needs a real
checkpoint or the network.
"""
import json

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer
from wesleygpt.release import RELEASE_FILES, export_release, fetch_release, load_release

CORPUS = ["hello world, hello WesleyGPT", "def f(x):\n    return x + 1\n", "naïve café 你好 🙂"] * 8
TINY = {"sequence_len": 32, "vocab_size": 256 + len(SPECIAL_TOKENS) + 35, "n_layer": 2,
        "n_head": 2, "n_kv_head": 2, "n_embd": 64, "window_pattern": "L"}


@pytest.fixture(scope="module")
def tokenizer():
    return RustBPETokenizer.train_from_iterator(iter(CORPUS), TINY["vocab_size"])


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = GPT(GPTConfig(**TINY))
    m.init_weights()
    return m.eval()


def _export(tmp_path, model, tokenizer, meta_extra=None):
    # Checkpoints from a torch.compile'd run carry this prefix on every key.
    state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    meta = {"step": 7, "model_config": dict(TINY), **(meta_extra or {})}
    export_release(state, meta, tokenizer.enc, tmp_path)
    return tmp_path


def test_round_trip_reproduces_the_models_outputs(tmp_path, model, tokenizer):
    loaded, loaded_tok = load_release(_export(tmp_path, model, tokenizer))
    text = "hello WesleyGPT 🙂"
    assert loaded_tok.encode(text) == tokenizer.encode(text)
    assert loaded_tok.get_bos_token_id() == tokenizer.get_bos_token_id()
    ids = torch.tensor([[tokenizer.get_bos_token_id(), *tokenizer.encode(text)]])
    with torch.inference_mode():
        assert torch.equal(loaded(ids), model(ids))


def test_release_holds_only_non_executable_formats(tmp_path, model, tokenizer):
    out = _export(tmp_path, model, tokenizer)
    # No pickle (.pkl/.pt/.bin): loading one runs arbitrary code on the downloader's machine.
    expected = {"config.json", "model.safetensors", "tokenizer.tiktoken", "tokenizer_config.json"}
    assert {p.name for p in out.iterdir()} == expected == set(RELEASE_FILES)


def test_config_carries_the_architecture_but_not_the_training_run(tmp_path, model, tokenizer):
    out = _export(tmp_path, model, tokenizer, {"user_config": {"run": "dummy"}, "dataloader_state_dict": {"pq_idx": 29}})
    config = json.loads((out / "config.json").read_text())
    assert config["model_config"] == TINY
    assert "user_config" not in config and "dataloader_state_dict" not in config


def test_fetch_names_files_the_download_did_not_deliver(tmp_path):
    def fake_download(repo_id, allow_patterns, local_dir):
        (local_dir / "config.json").write_text("{}")
        return str(local_dir)

    with pytest.raises(SystemExit, match="model.safetensors"):
        fetch_release("Wasue/WesleyGPT", tmp_path, download=fake_download)
