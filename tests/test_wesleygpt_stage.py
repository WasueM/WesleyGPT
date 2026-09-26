# Wesley wrote this
"""Which checkpoint files a server image needs for a given models.json."""
from wesleygpt.stage import files_for

SFT = {"id": "chat", "description": "", "source": "sft", "model_tag": "d12", "step": 934}


def test_needs_tokenizer_and_the_model_weights_and_meta_but_not_the_optimizer():
    assert files_for([SFT]) == [
        "chatsft_checkpoints/d12/meta_000934.json",
        "chatsft_checkpoints/d12/model_000934.pt",
        "tokenizer/token_bytes.pt",
        "tokenizer/tokenizer.pkl",
    ]


def test_each_source_maps_to_its_nanochat_directory_without_duplicates():
    base = {**SFT, "id": "base", "source": "base", "step": 2520}
    rl = {**SFT, "id": "rl", "source": "rl", "step": 60}
    files = files_for([SFT, base, rl, SFT])
    assert "base_checkpoints/d12/model_002520.pt" in files
    assert "chatrl_checkpoints/d12/model_000060.pt" in files
    assert len(files) == len(set(files)) == 8


def test_hf_download_asks_only_for_the_needed_files(tmp_path):
    from wesleygpt.stage import stage_from_hf
    wanted = files_for([SFT])
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        for f in kwargs["allow_patterns"]:
            (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / f).write_bytes(b"x")

    stage_from_hf("Wasue/wesleygpt-checkpoints", wanted, tmp_path, download=fake_download)
    assert calls[0]["repo_id"] == "Wasue/wesleygpt-checkpoints"
    assert sorted(calls[0]["allow_patterns"]) == wanted  # never the 1.2 GB optimizer
    assert calls[0]["local_dir"] == tmp_path


def test_hf_download_fails_loudly_when_the_repo_lacks_a_file(tmp_path):
    import pytest
    from wesleygpt.stage import stage_from_hf
    with pytest.raises(SystemExit, match="model_000934.pt"):
        stage_from_hf("Wasue/x", files_for([SFT]), tmp_path, download=lambda **kw: None)
