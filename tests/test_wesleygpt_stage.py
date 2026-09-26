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
