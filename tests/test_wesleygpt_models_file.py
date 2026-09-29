# Wesley wrote this
"""models.json: the served family, and the checks read_model_specs makes on it."""
import json

import pytest

from wesleygpt.runtime import DEFAULT_MODELS_FILE, read_model_specs


def write(tmp_path, specs):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(specs))
    return path


def spec(id_, **extra):
    return {"id": id_, "summary": "", "description": "", "source": "sft", "model_tag": "d12", "step": 1, **extra}


def test_an_alias_may_not_shadow_a_real_model_id(tmp_path):
    with pytest.raises(ValueError, match="wesleygpt-b"):
        read_model_specs(write(tmp_path, [spec("wesleygpt-a", aliases=["wesleygpt-b"]), spec("wesleygpt-b")]))


def test_an_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        read_model_specs(write(tmp_path, [spec("wesleygpt-a", mode="telepathy")]))


def test_the_old_chat_id_still_resolves_to_the_identity_model():
    specs = read_model_specs(DEFAULT_MODELS_FILE)
    owner = [s["id"] for s in specs if "wesleygpt-d12-chat" in s.get("aliases", [])]
    assert owner == ["wesleygpt-d12-identity"]


def test_a_spec_without_a_summary_is_rejected(tmp_path):
    bare = spec("wesleygpt-a")
    del bare["summary"]
    with pytest.raises(ValueError, match="summary"):
        read_model_specs(write(tmp_path, [bare]))


def test_every_served_summary_is_a_short_phrase():
    # The picker shows the summary on one line; the full description sits behind an info button.
    long = {s["id"]: s["summary"] for s in read_model_specs(DEFAULT_MODELS_FILE) if not 0 < len(s["summary"]) <= 40}
    assert long == {}
