# Wesley wrote this
"""Telling chat commands apart from messages to the model, and turning /video into a turn."""
import pytest

from wesleyqwen.chat import (DEFAULT_VIDEO_QUESTION, parse_command, parse_video, require_seeking_video_decoder,
                             video_turn)


def test_a_slash_word_is_a_command_with_its_argument():
    assert parse_command("/model full") == ("model", "full")
    assert parse_command("  /think  ") == ("think", "")


def test_ordinary_text_is_a_message():
    assert parse_command("What is your name?") is None
    assert parse_command("and/or both") is None


def test_video_argument_is_a_path_then_the_question():
    assert parse_video("/mnt/c/clip.mp4 what color is the car?") == ("/mnt/c/clip.mp4", "what color is the car?")


def test_a_quoted_video_path_may_contain_spaces():
    assert parse_video('"/mnt/c/My Videos/clip.mp4" who is talking') == ("/mnt/c/My Videos/clip.mp4", "who is talking")


def test_a_video_with_no_question_gets_the_default_one():
    assert parse_video("clip.mp4") == ("clip.mp4", DEFAULT_VIDEO_QUESTION)


def test_video_with_no_path_is_rejected():
    with pytest.raises(ValueError, match="usage: /video"):
        parse_video("   ")


def test_a_missing_video_file_is_rejected_by_name(tmp_path):
    missing = tmp_path / "nope.mp4"
    with pytest.raises(FileNotFoundError, match="nope.mp4"):
        video_turn(str(missing), "what happens?")


def test_a_video_turn_carries_the_video_before_the_question(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    assert video_turn(str(clip), "what happens?") == {
        "role": "user",
        "content": [{"type": "video", "video": str(clip)}, {"type": "text", "text": "what happens?"}]}


def test_video_without_torchcodec_is_refused_by_name(monkeypatch):
    import transformers.video_processing_utils as video_processing
    monkeypatch.setattr(video_processing, "is_torchcodec_available", lambda: False)
    with pytest.raises(RuntimeError, match="torchcodec"):
        require_seeking_video_decoder()


def test_video_with_torchcodec_is_allowed(monkeypatch):
    import transformers.video_processing_utils as video_processing
    monkeypatch.setattr(video_processing, "is_torchcodec_available", lambda: True)
    require_seeking_video_decoder()
