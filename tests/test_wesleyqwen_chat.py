# Wesley wrote this
"""Telling chat commands apart from messages to the model, and turning /video into a turn."""
import pytest

from wesleyqwen.chat import (DEFAULT_VIDEO_QUESTION, parse_command, parse_video, require_seeking_video_decoder,
                             text_only, video_turn, windowed_reply)


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


def test_text_only_history_mentions_media_instead_of_carrying_it():
    history = [{"role": "user", "content": [{"type": "video", "video": "/v.mp4"}, {"type": "text", "text": "what?"}]},
               {"role": "assistant", "content": "A talk."}]
    assert text_only(history) == [{"role": "user", "content": "[video] what?"}, {"role": "assistant", "content": "A talk."}]


def run_windows(duration, with_audio):
    seen, history = [], [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    def stream(messages, thinking):
        seen.append(messages)
        yield f"answer {len(seen)}"

    def cut(path, start, end):
        return f"/w/{start:.0f}.mp4", (f"/w/{start:.0f}.wav" if with_audio else None)

    pieces = list(windowed_reply(stream, cut, history, "/v/talk.mov", "React to this.", duration, False, str.strip))
    return seen, history, "".join(pieces)


def test_each_window_shows_the_clip_before_the_question_and_plays_the_audio_after():
    seen, _, _ = run_windows(136.3, with_audio=True)
    assert len(seen) == 5
    for messages in seen:
        kinds = [part["type"] for part in messages[-1]["content"]]
        assert kinds == ["video", "text", "audio"]


def test_each_window_knows_where_it_is_in_the_video():
    seen, _, _ = run_windows(136.3, with_audio=True)
    prompt = seen[1][-1]["content"][1]["text"]
    assert "part 2 of 5" in prompt and "0:27" in prompt


def test_earlier_windows_come_back_as_the_models_own_replies_not_as_user_text():
    seen, _, _ = run_windows(136.3, with_audio=True)
    third = seen[2]
    assert [m["role"] for m in third[2:-1]] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in third[2:-1] if m["role"] == "assistant"] == ["answer 1", "answer 2"]
    assert "answer" not in third[-1]["content"][1]["text"]


def test_windows_see_earlier_turns_but_never_earlier_media():
    seen, _, _ = run_windows(90, with_audio=True)
    assert seen[0][:2] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert all(isinstance(m["content"], str) for messages in seen for m in messages[:-1])


def test_the_reply_labels_each_window_and_history_keeps_only_text():
    _, history, text = run_windows(60, with_audio=True)
    assert text == "[0:00–0:30] answer 1\n\n[0:30–1:00] answer 2"
    assert history[-2] == {"role": "user", "content": "[video talk.mov, 1:00, watched and heard in 2 parts] React to this."}
    assert history[-1]["content"] == "[0:00–0:30] answer 1\n\n[0:30–1:00] answer 2"


def test_a_silent_video_is_watched_without_an_audio_part_and_says_so():
    seen, _, text = run_windows(20, with_audio=False)
    assert [part["type"] for part in seen[0][-1]["content"]] == ["video", "text"]
    assert text.startswith("[0:00–0:20, no audio track] ")
