# Wesley wrote this
"""Turning a model's per-window JSON into one transcript of everything said and shown."""
import json

import pytest

from wesleyqwen.transcript import assemble, parse_window, place, seconds, transcribe


def test_seconds_reads_plain_numbers_and_player_clocks():
    assert seconds(12) == 12.0
    assert seconds("12.5") == 12.5
    assert seconds("12.5s") == 12.5
    assert seconds("1:05") == 65.0
    assert seconds("0:01:07.5") == 67.5


def test_seconds_rejects_a_time_it_cannot_read_by_quoting_it():
    with pytest.raises(ValueError, match="0m0s03s164ms"):
        seconds("0m0s03s164ms")


def test_a_window_reply_wrapped_in_a_json_fence_still_parses():
    reply = '```json\n{"speech": [{"start": 1, "end": 2, "text": "Hi."}], "shown": [{"at": 0, "what": "Big Ben"}]}\n```'
    assert parse_window(reply) == {"speech": [{"start": 1.0, "end": 2.0, "text": "Hi."}],
                                   "shown": [{"at": 0.0, "what": "Big Ben"}]}


def test_prose_around_the_json_is_ignored():
    reply = 'Here you go:\n{"speech": [], "shown": [{"at": "0:03", "what": "a train"}]}\nHope that helps.'
    assert parse_window(reply)["shown"] == [{"at": 3.0, "what": "a train"}]


def test_a_second_json_object_after_the_first_is_read_too():
    reply = '{"speech": [{"start": 1, "text": "Hi."}], "shown": []}\n{"speech": [], "shown": [{"at": 2, "what": "a train"}]}'
    assert parse_window(reply) == {"speech": [{"start": 1.0, "end": 1.0, "text": "Hi."}],
                                   "shown": [{"at": 2.0, "what": "a train"}]}


def test_a_spoken_line_without_an_end_ends_where_it_starts():
    assert parse_window('{"speech": [{"start": 4, "text": "Amen."}], "shown": []}')["speech"] == [
        {"start": 4.0, "end": 4.0, "text": "Amen."}]


def test_a_thing_listed_over_and_over_is_kept_once_at_its_first_time():
    reply = json.dumps({"speech": [], "shown": [{"at": 0, "what": "black screen"}, {"at": 1, "what": "Black screen."},
                                                {"at": 2, "what": "Big Ben"}, {"at": 3, "what": "black screen"}]})
    assert parse_window(reply)["shown"] == [{"at": 0.0, "what": "black screen"}, {"at": 2.0, "what": "Big Ben"}]


def test_a_reply_missing_a_field_is_rejected_by_name():
    with pytest.raises(ValueError, match="shown"):
        parse_window('{"speech": []}')


def test_a_reply_with_no_json_is_rejected():
    with pytest.raises(ValueError, match="no JSON"):
        parse_window("I could not hear anything.")


def test_place_moves_clip_times_onto_the_video_clock_and_keeps_them_inside_the_window():
    window = {"speech": [{"start": 2, "end": 40, "text": "a"}], "shown": [{"at": 1.04, "what": "b"}]}
    assert place(window, 26.2, 52.4) == {"speech": [{"start": 28.2, "end": 52.4, "text": "a"}],
                                         "shown": [{"at": 27.2, "what": "b"}]}


def test_assemble_joins_every_spoken_line_in_order():
    first = {"speech": [{"start": 1, "end": 3, "text": "I was invited."}], "shown": [{"at": 0, "what": "Big Ben"}]}
    second = {"speech": [{"start": 30, "end": 33, "text": "My passport was expired."}], "shown": []}
    out = assemble("talk.mov", 60, "gemma", [first, second], [])
    assert out["words_spoken"] == "I was invited. My passport was expired."
    assert out["video"] == "talk.mov" and out["duration_seconds"] == 60 and out["model"] == "gemma"
    assert [line["text"] for line in out["speech"]] == ["I was invited.", "My passport was expired."]
    assert out["shown"] == [{"at": 0, "what": "Big Ben"}]


def walk(replies, duration=52.4, with_audio=True, tmp_path=None):
    """Run transcribe with a fake model that answers from replies, one per call."""
    sent, history = [], []
    replies = iter(replies)

    def stream(messages, thinking):
        sent.append((messages, thinking))
        yield next(replies)

    def cut(path, start, end):
        return f"/w/{start:.0f}.mp4", (f"/w/{start:.0f}.wav" if with_audio else None)

    save = str(tmp_path / "talk.transcript.json") if tmp_path else None
    text = "".join(transcribe(stream, cut, history, "/v/talk.mov", duration, "gemma", save))
    return sent, history, text


GOOD = '{"speech": [{"start": 1, "end": 2, "text": "Hello."}], "shown": [{"at": 0, "what": "a title card"}]}'


def test_each_window_is_asked_alone_with_frames_then_prompt_then_audio_and_no_thinking():
    sent, _, _ = walk([GOOD, GOOD])
    assert len(sent) == 2
    for messages, thinking in sent:
        assert len(messages) == 1 and thinking is False
        assert [part["type"] for part in messages[0]["content"]] == ["video", "text", "audio"]


def test_a_window_that_answers_with_bad_json_is_asked_once_more_saying_why():
    sent, _, text = walk(["not json", GOOD, GOOD])
    assert len(sent) == 3
    assert "unreadable reply (no JSON object" in text
    assert '"words_spoken": "Hello. Hello."' in text


def test_a_window_that_fails_twice_is_a_named_gap_and_the_rest_is_still_transcribed():
    _, _, text = walk(["nope", "still nope", GOOD])
    assert "could not read 0:00–0:26" in text
    out = json.loads(text[text.index("{"):text.rindex("}") + 1])
    assert out["words_spoken"] == "Hello."
    assert out["unreadable_windows"] == [{"start": 0.0, "end": 26.2, "problem": "no JSON object in the reply: 'still nope'"}]


def test_a_transcript_with_no_gaps_says_so():
    _, _, text = walk([GOOD, GOOD])
    assert '"unreadable_windows": []' in text


def test_the_transcript_is_saved_as_json(tmp_path):
    _, _, text = walk([GOOD, GOOD], tmp_path=tmp_path)
    saved = json.loads((tmp_path / "talk.transcript.json").read_text())
    assert [line["start"] for line in saved["speech"]] == [1.0, 27.2]
    assert str(tmp_path / "talk.transcript.json") in text


def test_the_conversation_keeps_the_transcript_as_a_timeline_not_as_json():
    # Kept as JSON, it taught Gemma to answer the next question in JSON.
    _, history, _ = walk([GOOD, GOOD])
    assert history[0]["content"] == "[transcript of video talk.mov, 0:52]"
    assert history[1]["content"] == ("0:00 shown: a title card\n0:01 said: Hello.\n"
                                     "0:26 shown: a title card\n0:27 said: Hello.")


def test_progress_is_reported_per_window_before_the_json():
    _, _, text = walk([GOOD, GOOD])
    assert text.index("[0:00–0:26]") < text.index("[0:26–0:52]") < text.index('"words_spoken"')


def test_a_silent_video_is_transcribed_from_frames_alone():
    sent, _, _ = walk(['{"speech": [], "shown": []}'], duration=20, with_audio=False)
    assert [part["type"] for part in sent[0][0][0]["content"]] == ["video", "text"]
