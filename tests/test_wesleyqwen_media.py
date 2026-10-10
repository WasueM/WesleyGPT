# Wesley wrote this
"""Cutting a long video into the windows a model that hears audio can take."""
import shutil
import subprocess

import pytest

from wesleyqwen.media import clock, cut, probe, windows

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg and ffprobe")


def test_windows_are_equal_contiguous_and_cover_the_whole_video():
    spans = windows(136.3, 30)
    assert len(spans) == 5
    assert spans[0][0] == 0 and spans[-1][1] == pytest.approx(136.3)
    assert all(a[1] == pytest.approx(b[0]) for a, b in zip(spans, spans[1:]))
    assert all(end - start == pytest.approx(136.3 / 5) for start, end in spans)


def test_no_window_is_longer_than_the_limit():
    for duration in (29.9, 30, 30.5, 61, 600):
        assert all(end - start <= 30 + 1e-9 for start, end in windows(duration, 30))


def test_an_exact_multiple_of_the_limit_needs_no_extra_window():
    assert windows(60, 30) == [(0, 30), (30, 60)]


def test_a_video_just_over_the_limit_splits_in_half_rather_than_leaving_a_sliver():
    assert windows(30.5, 30) == [(0, 15.25), (15.25, 30.5)]


def test_a_video_without_length_is_rejected():
    with pytest.raises(ValueError, match="duration"):
        windows(0, 30)


def test_clock_reads_like_a_video_player():
    assert clock(0) == "0:00"
    assert clock(75.9) == "1:15"
    assert clock(3605) == "1:00:05"


def make_video(path, with_audio):
    inputs = ["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=4"]
    if with_audio:
        inputs += ["-f", "lavfi", "-i", "sine=frequency=440:duration=4"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-pix_fmt", "yuv420p", "-shortest", str(path)], check=True)


@needs_ffmpeg
def test_probe_reports_length_and_whether_there_is_sound(tmp_path):
    loud, silent = tmp_path / "loud.mp4", tmp_path / "silent.mp4"
    make_video(loud, with_audio=True)
    make_video(silent, with_audio=False)
    assert probe(str(loud)) == (pytest.approx(4, abs=0.2), True)
    assert probe(str(silent)) == (pytest.approx(4, abs=0.2), False)


@needs_ffmpeg
def test_a_cut_window_has_its_own_clip_and_16k_mono_audio(tmp_path):
    video = tmp_path / "loud.mp4"
    make_video(video, with_audio=True)
    clip, wav = cut(str(video), 1, 3, str(tmp_path), with_audio=True)
    assert probe(clip)[0] == pytest.approx(2, abs=0.3)
    info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", wav],
                          capture_output=True, text=True, check=True).stdout.split()
    assert info == ["16000,1"]


@needs_ffmpeg
def test_a_window_of_a_silent_video_has_no_audio(tmp_path):
    video = tmp_path / "silent.mp4"
    make_video(video, with_audio=False)
    clip, wav = cut(str(video), 0, 2, str(tmp_path), with_audio=False)
    assert wav is None and probe(clip)[0] == pytest.approx(2, abs=0.3)


def test_an_unreadable_file_names_itself(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="junk.mp4"):
        probe(str(junk))
