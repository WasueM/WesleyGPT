# Wesley wrote this
"""Where browser uploads land, and which uploads a chat turn may point at."""
import pytest

from wesleyqwen.web import upload_name, uploaded_video


def test_an_upload_keeps_its_name_behind_a_unique_prefix():
    first, second = upload_name("clip.mp4"), upload_name("clip.mp4")
    assert first.endswith("-clip.mp4") and second.endswith("-clip.mp4")
    assert first != second


def test_an_upload_name_cannot_climb_out_of_the_upload_folder():
    name = upload_name("../../.bashrc")
    assert name.endswith("-.bashrc") and "/" not in name


def test_an_upload_name_drops_characters_a_shell_would_trip_on():
    assert upload_name("my clip (1).mp4").endswith("-my_clip__1_.mp4")


def test_an_uploaded_video_resolves_inside_the_upload_folder(tmp_path):
    (tmp_path / "abc-clip.mp4").write_bytes(b"")
    assert uploaded_video(str(tmp_path), "abc-clip.mp4") == str(tmp_path / "abc-clip.mp4")


def test_a_video_id_that_escapes_the_upload_folder_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not an uploaded video"):
        uploaded_video(str(tmp_path), "../secret.mp4")


def test_a_video_id_that_was_never_uploaded_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="gone.mp4"):
        uploaded_video(str(tmp_path), "gone.mp4")
