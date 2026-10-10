# Wesley wrote this
"""Cut a long video into the windows a model that hears audio can take, with ffmpeg.

Gemma 4 E2B hears at most 30 s of audio per clip, so a longer video is walked in
windows: each one is a short re-encoded clip for the frames plus a 16 kHz mono WAV
for the sound, the rate its feature extractor reads.
"""
import json
import math
import os
import subprocess

# The processor shrinks every frame to 70 tokens anyway; a smaller cut just encodes faster.
CLIP_HEIGHT = 480
AUDIO_RATE = 16000


def windows(duration, longest):
    """Equal, contiguous (start, end) spans covering duration, none longer than longest seconds.

    Equal spans rather than fixed 30 s ones, so a 30.5 s video is two 15 s halves and
    never a 0.5 s sliver too short to sample frames from.
    """
    if duration <= 0:
        raise ValueError(f"video duration must be positive, got {duration}")
    count = math.ceil(duration / longest)
    return [(duration * i / count, duration * (i + 1) / count) for i in range(count)]


def clock(seconds):
    """'1:15' for 75.9 s, '1:00:05' past an hour: how a video player labels a position."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def run(command, what):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{what} failed: {result.stderr.strip() or f'exit {result.returncode}'}")
    return result.stdout


def probe(path):
    """(duration in seconds, whether it has an audio stream)."""
    info = json.loads(run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                           "-of", "json", path], f"ffprobe of {path}"))
    return float(info["format"]["duration"]), any(s["codec_type"] == "audio" for s in info.get("streams", []))


def cut(path, start, end, work, with_audio):
    """(clip path, wav path or None) for the span start..end of path, written into work."""
    stem = os.path.join(work, f"{start:09.3f}")
    span = ["-ss", f"{start:.3f}", "-i", path, "-t", f"{end - start:.3f}"]
    clip = stem + ".mp4"
    run(["ffmpeg", "-v", "error", "-y", *span, "-an", "-vf", f"scale=-2:min({CLIP_HEIGHT}\\,ih)",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", clip], f"cutting {path} at {clock(start)}")
    if not with_audio:
        return clip, None
    wav = stem + ".wav"
    run(["ffmpeg", "-v", "error", "-y", *span, "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), wav],
        f"extracting the audio of {path} at {clock(start)}")
    return clip, wav
