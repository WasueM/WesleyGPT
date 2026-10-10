# Wesley wrote this
"""A JSON transcript of a video: every word spoken and every thing shown, with times on the video's clock.

The model transcribes one window of up to 30 s at a time (all it can hear at once), each window
on its own, so a long video costs no more per window than a short one. The model is only
trusted with the words and sights; the bookkeeping it gets wrong is done here instead. It
wraps its JSON in a ``` fence, writes times like "0m0s03s164ms" when left to choose a
format, stamps them from the start of the clip rather than the video, and can get stuck
listing the same sight once per frame ("black screen" 13 times), or split its answer into two
JSON objects. A window it cannot answer readably in ATTEMPTS tries becomes a named gap in
"unreadable_windows" rather than costing the windows around it.
"""
import json
import os
import re

from wesleyqwen import media
from wesleyqwen.scoring import final_answer

ATTEMPTS = 2
FORMAT = ('{"speech": [{"start": 0.0, "end": 4.2, "text": "the words, exactly as said"}],\n'
          ' "shown": [{"at": 0.0, "what": "one thing visible on screen"}]}')
CLOCK = re.compile(r"\s*(\d+(?::\d+){0,2}(?:\.\d+)?)\s*s?\s*")


def transcript_path(video):
    """Where the transcript of a video is saved: right beside it."""
    return video + ".transcript.json"


def window_prompt(length, has_audio):
    heard = ("Its frames come before this message and its audio after it." if has_audio
             else "It has no audio track, so leave \"speech\" empty.")
    return (f"Transcribe this {length:.0f}-second clip. {heard}\n"
            f"Reply with ONLY a JSON object of this form:\n{FORMAT}\n"
            '"speech": every word spoken, verbatim, one entry per sentence with its own start and end; '
            "nothing summarized, nothing left out.\n"
            '"shown": what is on screen, in time order: each scene, person, place and object, and any '
            "on-screen text quoted exactly. List each thing once, at the second it first appears.\n"
            "Times are seconds from the start of this clip, as plain numbers.")


def seconds(value):
    """A time the model wrote, as seconds: 12, "12.5", "12.5s", "1:05" or "0:01:07.5"."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = CLOCK.fullmatch(value) if isinstance(value, str) else None
    if not match:
        raise ValueError(f"cannot read the time {value!r}; expected seconds or m:ss")
    total = 0.0
    for part in match.group(1).split(":"):
        total = total * 60 + float(part)
    return total


def parse_window(text):
    """{"speech": [{start, end, text}], "shown": [{at, what}]} from one window's reply, times in clip seconds."""
    objects, first_error, decoder = [], None, json.JSONDecoder()
    at = text.find("{")
    while at >= 0:
        try:
            found, end = decoder.raw_decode(text, at)
            if isinstance(found, dict):
                objects.append(found)
            at = text.find("{", end)
        except json.JSONDecodeError as error:
            first_error = first_error or error
            at = text.find("{", at + 1)
    if not objects:
        raise ValueError(f"unreadable JSON in the reply ({first_error})" if first_error
                         else f"no JSON object in the reply: {text[:80]!r}")
    data = {}
    for key in ("speech", "shown"):
        lists = [found[key] for found in objects if key in found]
        if not lists or not all(isinstance(part, list) for part in lists):
            raise ValueError(f'the reply has no "{key}" list')
        data[key] = [entry for part in lists for entry in part]
    try:
        speech = [{"start": seconds(line["start"]), "end": seconds(line.get("end", line["start"])),
                   "text": line["text"].strip()} for line in data["speech"]]
        shown = [{"at": seconds(item["at"]), "what": item["what"].strip()} for item in data["shown"]]
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"a speech or shown entry is missing or misshapen ({error!r})") from error
    return {"speech": [line for line in speech if line["text"]], "shown": first_sightings(shown)}


def first_sightings(shown):
    """Each thing once, at the first time it was listed, however many times the model repeated it."""
    seen, kept = set(), []
    for item in shown:
        key = re.sub(r"[^a-z0-9 ]", "", item["what"].lower()).strip()
        if key and key not in seen:
            seen.add(key)
            kept.append(item)
    return kept


def place(window, start, end):
    """The window with its clip times moved onto the video's clock, none outside start..end."""
    def at(t):
        return round(min(max(start + t, start), end), 1)
    return {"speech": sorted(({**line, "start": at(line["start"]), "end": at(line["end"])} for line in window["speech"]),
                             key=lambda line: line["start"]),
            "shown": sorted(({**item, "at": at(item["at"])} for item in window["shown"]), key=lambda item: item["at"])}


def assemble(name, duration, model, windows, gaps):
    """One transcript from the placed windows, in order; gaps are the windows that could not be read."""
    speech = [line for window in windows for line in window["speech"]]
    return {"video": name, "duration_seconds": round(duration, 1), "model": model,
            "words_spoken": " ".join(line["text"] for line in speech),
            "speech": speech, "shown": [item for window in windows for item in window["shown"]],
            "unreadable_windows": gaps}


def timeline(transcript):
    """The transcript as plain lines in time order; what the conversation keeps of it."""
    events = sorted([(item["at"], 0, f"shown: {item['what']}") for item in transcript["shown"]] +
                    [(line["start"], 1, f"said: {line['text']}") for line in transcript["speech"]])
    return "\n".join(f"{media.clock(at)} {text}" for at, _, text in events)


def transcribe(stream, cut, history, path, duration, model, save_to):
    """Yield progress per window, then the transcript as JSON; save it to save_to and record it in history.

    cut(path, start, end) -> (clip, wav or None); stream(messages, thinking) yields text.
    """
    name, placed, gaps = os.path.basename(path), [], []
    for start, end in media.windows(duration, media.AUDIO_WINDOW_SECONDS):
        clip, wav = cut(path, start, end)
        label = f"{media.clock(start)}–{media.clock(end)}"
        yield f"[{label}] "
        turn = media.window_turn(clip, window_prompt(end - start, wav is not None), wav)
        window = None
        for attempt in range(1, ATTEMPTS + 1):
            try:
                window = parse_window(final_answer("".join(stream([turn], False))))
                break
            except ValueError as error:
                problem = str(error)
                if attempt < ATTEMPTS:
                    yield f"unreadable reply ({problem[:100]}), asking again… "
        if window is None:
            gaps.append({"start": round(start, 1), "end": round(end, 1), "problem": problem})
            yield f"could not read {label}: {problem}\n"
            continue
        placed.append(place(window, start, end))
        yield f"{len(placed[-1]['speech'])} lines heard, {len(placed[-1]['shown'])} things seen\n"
    result = assemble(name, duration, model, placed, gaps)
    body = json.dumps(result, indent=2, ensure_ascii=False)
    if save_to:
        with open(save_to, "w", encoding="utf-8") as out:
            out.write(body + "\n")
    yield f"\n{body}\n" + (f"\nSaved to {save_to}\n" if save_to else "")
    history.append({"role": "user", "content": f"[transcript of video {name}, {media.clock(duration)}]"})
    # Kept as JSON, the transcript taught the model to answer the next question in JSON too.
    history.append({"role": "assistant", "content": timeline(result)})
