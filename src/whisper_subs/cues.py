"""Whisper segments -> jp-subs transcript cues (`.ja.json`).

The `.ja.json` format is jp-subs' transcript: one cue per caption line,

    {"t_ms": 8400, "dur_ms": 3279, "ja": "...", "segs": [{"text": "...", "t_ms": 8400}, ...]}

where `segs` are word-level timings. YouTube supplies them on about half its
cues; Whisper supplies them on every one, so jp-subs' segmenter places every
sentence break on a real timestamp.

Writing this format means jp-subs' own CLI can translate our output unchanged:

    node <jp-subs>/core/bin/jpsub.js translate video.ja.json
"""

from datetime import UTC, datetime

#: Split a Whisper segment wherever consecutive words are further apart than
#: this. Whisper sometimes pins a segment's first word tens of seconds before
#: the rest (measured: 「え?」 at 160.9 s, then 「俺じゃないって言った」 at
#: 204.3 s, all one segment), which would show the whole line 45 s early.
MAX_WORD_GAP_S = 1.0


def split_on_word_gaps(words: list[dict], max_gap: float = MAX_WORD_GAP_S) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for w in words:
        if groups and w["start"] - groups[-1][-1]["end"] <= max_gap:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def _cue(start_s: float, end_s: float, ja: str, segs: list[dict] | None) -> dict:
    t_ms = round(start_s * 1000)
    cue = {"t_ms": t_ms, "dur_ms": max(1, round(end_s * 1000) - t_ms), "ja": ja}
    if segs:
        cue["segs"] = segs
    return cue


def to_cues(segments: list[dict], max_word_gap: float = MAX_WORD_GAP_S) -> list[dict]:
    cues = []
    for s in segments:
        words = s.get("words") or []
        if not words:
            if s["text"].strip():
                cues.append(_cue(s["start"], s["end"], s["text"].strip(), None))
            continue

        for group in split_on_word_gaps(words, max_word_gap):
            segs = [{"text": w["word"], "t_ms": round(w["start"] * 1000)} for w in group]
            # Whisper puts a leading space on the first word. Keep `ja` and the
            # concatenated segs identical: segment.js reads text from segs when
            # present, and the two must agree.
            segs[0]["text"] = segs[0]["text"].lstrip()
            ja = "".join(x["text"] for x in segs)
            if not ja.strip():
                continue
            # Word times, not segment times: the segment's own start and end
            # carry the same drift the split exists to remove.
            cues.append(_cue(group[0]["start"], group[-1]["end"], ja, segs))
    return cues


def to_transcript(whisper: dict, title: str) -> dict:
    cues = to_cues(whisper["segments"])
    opts = whisper["options"]
    return {
        "title": title,
        "duration_s": round(whisper["duration"]),
        "source": f"whisper {opts['model']} ({opts['device']}/{opts['compute_type']})",
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "cue_count": len(cues),
        "cues": cues,
    }
