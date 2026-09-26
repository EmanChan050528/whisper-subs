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

from whisper_subs.filters import clean

#: Split a Whisper segment wherever consecutive words are further apart than
#: this. Whisper sometimes pins a segment's first word tens of seconds before
#: the rest (measured: 「え?」 at 160.9 s, then 「俺じゃないって言った」 at
#: 204.3 s, all one segment), which would show the whole line 45 s early.
MAX_WORD_GAP_S = 1.0

#: No single word is spoken for longer than this. Over silence or music
#: Whisper stretches a word's end (measured: 「でした」 6.7 s, 「予」 11.3 s),
#: and the subtitle then stays up long after the speech has stopped. Applied
#: to a cue's LAST word only: inside a line a stretched word is harmless, and
#: clamping it there opened a gap the 1 s rule then split (奥|深い).
MAX_WORD_S = 1.5

#: A cue longer than this is split at its widest pause between words...
MAX_CUE_S = 7.0
#: ...but only at a real pause, and only into halves with this much text.
#: Dense, unbroken speech (「あら誰かと思えば…子じゃない」, 7.9 s) is left whole:
#: chopping a sentence mid-clause costs the translation more than a long
#: subtitle costs the reader.
MIN_SPLIT_GAP_S = 0.15
MIN_SPLIT_CHARS = 6


def _breakable(before: str, after: str) -> bool:
    """Could a clause end between these two word pieces?

    Whisper's "words" are fragments, and speakers pause inside words too:
    splitting at the widest pause alone cut 奥|ゆかしい and 奥|深い. A clause
    ends after hiragana (a particle or an inflection: が, て, けど, ました) or
    punctuation, not after kanji, and not before a small kana or a long mark.
    """
    before, after = before.strip(), after.strip()
    if not before or not after:
        return False
    last, first = before[-1], after[0]
    ends_ok = "ぁ" <= last <= "ゟ" or last in "、。，．！？!?…」』)"
    starts_ok = first not in "ぁぃぅぇぉゃゅょっゎァィゥェォャュョッヮーゝゞ々"
    return ends_ok and starts_ok


def split_on_word_gaps(words: list[dict], max_gap: float = MAX_WORD_GAP_S) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for w in words:
        if groups and w["start"] - groups[-1][-1]["end"] <= max_gap:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def clamp_last_word(group: list[dict], max_s: float = MAX_WORD_S) -> list[dict]:
    last = group[-1]
    return [*group[:-1], {**last, "end": min(last["end"], last["start"] + max_s)}]


def split_long(group: list[dict], max_s: float = MAX_CUE_S) -> list[list[dict]]:
    """Split a run of words longer than max_s at its widest real pause, recursively."""
    if len(group) < 2 or group[-1]["end"] - group[0]["start"] <= max_s:
        return [group]

    def chars(ws):
        return len("".join(w["word"] for w in ws).strip())

    best, best_gap = None, MIN_SPLIT_GAP_S
    for i in range(1, len(group)):
        gap = group[i]["start"] - group[i - 1]["end"]
        if (gap >= best_gap and _breakable(group[i - 1]["word"], group[i]["word"])
                and min(chars(group[:i]), chars(group[i:])) >= MIN_SPLIT_CHARS):
            best, best_gap = i, gap
    if best is None:
        return [group]
    return split_long(group[:best], max_s) + split_long(group[best:], max_s)


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

        groups = [clamp_last_word(piece) for g in split_on_word_gaps(words, max_word_gap)
                  for piece in split_long(g)]
        for group in groups:
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
    kept, dropped = clean(whisper["segments"])
    cues = to_cues(kept)
    opts = whisper["options"]
    return {
        "title": title,
        "duration_s": round(whisper["duration"]),
        "source": f"whisper {opts['model']} ({opts['device']}/{opts['compute_type']})",
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "cue_count": len(cues),
        "cues": cues,
        # Kept for inspection: what the hallucination filters removed, and why.
        "dropped": [{"t_ms": round(d["start"] * 1000), "ja": d["text"], "reason": d["dropped"]}
                    for d in dropped],
    }
