"""Caption cues -> translation units. Port of jp-subs' core/segment.js.

Cues are timed for reading, not for grammar, and their granularity varies
enormously, so work at sentence level instead of cue level: explode every cue
into sentence pieces, then accumulate pieces into units until sentence-final
punctuation, a silence of `gap_ms`, or `max_chars`. Fragmented cues get
joined; overloaded cues get split. One pass, both shapes.

Whisper leaves conversation almost unpunctuated, so on Whisper input the gap
does nearly all the work, and 2000 ms (tuned for YouTube's cues) makes units
three times too long. `WHISPER` is the measured preset (docs/benchmarks.md).
"""

import re

from whisper_subs._js import js_len, js_round

_PUNCT = "。．！？!?"
_CLOSERS = "\"'」』）)】〉》"

#: Sentence-final punctuation, allowing trailing quotes/brackets.
SENTENCE_END = re.compile(f"[{_PUNCT}]+[{re.escape(_CLOSERS)}]*\\s*\\Z")
_ENDS_AT = re.compile(f"[{_PUNCT}]+[{re.escape(_CLOSERS)}]*\\Z")

#: A cue that is only a bracketed tag: [音楽], [拍手], [Music].
TAG_ONLY = re.compile(r"[\[［][^\]］]*[\]］]")

DEFAULTS = {
    # Silence (ms) between cues that ends a unit on its own.
    "gap_ms": 2000,
    # Never let a unit grow past this many characters.
    "max_chars": 64,
    # Drop cues that are nothing but a [music] style tag.
    "drop_tag_only_cues": True,
    # Not in jp-subs: end a unit at the end of a cue once the unit has at least
    # this many characters. None keeps jp-subs' behaviour (tested for parity).
    "cue_end_min_chars": None,
}

#: For Whisper transcripts (docs/benchmarks.md, milestone 2):
#: - gap 500 ms: 2000 (tuned for YouTube's cues) made units three times too long.
#: - Whisper's segments already follow sentences; merging several into one unit
#:   made the model split its English back across line numbers, shifting every
#:   later line in the chunk (5 of 6 runs on one chunk). Ending units at cue
#:   ends once they hold 8+ characters: 0 of 12. Shorter fragments (そう, 外で)
#:   still merge into their neighbour.
WHISPER = {**DEFAULTS, "gap_ms": 500, "cue_end_min_chars": 8}


def _ends_sentence(text: str) -> bool:
    return SENTENCE_END.search(text) is not None


def _split_sentences(text: str) -> list[str]:
    """`text.split(/(?<=[。．！？!?]+["'」』）\\)】〉》]*)/u)`.

    Python has no variable-width lookbehind. The JS splits at EVERY position
    whose prefix ends in punctuation-plus-closers, so a closing bracket or a
    second 「!」 becomes a piece of its own; reproduce that exactly.
    """
    parts, last = [], 0
    for p in range(1, len(text)):
        if _ENDS_AT.search(text, 0, p):
            parts.append(text[last:p])
            last = p
    parts.append(text[last:])
    return parts


def _char_times(cue: dict) -> list[tuple[str, int]] | None:
    """Per-character timestamps from word-level `segs`, or None without them."""
    segs = cue.get("segs")
    if not isinstance(segs, list) or not segs:
        return None
    chars = [(ch, seg.get("t_ms")) for seg in segs for ch in (seg.get("text") or "")]
    return chars or None


def _to_pieces(cues: list[dict], drop_tag_only_cues: bool) -> list[dict]:
    pieces = []

    for index, cue in enumerate(cues):
        cue_text = (cue.get("ja") or "").strip()
        if not cue_text:
            continue
        if drop_tag_only_cues and TAG_ONLY.fullmatch(cue_text):
            continue

        cue_end = cue["t_ms"] + (cue.get("dur_ms") or 0)
        timed = _char_times(cue)

        if timed:
            # Walk the characters, closing a piece at sentence-final punctuation.
            # Every boundary lands on a timestamp that was actually reported.
            buf = ""
            start_ms = None

            def flush_piece(end_ms, cue=cue, index=index):
                nonlocal buf, start_ms
                text = buf.strip()
                buf = ""
                if not text:
                    start_ms = None
                    return
                start = start_ms if start_ms is not None else cue["t_ms"]
                pieces.append({
                    "text": text,
                    "start_ms": start,
                    "end_ms": max(end_ms, start + 1),
                    "cue_index": index,
                    "ends_sentence": _ends_sentence(text),
                })
                start_ms = None

            for i, (ch, t_ms) in enumerate(timed):
                if buf == "" and ch.strip():
                    start_ms = t_ms
                buf += ch
                if _ends_sentence(buf):
                    # End when the next word actually begins; else the cue end.
                    flush_piece(timed[i + 1][1] if i + 1 < len(timed) else cue_end)
            flush_piece(cue_end)
            continue

        # No word timings: apportion starts, but anchor every end to the cue end
        # so nothing expires before the speech in that cue has finished.
        parts = [p.strip() for p in _split_sentences(cue_text)]
        parts = [p for p in parts if p]
        total = sum(js_len(p) for p in parts) or 1
        duration = cue.get("dur_ms") or 0

        offset = 0.0
        for part in parts:
            start = cue["t_ms"] + js_round(duration * offset)
            offset += js_len(part) / total
            pieces.append({
                "text": part,
                "start_ms": start,
                "end_ms": cue_end,
                "cue_index": index,
                "ends_sentence": _ends_sentence(part),
            })

    return pieces


def segment(cues: list[dict], options: dict | None = None) -> list[dict]:
    """-> [{"ja", "start_ms", "end_ms", "cue_index": [int, ...]}, ...]"""
    opts = {**DEFAULTS, **(options or {})}
    pieces = _to_pieces(cues, opts["drop_tag_only_cues"])

    units = []
    current = None

    def flush():
        nonlocal current
        if current and current["ja"].strip():
            units.append(current)
        current = None

    for i, piece in enumerate(pieces):
        if current:
            silence = piece["start_ms"] - current["end_ms"]
            too_long = js_len(current["ja"]) + js_len(piece["text"]) > opts["max_chars"]
            if silence >= opts["gap_ms"] or too_long:
                flush()

        if not current:
            current = {"ja": "", "start_ms": piece["start_ms"], "end_ms": piece["end_ms"],
                       "cue_index": []}

        current["ja"] += piece["text"]
        current["end_ms"] = max(current["end_ms"], piece["end_ms"])
        if piece["cue_index"] not in current["cue_index"]:
            current["cue_index"].append(piece["cue_index"])

        # A piece that ends a sentence ends the unit.
        if piece["ends_sentence"]:
            flush()
        elif opts["cue_end_min_chars"] is not None and current:
            last_of_cue = i + 1 == len(pieces) or pieces[i + 1]["cue_index"] != piece["cue_index"]
            if last_of_cue and js_len(current["ja"]) >= opts["cue_end_min_chars"]:
                flush()

    flush()
    return units
