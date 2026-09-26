"""Drop text Whisper invented. Runs on Whisper segments, before cues are built.

Whisper was trained on subtitled video, so over music or near-silence it
writes what subtitle files contain there: sign-offs and credits. Seen on the
samples (docs/benchmarks.md):

  ご視聴ありがとうございました   gap fill over music (nsp 0.83), twice over BGM
  作詞・作曲・編曲 初音ミク      over the song's instrumental intro

A person can say ご視聴ありがとうございました too, at the end of a stream. The
difference is the audio: a real sign-off is clear speech, a phantom sits over
music or silence with a high no_speech_prob. So a phrase match alone is not
enough to drop a line; it also needs weak evidence of speech.
"""

import re

#: Phrases Whisper produces from silence or music. Matched against the whole
#: segment with spaces and punctuation removed, so a phrase inside a longer
#: real sentence is never touched.
PHANTOMS = [
    r"ご視聴(?:ありがとうございま(?:した|す))?",
    r"(?:最後まで)?ご覧いただきありがとうございま(?:した|す)",
    r"チャンネル登録(?:よろしくお願いします|お願いします)?",
    r"(?:高評価|グッドボタン)(?:とチャンネル登録)?(?:よろしくお願いします|お願いします)?",
    r"字幕(?:作成|制作)?(?:者)?.{0,12}",
    r"(?:作詞|作曲|編曲)(?:・?(?:作詞|作曲|編曲))*.{0,20}",
    r"(?:次回|また次)の?(?:動画|配信)(?:で|も)?(?:お会いしましょう|会いましょう)?",
]
_PHANTOM = re.compile("|".join(f"(?:{p})" for p in PHANTOMS))
_STRIP = re.compile(r"[\s、。，．！？!?・「」『』…]")

#: A phantom phrase is dropped only with at least this no_speech_prob. The
#: phantoms seen so far scored 0.83-0.92.
PHANTOM_MIN_NO_SPEECH = 0.5

#: A line repeated this many times in a row is collapsed to its first copy,
#: if it is long enough that the repetition cannot be a real そうそうそう.
REPEAT_MIN_RUN = 3
REPEAT_MIN_CHARS = 6


def is_phantom(segment: dict) -> bool:
    text = _STRIP.sub("", segment.get("text", ""))
    return (bool(text) and _PHANTOM.fullmatch(text) is not None
            and segment.get("no_speech_prob", 0) >= PHANTOM_MIN_NO_SPEECH)


def clean(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """-> (kept, dropped). Each dropped segment carries a `dropped` reason."""
    kept, dropped = [], []
    for s in segments:
        if is_phantom(s):
            dropped.append({**s, "dropped": "phantom phrase"})
        else:
            kept.append(s)

    # Runs of identical consecutive lines: a long line said 3+ times in a row
    # is Whisper looping, so keep the first copy only.
    out = []
    i = 0
    while i < len(kept):
        text = kept[i]["text"].strip()
        j = i
        while j + 1 < len(kept) and kept[j + 1]["text"].strip() == text:
            j += 1
        out.append(kept[i])
        rest = kept[i + 1:j + 1]
        if len(rest) + 1 >= REPEAT_MIN_RUN and len(text) >= REPEAT_MIN_CHARS:
            dropped.extend({**r, "dropped": "repeated line"} for r in rest)
        else:
            out.extend(rest)
        i = j + 1
    return out, dropped
