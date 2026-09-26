"""SubRip output. `timestamp`, `wrap` and `units_to_srt` port jp-subs' core/srt.js."""

from whisper_subs._js import js_len

MAX_LINE_CHARS = 42
# Two lines is the convention, but ~6% of translations exceed what two lines
# can hold (Japanese expands 2-4x into English). Jamming the remainder onto
# line two produced an unreadable run-on, so allow a third.
MAX_LINES = 3


def timestamp(ms: float) -> str:
    clamped = max(0, round(ms))
    h = clamped // 3_600_000
    m = clamped // 60_000 % 60
    s = clamped // 1000 % 60
    return f"{h:02d}:{m:02d}:{s:02d},{clamped % 1000:03d}"


def cues_to_srt(cues: list[dict], min_duration_ms: int = 700) -> str:
    """One block per cue, Japanese text as-is.

    Short cues are stretched to `min_duration_ms` so they can be read, but never
    into the next cue.
    """
    blocks = []
    for i, cue in enumerate(cues):
        start = cue["t_ms"]
        end = max(start + cue["dur_ms"], start + min_duration_ms)
        if i + 1 < len(cues):
            end = min(end, cues[i + 1]["t_ms"])
        end = max(end, start + 1)
        blocks.append(f"{len(blocks) + 1}\n{timestamp(start)} --> {timestamp(end)}\n{cue['ja']}\n")
    return "\n".join(blocks)


def wrap(text: str, max_chars: int = MAX_LINE_CHARS, max_lines: int = MAX_LINES) -> str:
    """Greedy word wrap, capped at `max_lines`."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        if not line:
            line = word
        elif js_len(line) + 1 + js_len(word) <= max_chars:
            line += f" {word}"
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    # Too long to show properly: keep the cap, put the remainder on the last line.
    return "\n".join([*lines[: max_lines - 1], " ".join(lines[max_lines - 1:])])


def units_to_srt(units: list[dict], translations: list[str], min_duration_ms: int = 700,
                 japanese: bool = False) -> str:
    """Translated units -> .srt.

    `japanese=True` puts the source line above the English (bilingual). Only
    the English is wrapped: Japanese has no spaces to wrap at, and a unit is
    already capped at 64 characters.
    """
    blocks = []
    for unit, translation in zip(units, translations, strict=True):
        text = (translation or "").strip()
        if not text:
            continue
        start = unit["start_ms"]
        # Never let a cue vanish because the source cue had no duration.
        end = max(unit["end_ms"], start + min_duration_ms)
        body = f"{unit['ja']}\n{wrap(text)}" if japanese else wrap(text)
        blocks.append(f"{len(blocks) + 1}\n{timestamp(start)} --> {timestamp(end)}\n{body}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")
