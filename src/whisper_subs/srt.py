"""SubRip output. `timestamp` is a port of jp-subs' core/srt.js."""


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
