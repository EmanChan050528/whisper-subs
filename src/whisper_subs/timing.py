"""When each subtitle is on screen.

Units carry the time the words were spoken. That is the right start, but not
always the right end: a short utterance (そう, うわっ) is spoken in a few hundred
milliseconds and its subtitle flashes by, and English runs longer than the
Japanese it translates, so a line can end before it can be read. Before this
pass, NSY6YHXbxtA had 40 cues under 1 s, 40 over 20 characters per second,
and 3 overlapping the next cue.

The fix only ever moves the END, and only into silence: the start stays where
the speech starts.
"""

#: Reading speed to aim for, English characters per second. Netflix's adult
#: English guideline is 20; 17 leaves margin for dense lines.
TARGET_CPS = 17
MIN_S = 1.0
#: Only a line that genuinely needs it (a long unit, slowly spoken) goes past this.
MAX_S = 7.0
#: Gaps shorter than this read as flicker; close them, leaving FRAME_GAP_S.
CLOSE_GAP_S = 0.25
#: Two frames at 24 fps: enough for the eye to register a change of subtitle.
FRAME_GAP_S = 0.083


def display_times(units: list[dict], translations: list[str]) -> list[tuple[int, int] | None]:
    """-> (start_ms, end_ms) per unit, or None where there is no translation."""
    shown = [i for i, t in enumerate(translations) if (t or "").strip()]
    times: list[tuple[int, int] | None] = [None] * len(units)

    for k, i in enumerate(shown):
        start = units[i]["start_ms"]
        spoken_end = max(units[i]["end_ms"], start + 1)
        text = " ".join(translations[i].split())
        wanted = max(MIN_S, len(text) / TARGET_CPS) * 1000
        end = max(spoken_end, start + min(wanted, MAX_S * 1000))

        if k + 1 < len(shown):
            next_start = units[shown[k + 1]]["start_ms"]
            limit = next_start - FRAME_GAP_S * 1000
            if next_start - end < CLOSE_GAP_S * 1000:
                end = limit          # close a flicker gap
            end = min(end, limit)    # and never overlap
        end = max(end, start + 1)
        times[i] = (start, round(end))
    return times
