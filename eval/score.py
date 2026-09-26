"""Score English subtitles as they will actually be shown.

    python eval/score.py out/clip.en.srt [more.en.srt ...] \
        --reference tests/fixtures/youtube_NSY6YHXbxtA.ja.json

Reads the .srt, which is what the viewer sees. Reports:

  coverage   reference lines with no subtitle on screen at the time, if a
             reference .ja.json is given (YouTube's captions: not ground truth)
  gaps       seconds of reference speech inside stretches of 10 s or more with
             no subtitle, the thing a viewer notices as "nothing for a while"
  timing     cues under 1 s (flashes), over 7 s, overlapping the next one, and
             reading speed over 20 characters per second

Not measured: translations landing on a neighbour's line. An English-length
heuristic caught 1 of 52 deliberately shifted lines, so it was removed rather
than kept as false reassurance. Checking it needs a model (DEVELOPMENT.md 3.1).
"""

import argparse
import json
import re
import sys
from pathlib import Path

STAMP = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{3})\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d{3})")


def parse_srt(text: str) -> list[dict]:
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = block.split("\n")
        for i, line in enumerate(lines):
            m = STAMP.search(line)
            if m:
                g = [int(x) for x in m.groups()]
                start = ((g[0] * 60 + g[1]) * 60 + g[2]) * 1000 + g[3]
                end = ((g[4] * 60 + g[5]) * 60 + g[6]) * 1000 + g[7]
                cues.append({"start": start, "end": end, "text": "\n".join(lines[i + 1:])})
                break
    return cues


def score(srt_path: Path, reference: list[dict] | None) -> dict:
    cues = parse_srt(srt_path.read_text(encoding="utf-8"))
    row: dict = {"file": "/".join(srt_path.parts[-2:]), "cues": len(cues)}

    durs = [(c["end"] - c["start"]) / 1000 for c in cues]
    row["under_1s"] = sum(d < 1 for d in durs)
    row["over_7s"] = sum(d > 7 for d in durs)
    row["overlaps"] = sum(cues[i]["end"] > cues[i + 1]["start"] for i in range(len(cues) - 1))
    row["cps_over_20"] = sum(
        len(c["text"].replace("\n", " ")) / max(d, 0.001) > 20
        for c, d in zip(cues, durs, strict=True))

    if reference:
        def shown(t0, t1):
            return any(c["start"] < t1 and c["end"] > t0 for c in cues)
        missed = [r for r in reference if not shown(r["t_ms"], r["t_ms"] + r["dur_ms"])]
        row["ref_missed"] = f"{len(missed)}/{len(reference)} ({len(missed) / len(reference):.0%})"

        # Reference speech that falls inside a long subtitle-free stretch.
        edges = [0] + [x for c in cues for x in (c["start"], c["end"])] + [10**12]
        long_gaps = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)
                     if edges[i + 1] - edges[i] >= 10_000]
        stranded = [r for r in missed
                    if any(a <= r["t_ms"] < b for a, b in long_gaps)]
        row["speech_in_10s_gaps"] = f"{sum(r['dur_ms'] for r in stranded) / 1000:.0f}s"
    return row


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("srt", type=Path, nargs="+")
    p.add_argument("--reference", type=Path, help="a jp-subs .ja.json, e.g. YouTube's captions")
    args = p.parse_args()

    reference = None
    if args.reference:
        cues = json.loads(args.reference.read_text(encoding="utf-8"))["cues"]
        # One-character reference cues are mostly stray digits and filler.
        reference = [c for c in cues if len(c["ja"].strip()) > 1]

    rows = [score(path, reference) for path in args.srt]
    cols = list(dict.fromkeys(k for r in rows for k in r))
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
