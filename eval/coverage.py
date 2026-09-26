"""How much of a reference transcript does a Whisper transcript cover?

    python eval/coverage.py samples/out/clip-sample.ja.json \
        "D:/Desktop/Translator Project/eval/fixtures/NSY6YHXbxtA_full.ja.json"

Both files are jp-subs `.ja.json` transcripts. The reference is YouTube's own
ASR, which is not ground truth: it misses lines too (in-game voice acting,
mostly), so "extra" cues are frequently real speech rather than hallucination.
Read the listed lines before concluding either way.

A reference cue counts as covered if any candidate cue overlaps it in time.
"""

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cues"]


def overlaps(a: dict, b: dict) -> bool:
    return a["t_ms"] < b["t_ms"] + b["dur_ms"] and a["t_ms"] + a["dur_ms"] > b["t_ms"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("candidate", type=Path, nargs="+")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--show", type=int, default=0, help="list N missed and N extra lines")
    args = p.parse_args()

    # One-character reference cues are mostly stray digits and filler.
    ref = [c for c in load(args.reference) if len(c["ja"].strip()) > 1]
    ref_chars = sum(len(c["ja"]) for c in ref)
    print(f"reference: {args.reference.name}, {len(ref)} cues, {ref_chars} chars\n")
    print(f"{'candidate':<44}{'cues':>6}{'missed':>9}{'missed chars':>14}{'extra':>7}")

    for path in args.candidate:
        cand = load(path)
        missed = [c for c in ref if not any(overlaps(c, x) for x in cand)]
        extra = [x for x in cand if not any(overlaps(x, c) for c in ref)]
        chars = sum(len(c["ja"]) for c in missed)
        name = "/".join(path.parts[-2:])
        print(f"{name:<44}{len(cand):>6}{len(missed):>5} ({len(missed) / len(ref):3.0%})"
              f"{chars:>8} ({chars / ref_chars:3.0%}){len(extra):>7}")
        for label, rows in (("missed", missed), ("extra", extra)):
            for c in rows[: args.show]:
                print(f"    {label:<6} {c['t_ms'] / 1000:7.1f}  {c['ja']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
