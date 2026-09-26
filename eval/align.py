"""Report translations that landed on the wrong line.

    python eval/align.py out/video.en.json [more.en.json ...] [--model qwen3.5:9b]

Exits 1 if any shifted run is found. See whisper_subs/align.py for the method;
`whisper-subs --check` runs the same check and also repairs what it finds.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_subs.align import DEFAULT_JUDGE, MIN_RUN, judge, runs


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("en_json", type=Path, nargs="+")
    p.add_argument("--model", default=DEFAULT_JUDGE)
    p.add_argument("--min-run", type=int, default=MIN_RUN)
    args = p.parse_args()

    status = 0
    for path in args.en_json:
        units = json.loads(path.read_text(encoding="utf-8"))["units"]
        verdicts = judge(units, args.model)
        counts = {k: verdicts.count(k) for k in ("same", "prev", "next", "none", "?")}
        found = runs(verdicts, args.min_run)
        print(f"{path}: {len(units)} lines | " + ", ".join(f"{k} {v}" for k, v in counts.items())
              + f" | shifted runs: {len(found)}")
        for a, b, direction in found:
            status = 1
            print(f"  lines {a + 1}-{b + 1} look like the {direction} line's English, e.g.")
            for i in range(a, min(b + 1, a + 3)):
                print(f"    {i + 1:5d} {units[i]['ja'][:24]:<26}| {units[i]['en']}")
    return status


if __name__ == "__main__":
    sys.exit(main())
