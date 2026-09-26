"""whisper-subs: Japanese audio or video in, English subtitles out."""

import argparse
import sys
from pathlib import Path

from whisper_subs import __version__
from whisper_subs.job import JobError, Settings, run_job
from whisper_subs.ollama import DEFAULT_MODEL as DEFAULT_LLM
from whisper_subs.transcribe import DEFAULT_MODEL, FAST_MODEL


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="whisper-subs",
        description="Subtitle Japanese audio or video locally: Whisper transcribes, a local "
                    "LLM on Ollama translates. Also accepts a .ja.json transcript (from an "
                    "earlier run, or from jp-subs) to translate without transcribing.",
    )
    p.add_argument("input", type=Path,
                   help="video or audio file (mp4, mkv, mp3, wav, ...) or a .ja.json transcript")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: beside the input)")
    p.add_argument("--glossary", metavar="NAME", default=None,
                   help="a glossary shared by files of one channel or series: it keeps "
                        "names and terms translated the same way every time, and each run "
                        "adds what it finds. Stored in "
                        "%%APPDATA%%/whisper-subs/glossaries/NAME.json; edit freely")

    w = p.add_argument_group("transcription")
    w.add_argument("--model", default=None, help=f"Whisper model (default {DEFAULT_MODEL})")
    w.add_argument("--fast", action="store_true",
                   help=f"use {FAST_MODEL}: ~4x faster, less accurate on rare words")
    w.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    w.add_argument("--compute-type", default="auto",
                   help="float16, int8_float16, int8, ... (default: float16 on GPU, int8 on CPU)")
    w.add_argument("--vad", action="store_true",
                   help="skip non-speech with voice-activity detection. Off by default: it "
                        "drops quiet or backgrounded speech and all singing, but can cut "
                        "hallucinations on audio with long silences")
    w.add_argument("--vad-threshold", type=float, default=0.5,
                   help="with --vad: speech probability needed to keep audio")
    w.add_argument("--prompt", default=None,
                   help="initial prompt for Whisper, e.g. names that keep being misheard")
    w.add_argument("--no-gap-fill", action="store_true",
                   help="skip the second pass over speech the first pass missed")
    w.add_argument("--force", action="store_true",
                   help="start over: ignore cached results and interrupted-run checkpoints")

    t = p.add_argument_group("translation")
    t.add_argument("--ja-only", action="store_true", help="stop after the Japanese subtitles")
    t.add_argument("--llm-model", default=DEFAULT_LLM,
                   help=f"Ollama model for translation (default {DEFAULT_LLM})")
    t.add_argument("--bilingual", action="store_true",
                   help="also write .ja-en.srt with Japanese above the English")
    t.add_argument("--size", type=int, default=20, help="units translated per request")
    t.add_argument("--parallel", type=int, default=2,
                   help="translation requests in flight at once (default 2; 1 = one at a time)")
    t.add_argument("--context-before", type=int, default=10, help="read-only units before")
    t.add_argument("--context-after", type=int, default=6, help="read-only units after")
    t.add_argument("--limit", type=int, default=None,
                   help="only translate the first N cues. Cheap; use it to try a model")
    t.add_argument("--gap-ms", type=int, default=None,
                   help="silence that ends a translation unit (default 500 for Whisper "
                        "transcripts, 2000 for others)")

    p.add_argument("-V", "--version", action="version", version=f"whisper-subs {__version__}")
    args = p.parse_args(argv)
    if args.model and args.fast:
        p.error("--model and --fast are mutually exclusive")
    return args


def settings_from(args) -> Settings:
    return Settings(
        out=args.out, glossary=args.glossary, model=args.model, fast=args.fast,
        device=args.device, compute_type=args.compute_type, vad=args.vad,
        vad_threshold=args.vad_threshold, prompt=args.prompt, gap_fill=not args.no_gap_fill,
        force=args.force, ja_only=args.ja_only, llm_model=args.llm_model,
        bilingual=args.bilingual, size=args.size, parallel=args.parallel,
        context_before=args.context_before, context_after=args.context_after,
        limit=args.limit, gap_ms=args.gap_ms,
    )


def _clock(s: float) -> str:
    s = int(s)
    if s >= 3600:
        return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


class Printer:
    """Turns job events into terminal output: log lines, and one
    self-overwriting line for transcription progress."""

    def __init__(self):
        self.progress_line = False

    def log(self, msg: str) -> None:
        self._end_progress()
        print(f"  {msg}", flush=True)

    def progress(self, stage: str, done: float, total: float) -> None:
        if stage != "transcribe":
            return  # translation logs a line per chunk already
        pct = 100 * done / total if total else 100
        print(f"\r  transcribing {pct:5.1f}%  ({_clock(done)} / {_clock(total)})", end="",
              file=sys.stderr, flush=True)
        self.progress_line = True

    def _end_progress(self) -> None:
        if self.progress_line:
            print(file=sys.stderr)
            self.progress_line = False


def main(argv: list[str] | None = None) -> int:
    # The Windows console defaults to cp1252 and cannot print Japanese.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # not on pytest's capture objects
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(sys.argv[1:] if argv is None else argv)

    out = Printer()
    try:
        result = run_job(args.input, settings_from(args), log=out.log, progress=out.progress)
    except JobError as err:
        out._end_progress()
        print(f"error: {err}", file=sys.stderr)
        return 1

    if result.failures:
        print(f"\n  {len(result.failures)} problem(s):")
        for f in result.failures:
            print(f"    - {f}")
    if result.missing:
        print(f"\n  {result.missing} line(s) have no translation. Run the same command again "
              f"to retry only those.")
    print()
    for path in result.outputs:
        print(f"  {path}")
    return 0 if (args.ja_only or result.translated) else 1


if __name__ == "__main__":
    sys.exit(main())
