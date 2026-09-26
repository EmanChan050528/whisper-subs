"""whisper-subs: Japanese audio or video in, subtitles out."""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from whisper_subs import __version__
from whisper_subs.cues import to_transcript
from whisper_subs.srt import cues_to_srt
from whisper_subs.transcribe import DEFAULT_MODEL, FAST_MODEL, Options

# Options that change what Whisper produces. A cached .whisper.json is reused
# only if all of these match.
CACHE_KEYS = ("model", "language", "beam_size", "vad", "vad_threshold",
              "condition_on_previous_text",
              "initial_prompt")


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="whisper-subs",
        description="Subtitle Japanese audio or video locally with Whisper.",
    )
    p.add_argument("input", type=Path, help="video or audio file (mp4, mkv, mp3, wav, ...)")
    p.add_argument("--model", default=None, help=f"Whisper model (default {DEFAULT_MODEL})")
    p.add_argument("--fast", action="store_true",
                   help=f"use {FAST_MODEL}: ~4x faster, less accurate on rare words")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--compute-type", default="auto",
                   help="float16, int8_float16, int8, ... (default: float16 on GPU, int8 on CPU)")
    p.add_argument("--vad", action="store_true",
                   help="skip non-speech with voice-activity detection. Off by default: it "
                        "drops quiet or backgrounded speech and all singing, but can cut "
                        "hallucinations on audio with long silences")
    p.add_argument("--vad-threshold", type=float, default=0.5,
                   help="with --vad: speech probability needed to keep audio")
    p.add_argument("--prompt", default=None,
                   help="initial prompt for Whisper, e.g. names or a punctuated sample")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: beside the input)")
    p.add_argument("--force", action="store_true",
                   help="re-transcribe even if a cached result exists")
    p.add_argument("--ja-only", action="store_true",
                   help="Japanese subtitles only (the only mode until translation lands)")
    p.add_argument("-V", "--version", action="version", version=f"whisper-subs {__version__}")
    args = p.parse_args(argv)
    if args.model and args.fast:
        p.error("--model and --fast are mutually exclusive")
    return args


def progress(done: float, total: float) -> None:
    def clock(s):
        s = int(s)
        if s >= 3600:
            return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"
        return f"{s // 60}:{s % 60:02d}"
    pct = 100 * done / total if total else 100
    print(f"\r  transcribing {pct:5.1f}%  ({clock(done)} / {clock(total)})", end="",
          file=sys.stderr, flush=True)


def source_id(src: Path) -> dict:
    """Identifies the input file, so a cache written for movie.mp4 is never
    reused for movie.mkv, or for movie.mp4 after it has been replaced."""
    st = src.stat()
    return {"name": src.name, "size": st.st_size, "mtime": int(st.st_mtime)}


def load_cached(path: Path, opts: Options, src: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if cached.get("source") != source_id(src):
        return None
    want = asdict(opts)
    if all(cached.get("options", {}).get(k) == want[k] for k in CACHE_KEYS):
        return cached
    return None


def main(argv: list[str] | None = None) -> int:
    # The Windows console defaults to cp1252 and cannot print Japanese.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # not on pytest's capture objects
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(sys.argv[1:] if argv is None else argv)

    opts = Options(
        model=FAST_MODEL if args.fast else (args.model or DEFAULT_MODEL),
        device=args.device,
        compute_type=args.compute_type,
        vad=args.vad,
        vad_threshold=args.vad_threshold,
        initial_prompt=args.prompt,
    )

    src: Path = args.input
    if not src.is_file():
        print(f"error: no such file: {src}", file=sys.stderr)
        return 1
    out_dir = args.out or src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    whisper_path = out_dir / f"{stem}.whisper.json"
    ja_json_path = out_dir / f"{stem}.ja.json"
    ja_srt_path = out_dir / f"{stem}.ja.srt"

    whisper = None if args.force else load_cached(whisper_path, opts, src)
    if whisper:
        print(f"using cached transcription: {whisper_path.name} (--force to redo)")
    else:
        from whisper_subs.audio import AudioError, load
        from whisper_subs.transcribe import transcribe

        try:
            audio = load(src)
        except AudioError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        print(f"{src.name}: {len(audio) / 16000:.0f}s of audio, model {opts.model}")
        started = time.perf_counter()
        whisper = {"source": source_id(src), **transcribe(audio, opts, on_progress=progress)}
        elapsed = time.perf_counter() - started
        print(file=sys.stderr)
        print(f"  {len(whisper['segments'])} segments in {elapsed:.0f}s "
              f"({whisper['duration'] / elapsed:.1f}x real time, "
              f"{whisper['options']['device']}/{whisper['options']['compute_type']})")
        whisper_path.write_text(json.dumps(whisper, ensure_ascii=False, indent=1), encoding="utf-8")

    transcript = to_transcript(whisper, title=stem)
    ja_json_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    ja_srt_path.write_text(cues_to_srt(transcript["cues"]), encoding="utf-8")

    print(f"\n  {ja_srt_path}\n  {ja_json_path}\n  {whisper_path}")
    if not args.ja_only:
        print("\nEnglish translation is not built in yet (milestone 2). Meanwhile jp-subs can")
        print("translate the .ja.json directly:")
        print(f'  node "<jp-subs>/core/bin/jpsub.js" translate "{ja_json_path}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
