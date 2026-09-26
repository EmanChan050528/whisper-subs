"""whisper-subs: Japanese audio or video in, English subtitles out."""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from whisper_subs import __version__
from whisper_subs.cues import to_transcript
from whisper_subs.ollama import DEFAULT_MODEL as DEFAULT_LLM
from whisper_subs.srt import cues_to_srt, units_to_srt
from whisper_subs.transcribe import DEFAULT_MODEL, FAST_MODEL, Options

# Options that change what Whisper produces. A cached .whisper.json is reused
# only if all of these match.
CACHE_KEYS = ("model", "language", "beam_size", "vad", "vad_threshold",
              "condition_on_previous_text", "initial_prompt")


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
    w.add_argument("--force", action="store_true",
                   help="re-transcribe even if a cached result exists")

    t = p.add_argument_group("translation")
    t.add_argument("--ja-only", action="store_true", help="stop after the Japanese subtitles")
    t.add_argument("--llm-model", default=DEFAULT_LLM,
                   help=f"Ollama model for translation (default {DEFAULT_LLM})")
    t.add_argument("--bilingual", action="store_true",
                   help="also write .ja-en.srt with Japanese above the English")
    t.add_argument("--size", type=int, default=20, help="units translated per request")
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


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def transcribe_step(src: Path, out_dir: Path, stem: str, args) -> dict | None:
    """Media file -> jp-subs transcript, writing .whisper.json, .ja.json, .ja.srt."""
    opts = Options(
        model=FAST_MODEL if args.fast else (args.model or DEFAULT_MODEL),
        device=args.device,
        compute_type=args.compute_type,
        vad=args.vad,
        vad_threshold=args.vad_threshold,
        initial_prompt=args.prompt,
    )
    whisper_path = out_dir / f"{stem}.whisper.json"

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
            return None
        if not args.ja_only:
            # An LLM left resident from an earlier run competes with Whisper for
            # VRAM; both fit on 12 GB only barely (docs/benchmarks.md).
            from whisper_subs.ollama import unload
            unload(args.llm_model)
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
    write_json(out_dir / f"{stem}.ja.json", transcript)
    (out_dir / f"{stem}.ja.srt").write_text(cues_to_srt(transcript["cues"]), encoding="utf-8")
    print(f"  {out_dir / f'{stem}.ja.srt'}")
    return transcript


def translate_step(transcript: dict, out_dir: Path, stem: str, args) -> int:
    from whisper_subs.ollama import ollama_backend
    from whisper_subs.pipeline import run
    from whisper_subs.segment import WHISPER

    is_whisper = str(transcript.get("source", "")).startswith("whisper")
    gap_ms = args.gap_ms if args.gap_ms is not None else (
        WHISPER["gap_ms"] if is_whisper else 2000)
    options = {"gap_ms": gap_ms, "size": args.size,
               "cue_end_min_chars": WHISPER["cue_end_min_chars"] if is_whisper else None,
               # Copy-then-translate replies keep each line's English on that
               # line; jp-subs' plain format shifted lines on fragmented speech.
               "echo": True,
               "context_before": args.context_before, "context_after": args.context_after}

    if args.limit:
        # By cue, as jp-subs does, so segmentation still sees natural boundaries.
        transcript = {**transcript, "cues": transcript["cues"][: args.limit]}
        print(f"(limited to the first {args.limit} cues)")

    print(f"\ntranslating with {args.llm_model} (gap {gap_ms} ms)")
    started = time.perf_counter()
    result = run(transcript, ollama_backend(model=args.llm_model), options,
                 log=lambda m: print(f"  {m}", flush=True))
    seconds = time.perf_counter() - started

    units, translations = result["units"], result["translations"]
    en_srt = out_dir / f"{stem}.en.srt"
    en_srt.write_text(units_to_srt(units, translations), encoding="utf-8")
    write_json(out_dir / f"{stem}.en.json", {
        "title": transcript.get("title"),
        "source": transcript.get("source"),
        "backend": "ollama",
        "model": args.llm_model,
        "gap_ms": gap_ms,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "glossary": result["glossary"],
        "units": [{"t_ms": u["start_ms"], "end_ms": u["end_ms"], "ja": u["ja"],
                   "en": translations[i] or None} for i, u in enumerate(units)],
    })
    written = [en_srt, out_dir / f"{stem}.en.json"]
    if args.bilingual:
        both = out_dir / f"{stem}.ja-en.srt"
        both.write_text(units_to_srt(units, translations, japanese=True), encoding="utf-8")
        written.append(both)

    print(f"\n  {result['translated']}/{len(units)} units translated in {seconds:.0f}s")
    if result["failures"]:
        print(f"\n  {len(result['failures'])} problem(s):")
        for f in result["failures"]:
            print(f"    - {f}")
    for path in written:
        print(f"  {path}")
    return 0 if result["translated"] else 1


def main(argv: list[str] | None = None) -> int:
    # The Windows console defaults to cp1252 and cannot print Japanese.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # not on pytest's capture objects
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(sys.argv[1:] if argv is None else argv)

    src: Path = args.input
    if not src.is_file():
        print(f"error: no such file: {src}", file=sys.stderr)
        return 1
    from_json = src.name.endswith(".ja.json")
    if from_json and args.ja_only:
        print("error: --ja-only with a .ja.json input leaves nothing to do", file=sys.stderr)
        return 1
    out_dir = args.out or src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.name.removesuffix(".ja.json") if from_json else src.stem

    if not args.ja_only:
        # Before transcribing: a missing model should cost seconds, not an hour.
        from whisper_subs.ollama import BackendError, check_model
        try:
            check_model(args.llm_model)
        except BackendError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1

    if from_json:
        transcript = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(transcript.get("cues"), list):
            print(f"error: {src.name} has no \"cues\" array", file=sys.stderr)
            return 1
    else:
        transcript = transcribe_step(src, out_dir, stem, args)
        if transcript is None:
            return 1

    if args.ja_only:
        return 0
    return translate_step(transcript, out_dir, stem, args)


if __name__ == "__main__":
    sys.exit(main())
