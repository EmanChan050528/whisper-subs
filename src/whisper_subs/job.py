"""One file through the whole pipeline, for any front end.

The CLI and the GUI both call `run_job`. It reports through two callbacks
instead of printing, and checks `should_stop` between units of work, so a GUI
can show progress and stop cleanly. Because every stage checkpoints (3.6),
stopping is really pausing: running the same file again carries on.
"""

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from whisper_subs import glossary, ollama
from whisper_subs.cues import to_transcript
from whisper_subs.srt import cues_to_srt, units_to_srt
from whisper_subs.timing import display_times
from whisper_subs.transcribe import DEFAULT_MODEL, FAST_MODEL, Options

# Options that change what Whisper produces. A cached .whisper.json is reused
# only if all of these match.
CACHE_KEYS = ("model", "language", "beam_size", "vad", "vad_threshold",
              "condition_on_previous_text", "initial_prompt", "no_speech_threshold",
              "log_prob_threshold", "hallucination_silence_threshold", "gap_fill",
              "gap_fill_vad_threshold", "gap_fill_min_s", "hotwords", "window_s")


class JobError(Exception):
    """Something the user can fix: a missing file, Ollama not running, ..."""


class Stopped(Exception):
    """should_stop() said so. Checkpoints are saved; a rerun resumes."""


@dataclass
class Settings:
    out: Path | None = None
    glossary: str | None = None
    # transcription
    model: str | None = None
    fast: bool = False
    device: str = "auto"
    compute_type: str = "auto"
    vad: bool = False
    vad_threshold: float = 0.5
    prompt: str | None = None
    gap_fill: bool = True
    force: bool = False
    # translation
    ja_only: bool = False
    llm_model: str = ollama.DEFAULT_MODEL
    bilingual: bool = False
    size: int = 20
    parallel: int = 2
    context_before: int = 10
    context_after: int = 6
    limit: int | None = None
    gap_ms: int | None = None
    # After translating, ask a model which line each English line belongs to,
    # and re-translate runs that landed on a neighbour (align.py). About a
    # quarter of a second per line.
    check: bool = False


@dataclass
class Result:
    outputs: list[Path] = field(default_factory=list)
    units: int = 0
    translated: int = 0
    failures: list[str] = field(default_factory=list)
    #: With Settings.check: shifted runs the repair could not fix.
    shifted_left: int = 0

    @property
    def missing(self) -> int:
        return self.units - self.translated


Log = Callable[[str], None]
#: (stage, done, total): stage is "transcribe" (seconds of audio) or
#: "translate" (chunks).
Progress = Callable[[str, float, float], None]


def _quiet(*_args) -> None:
    pass


def source_id(src: Path) -> dict:
    """Identifies the input file, so a cache written for movie.mp4 is never
    reused for movie.mkv, or for movie.mp4 after it has been replaced."""
    st = src.stat()
    return {"name": src.name, "size": st.st_size, "mtime": int(st.st_mtime)}


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None  # missing, or half-written by a crash: start that step over


def load_cached(path: Path, opts: Options, src: Path) -> dict | None:
    """A saved transcription (or checkpoint) made from this file with these options."""
    cached = read_json(path)
    if not cached or cached.get("source") != source_id(src):
        return None
    want = asdict(opts)
    if all(cached.get("options", {}).get(k) == want[k] for k in CACHE_KEYS):
        return cached
    return None


def write_json(path: Path, data, indent: int | None = 2) -> None:
    # Write-then-rename, so a crash mid-write never leaves a torn file behind
    # for the next run to trust.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
    tmp.replace(path)


def whisper_options(s: Settings) -> Options:
    return Options(
        model=FAST_MODEL if s.fast else (s.model or DEFAULT_MODEL),
        device=s.device,
        compute_type=s.compute_type,
        vad=s.vad,
        vad_threshold=s.vad_threshold,
        initial_prompt=s.prompt,
        gap_fill=s.gap_fill,
    )


def run_job(src: Path, s: Settings, log: Log = _quiet, progress: Progress = _quiet,
            should_stop: Callable[[], bool] = lambda: False) -> Result:
    """Media file (or .ja.json transcript) -> subtitle files beside it."""
    src = Path(src)
    if not src.is_file():
        raise JobError(f"no such file: {src}")
    stored = None
    if s.glossary:
        try:
            stored = glossary.load(s.glossary)
        except (ValueError, json.JSONDecodeError) as err:
            raise JobError(f"glossary: {err}") from err
        known = len(stored["names"]) + len(stored["terms"])
        log(f"glossary '{s.glossary}': {known} entries from {stored.get('files', 0)} "
            f"earlier file(s)")
    from_json = src.name.endswith(".ja.json")
    if from_json and s.ja_only:
        raise JobError("Japanese-only with a .ja.json input leaves nothing to do")
    out_dir = s.out or src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.name.removesuffix(".ja.json") if from_json else src.stem

    if not s.ja_only:
        # Before transcribing: a missing model should cost seconds, not an hour.
        try:
            ollama.check_model(s.llm_model)
        except ollama.BackendError as err:
            raise JobError(str(err)) from err

    result = Result()
    if from_json:
        transcript = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(transcript.get("cues"), list):
            raise JobError(f'{src.name} has no "cues" array')
    else:
        transcript = _transcribe(src, out_dir, stem, s, log, progress, should_stop)
        result.outputs.append(out_dir / f"{stem}.ja.srt")

    if s.ja_only:
        return result
    return _translate(transcript, out_dir, stem, s, stored, log, progress, should_stop, result)


def _transcribe(src, out_dir, stem, s, log, progress, should_stop) -> dict:
    """Media file -> jp-subs transcript, writing .whisper.json, .ja.json, .ja.srt."""
    opts = whisper_options(s)
    whisper_path = out_dir / f"{stem}.whisper.json"
    partial_path = out_dir / f"{stem}.whisper.partial.json"
    if s.force:
        partial_path.unlink(missing_ok=True)

    whisper = None if s.force else load_cached(whisper_path, opts, src)
    if whisper:
        log(f"using cached transcription: {whisper_path.name}")
    else:
        from whisper_subs.audio import AudioError, check, stream
        from whisper_subs.transcribe import transcribe

        try:
            duration = check(src)
        except AudioError as err:
            raise JobError(str(err)) from err
        # An LLM left resident by an earlier run competes with Whisper for the
        # GPU: an hour took 329 s with qwen3.5:9b still loaded, 270 s without
        # (docs/benchmarks.md). Also when Japanese-only, which never needs it.
        ollama.unload_all()
        length = f"{duration:.0f}s of audio" if duration else "unknown length"
        log(f"{src.name}: {length}, model {opts.model}")
        partial = load_cached(partial_path, opts, src)

        def checkpoint(state):
            write_json(partial_path, {"source": source_id(src), "options": asdict(opts),
                                      "state": state}, indent=None)

        def stop_check():
            if should_stop():
                raise Stopped

        started = time.perf_counter()
        try:
            whisper = {"source": source_id(src),
                       # Streamed: a long file is never held in memory whole.
                       **transcribe(stream(src), opts, duration=duration,
                                    on_progress=lambda d, t: progress("transcribe", d, t),
                                    log=log, resume=partial["state"] if partial else None,
                                    on_checkpoint=checkpoint, check_stop=stop_check)}
        except AudioError as err:  # e.g. decodes to nothing, found only while streaming
            raise JobError(str(err)) from err
        elapsed = time.perf_counter() - started
        log(f"{len(whisper['segments'])} segments in {elapsed:.0f}s "
            f"({whisper['duration'] / elapsed:.1f}x real time, "
            f"{whisper['options']['device']}/{whisper['options']['compute_type']})")
        write_json(whisper_path, whisper, indent=1)
        partial_path.unlink(missing_ok=True)

    transcript = to_transcript(whisper, title=stem)
    write_json(out_dir / f"{stem}.ja.json", transcript)
    (out_dir / f"{stem}.ja.srt").write_text(cues_to_srt(transcript["cues"]), encoding="utf-8")
    return transcript


def translation_key(units: list[dict], s: Settings, options: dict) -> str:
    """Identifies what a translation checkpoint was made from."""
    basis = {"units": [(u["ja"], u["start_ms"]) for u in units], "model": s.llm_model,
             "options": {k: options[k] for k in ("size", "echo", "context_before",
                                                 "context_after")}}
    return hashlib.sha256(json.dumps(basis, ensure_ascii=False).encode()).hexdigest()


def _translate(transcript, out_dir, stem, s, stored, log, progress, should_stop,
               result: Result) -> Result:
    from whisper_subs.pipeline import analyse, translate_units
    from whisper_subs.segment import WHISPER, segment

    is_whisper = str(transcript.get("source", "")).startswith("whisper")
    gap_ms = s.gap_ms if s.gap_ms is not None else (WHISPER["gap_ms"] if is_whisper else 2000)
    options = {"gap_ms": gap_ms, "size": s.size,
               "cue_end_min_chars": WHISPER["cue_end_min_chars"] if is_whisper else None,
               # Copy-then-translate replies keep each line's English on that
               # line; jp-subs' plain format shifted lines on fragmented speech.
               "echo": True,
               "parallel": s.parallel,
               "context_before": s.context_before, "context_after": s.context_after,
               "should_stop": should_stop}

    if s.limit:
        # By cue, as jp-subs does, so segmentation still sees natural boundaries.
        transcript = {**transcript, "cues": transcript["cues"][: s.limit]}
        log(f"(limited to the first {s.limit} cues)")

    log(f"translating with {s.llm_model} (gap {gap_ms} ms)")
    cues = transcript.get("cues") or []
    if not cues:
        raise JobError("the transcript has no cues")
    units = segment(cues, {"gap_ms": gap_ms, "cue_end_min_chars": options["cue_end_min_chars"]})
    log(f"segmented {len(cues)} cues into {len(units)} units")

    # A long file's translation takes many minutes. Save the glossary after
    # pass 1 and the translations after every chunk, so an interruption costs
    # one chunk, not the run.
    partial_path = out_dir / f"{stem}.en.partial.json"
    key = translation_key(units, s, options)
    partial = None if s.force else read_json(partial_path)
    if partial and partial.get("key") != key:
        partial = None

    def checkpoint(found_glossary, translations):
        write_json(partial_path, {"key": key, "glossary": found_glossary,
                                  "translations": translations}, indent=None)

    backend = ollama.ollama_backend(model=s.llm_model)
    started = time.perf_counter()
    if partial:
        found_glossary = partial["glossary"]
        done = sum(1 for t in partial["translations"] if t)
        log(f"resuming: pass 1 and {done}/{len(units)} lines already done")
    else:
        found_glossary = analyse(units, backend, transcript, log,
                                 seed=glossary.seed(stored) if stored else None)
        checkpoint(found_glossary, [""] * len(units))

    def on_chunk(translations, done, total):
        checkpoint(found_glossary, translations)
        progress("translate", done, total)

    out = translate_units(units, found_glossary, backend, options, log, on_progress=on_chunk,
                          initial=partial["translations"] if partial else None)
    translations = out["translations"]
    if out.get("stopped"):
        checkpoint(found_glossary, translations)
        raise Stopped
    if s.check:
        from whisper_subs.align import check_and_repair
        translations, report = check_and_repair(units, translations, found_glossary, backend,
                                                options, log)
        result.shifted_left = len(report["left"])
    seconds = time.perf_counter() - started

    result.units = len(units)
    result.translated = sum(1 for t in translations if t)
    result.failures = out["failures"]
    if result.missing:
        # Keep the checkpoint: typically Ollama stopped answering partway, and
        # a rerun should retry only these lines, not the whole file.
        checkpoint(found_glossary, translations)
    else:
        partial_path.unlink(missing_ok=True)

    en_srt = out_dir / f"{stem}.en.srt"
    times = display_times(units, translations)
    en_srt.write_text(units_to_srt(units, translations, times=times), encoding="utf-8")
    en_json = out_dir / f"{stem}.en.json"
    write_json(en_json, {
        "title": transcript.get("title"),
        "source": transcript.get("source"),
        "backend": "ollama",
        "model": s.llm_model,
        "gap_ms": gap_ms,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "glossary": found_glossary,
        "units": [{"t_ms": u["start_ms"], "end_ms": u["end_ms"], "ja": u["ja"],
                   "en": translations[i] or None} for i, u in enumerate(units)],
    })
    result.outputs = [en_srt, en_json, *result.outputs]
    if s.bilingual:
        both = out_dir / f"{stem}.ja-en.srt"
        both.write_text(units_to_srt(units, translations, japanese=True, times=times),
                        encoding="utf-8")
        result.outputs.insert(1, both)

    log(f"{result.translated}/{len(units)} units translated in {seconds:.0f}s")
    # Merged once per file: by the run that did pass 1, not again on a resume.
    if s.glossary and found_glossary and not partial:
        where, added = glossary.remember(s.glossary, found_glossary, source=stem)
        log(f"glossary '{s.glossary}': {added} new entr{'y' if added == 1 else 'ies'}, {where}")
    return result
