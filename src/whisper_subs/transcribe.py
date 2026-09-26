"""Whisper transcription, returning plain JSON-serialisable data.

The raw result is saved as `<stem>.whisper.json` so cue building, and later
translation, can be re-run without transcribing again — transcription is the
slow, GPU-hungry step and everything after it is cheap to iterate on.
"""

import gc
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import numpy as np

from whisper_subs.audio import SAMPLE_RATE, blocks_of
from whisper_subs.gpu import enable_cuda_dlls

# Hugging Face's Xet transfer hung at 0 bytes on the first large-v3 download
# (docs/benchmarks.md); plain HTTP was reliable. Symlink warnings on Windows
# are harmless noise.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DEFAULT_MODEL = "large-v3"
FAST_MODEL = "large-v3-turbo"


@dataclass
class Options:
    model: str = DEFAULT_MODEL
    device: str = "auto"
    compute_type: str = "auto"
    language: str = "ja"
    beam_size: int = 5
    # Off by default. On NSY6YHXbxtA (game voice acting under a reaction video)
    # Silero VAD discarded 43% of the lines YouTube's captions have; without it,
    # 22%. On an hour of plain conversation, VAD-off added only short real
    # interjections and no hallucinations. It also drops sung vocals entirely.
    # See docs/benchmarks.md. Worth enabling for audio with long silences.
    vad: bool = False
    # Silero's speech probability cut-off (faster-whisper default 0.5).
    vad_threshold: float = 0.5
    # On long noisy audio, conditioning on previous text lets one hallucination
    # repeat itself for minutes.
    condition_on_previous_text: bool = False
    initial_prompt: str | None = None
    # Words Whisper should expect. Not set by the CLI: given a series' character
    # names, Whisper recited the list over music in place of the real speech
    # (see glossary.py). Kept for experiments.
    hotwords: str | None = None
    # Whisper drops a whole 30 s window as silence when no_speech_prob exceeds
    # this AND the text's avg_logprob is below log_prob_threshold. None never
    # drops. faster-whisper's defaults are 0.6 and -1.0.
    no_speech_threshold: float | None = 0.6
    log_prob_threshold: float | None = -1.0
    # With word timestamps: skip silent stretches longer than this many seconds
    # around a suspected hallucination. None disables.
    hallucination_silence_threshold: float | None = None
    # Second pass over speech the first pass skipped (see _gap_fill). Measured
    # on NSY6YHXbxtA: VAD threshold 0.2 and holes of 1 s or more worked best.
    gap_fill: bool = True
    gap_fill_vad_threshold: float = 0.2
    gap_fill_min_s: float = 1.0
    # Long files are transcribed in windows of about this many seconds, each
    # saved to a checkpoint as it finishes, so a crash loses one window at most.
    # Whisper already decodes in 30 s pieces and, with
    # condition_on_previous_text off, carries nothing across them, so windows
    # cut in pauses change little (docs/benchmarks.md).
    window_s: float = 600.0


def resolve_device(opts: Options) -> tuple[str, str]:
    enable_cuda_dlls()
    import ctranslate2

    device = opts.device
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute = opts.compute_type
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def _segment_dict(s, offset: float = 0.0, keep=None) -> dict | None:
    """faster-whisper Segment -> plain dict, shifted by `offset` seconds.
    `keep(start, end)` filters words; a segment left with none is dropped."""
    words = [
        {"start": round(w.start + offset, 3), "end": round(w.end + offset, 3),
         "word": w.word, "p": round(w.probability, 3)}
        for w in (s.words or [])
    ]
    if keep is not None:
        words = [w for w in words if keep(w["start"], w["end"])]
        if not words:
            return None
    return {
        "start": words[0]["start"] if keep is not None else round(s.start + offset, 3),
        "end": words[-1]["end"] if keep is not None else round(s.end + offset, 3),
        "text": ("".join(w["word"] for w in words) if keep is not None else s.text).strip(),
        "avg_logprob": round(s.avg_logprob, 4),
        "no_speech_prob": round(s.no_speech_prob, 4),
        "compression_ratio": round(s.compression_ratio, 3),
        "words": words,
    }


def find_holes(speech: list[tuple[float, float]], words: list[tuple[float, float]],
               min_hole: float, margin: float = 0.5,
               merge: float = 2.0) -> list[tuple[float, float]]:
    """Stretches of detected speech with no transcribed word within `margin` s.

    `speech` and `words` are (start, end) in seconds, both sorted. Holes shorter
    than `min_hole` are ignored; holes closer than `merge` are joined, so one
    re-transcription covers a run of nearby misses.
    """
    holes = []
    for a, b in speech:
        cur = a
        for ws, we in words:
            if we + margin < cur:
                continue
            if ws - margin > b:
                break
            if ws - margin > cur:
                holes.append((cur, ws - margin))
            cur = max(cur, we + margin)
        if cur < b:
            holes.append((cur, b))
    merged: list[tuple[float, float]] = []
    for h in (h for h in holes if h[1] - h[0] >= min_hole):
        if merged and h[0] - merged[-1][1] < merge:
            merged[-1] = (merged[-1][0], h[1])
        else:
            merged.append(h)
    return merged


def speech_map(audio: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    """Silero VAD speech stretches, in seconds. Cheap next to Whisper itself."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    return [(r["start"] / SAMPLE_RATE, r["end"] / SAMPLE_RATE)
            for r in get_speech_timestamps(audio, VadOptions(
                threshold=threshold, min_silence_duration_ms=500, speech_pad_ms=200))]


def _gaps(speech: list[tuple[float, float]], end: float) -> list[tuple[float, float]]:
    """Non-speech stretches between (and around) speech, up to `end`."""
    if not speech:
        return [(0.0, end)]
    inner = [(speech[i][1], speech[i + 1][0]) for i in range(len(speech) - 1)]
    return [g for g in [(0.0, speech[0][0]), *inner, (speech[-1][1], end)] if g[1] > g[0]]


def choose_cut(mark: float, gaps: list[tuple[float, float]], search_s: float,
               after: float, margin: float = 0.25) -> float:
    """The point inside a pause nearest `mark` (within search_s, and later than
    `after`), at least `margin` from the speech either side, so no word
    straddles the cut. In a short pause that is its middle; in a long silence,
    the mark itself. With no pause nearby, the mark."""
    best = None
    for a, b in gaps:
        pad = min(margin, (b - a) / 2)
        t = min(max(mark, a + pad), b - pad)
        if abs(t - mark) <= search_s and t > after + 1 and (best is None or
                                                          abs(t - mark) < abs(best - mark)):
            best = t
    return round(mark if best is None else best, 3)


def plan_windows(duration: float, speech: list[tuple[float, float]], window_s: float,
                 search_s: float = 60.0) -> list[tuple[float, float]]:
    """All the windows for a fully known file. transcribe() makes the same cuts
    as the audio streams in; this is the whole-file view of that rule."""
    gaps = _gaps(speech, duration)
    edges = [0.0]
    mark = window_s
    while mark < duration - window_s / 4:  # no tiny last window
        cut = choose_cut(mark, gaps, search_s, edges[-1])
        if cut > edges[-1] + 1:
            edges.append(cut)
        mark = edges[-1] + window_s
    edges.append(round(duration, 3))
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def transcribe(
    audio: np.ndarray | Iterable[np.ndarray],
    opts: Options,
    on_progress: Callable[[float, float], None] = lambda done, total: None,
    log: Callable[[str], None] = lambda msg: None,
    resume: dict | None = None,
    on_checkpoint: Callable[[dict], None] = lambda state: None,
    check_stop: Callable[[], None] = lambda: None,
    duration: float | None = None,
) -> dict:
    """Returns {"options", "duration", "windows", "segments": [...]}, times in s.

    `audio` is a whole array, or blocks as they are decoded (audio.stream), so
    a long file is never held in memory: each ~window_s window is transcribed
    once enough audio has arrived to choose its cut, then dropped. `duration`
    (for progress, and to avoid a tiny last window) is exact for an array and
    the container's estimate for a stream.

    After each window `on_checkpoint(state)` receives a JSON-serialisable
    state; passing it back as `resume` skips windows already done, provided
    the cuts come out the same (they do for the same file and options).
    `check_stop()` is called between segments; it stops the run by raising.
    """
    if isinstance(audio, np.ndarray):
        duration = len(audio) / SAMPLE_RATE
        blocks: Iterable[np.ndarray] = blocks_of(audio)
    else:
        blocks = audio
    total = duration or 0.0
    window_s = opts.window_s
    search_s = min(60.0, window_s / 2)

    device, compute = resolve_device(opts)
    common = {
        "language": opts.language,
        "beam_size": opts.beam_size,
        "word_timestamps": True,
        "condition_on_previous_text": opts.condition_on_previous_text,
        "initial_prompt": opts.initial_prompt,
        "hotwords": opts.hotwords,
        "no_speech_threshold": opts.no_speech_threshold,
        "log_prob_threshold": opts.log_prob_threshold,
        "hallucination_silence_threshold": opts.hallucination_silence_threshold,
    }
    saved = resume or {}
    state = {"windows": [], "done": {}, "language": saved.get("language")}
    model = None
    buf = np.empty(0, dtype=np.float32)
    buf_t0 = 0.0         # time of buf[0]
    seen = 0.0           # seconds decoded so far
    speech: list[tuple[float, float]] = []
    win_start = 0.0
    gap_stats = [0, 0.0, 0]  # stretches, seconds, recovered
    diverged = False     # once a window differs from the checkpoint, none after can match
    reused = 0

    def at(t: float) -> int:
        """Index in buf of time t."""
        return round((t - buf_t0) * SAMPLE_RATE)

    def process(a: float, b: float) -> None:
        nonlocal model, buf, buf_t0, diverged, reused
        i = len(state["windows"])
        key = str(i)
        reusable = (not diverged and saved.get("windows", [])[i:i + 1] == [[a, b]]
                    and key in (saved.get("done") or {}))
        diverged = diverged or not reusable
        reused += reusable
        if reusable:
            found = saved["done"][key]
        else:
            if model is None:
                from faster_whisper import WhisperModel
                model = WhisperModel(opts.model, device=device, compute_type=compute)
            piece = buf[at(a):at(b)]
            segments, info = model.transcribe(
                piece, vad_filter=opts.vad, vad_parameters={"threshold": opts.vad_threshold},
                **common)
            found = []
            # The generator does the actual decoding; consuming it is the slow part.
            for seg in segments:
                check_stop()
                found.append(_segment_dict(seg, offset=a))
                on_progress(min(a + seg.end, total or a + seg.end), total or b)
            state["language"] = state["language"] or info.language
            if opts.gap_fill:
                inside = [(max(x, a), min(y, b)) for x, y in speech if y > a and x < b]
                found = _gap_fill(model, piece, a, found, inside, opts, common, gap_stats,
                                  check_stop)
        state["windows"].append([a, b])
        state["done"][key] = found
        if not reusable:
            on_checkpoint(state)
        on_progress(b, max(total, b))
        # Drop the audio this window used.
        drop = at(b)
        buf, buf_t0 = buf[drop:], buf_t0 + drop / SAMPLE_RATE

    try:
        for block in blocks:
            t = seen
            seen += len(block) / SAMPLE_RATE
            buf = np.concatenate([buf, block]) if buf.size else block
            for x, y in speech_map(block, opts.gap_fill_vad_threshold):
                x, y = x + t, y + t
                if speech and x - speech[-1][1] < 0.05:  # one stretch across blocks
                    speech[-1] = (speech[-1][0], y)
                else:
                    speech.append((x, y))
            # A cut can be chosen once the audio reaches past its search range.
            while seen - win_start >= window_s + search_s:
                mark = win_start + window_s
                if total and total - mark < window_s / 4:
                    break  # no tiny last window; the tail joins this one
                cut = choose_cut(mark, _gaps(speech, seen), search_s, win_start)
                process(win_start, cut)
                win_start = cut
        if seen > win_start:
            process(win_start, round(seen, 3))
    finally:
        if model is not None:
            # Free VRAM before anything else (Ollama, later) wants it.
            del model
            gc.collect()

    if gap_stats[0]:
        log(f"gap fill: re-transcribed {gap_stats[0]} stretch(es) of missed speech "
            f"({gap_stats[1]:.0f}s), recovered {gap_stats[2]} segment(s)")
    if reused:
        log(f"resumed: {reused} of {len(state['windows'])} window(s) were already done")
    return {
        "options": {**asdict(opts), "device": device, "compute_type": compute},
        "language": state["language"],
        "duration": round(seen, 3),
        "windows": state["windows"],
        "segments": [seg for i in range(len(state["windows"])) for seg in state["done"][str(i)]],
    }


def _gap_fill(model, piece, offset, segments, speech, opts, common, stats,
              check_stop=lambda: None) -> list[dict]:
    """Second pass, within one window, over speech the first pass skipped.

    Decoding a 30 s window of voice acting under music, Whisper sometimes jumps
    its timestamp past the speech: on NSY6YHXbxtA, 22% of the lines YouTube
    captioned had no words at all, and turning every skip threshold off changed
    nothing. Re-transcribing just those stretches, found with Silero VAD,
    recovers most of them (22% -> 10% missed) for ~30 s on a 22 min video.
    Done per window while its audio is in memory; windows are cut in pauses,
    so a stretch never straddles two.
    """
    words = sorted((w["start"], w["end"]) for s in segments for w in s["words"])
    holes = find_holes(speech, words, opts.gap_fill_min_s)
    if not holes:
        return segments
    end = offset + len(piece) / SAMPLE_RATE
    added = []
    for a, b in holes:
        check_stop()
        # A second of padding gives Whisper context; words in the padding are
        # already transcribed, so only words inside the hole are kept.
        a0, b0 = max(offset, a - 1.0), min(end, b + 1.0)
        clip = piece[int((a0 - offset) * SAMPLE_RATE):int((b0 - offset) * SAMPLE_RATE)]
        found, _ = model.transcribe(clip, vad_filter=False, **common)
        for s in found:
            seg = _segment_dict(s, offset=a0,
                                keep=lambda ws, we, a=a, b=b: we > a - 0.2 and ws < b + 0.2)
            if seg:
                seg["gap_fill"] = True
                added.append(seg)
    stats[0] += len(holes)
    stats[1] += sum(b - a for a, b in holes)
    stats[2] += len(added)
    return sorted(segments + added, key=lambda s: s["start"])
