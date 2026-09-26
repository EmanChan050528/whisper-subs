"""Whisper transcription, returning plain JSON-serialisable data.

The raw result is saved as `<stem>.whisper.json` so cue building, and later
translation, can be re-run without transcribing again — transcription is the
slow, GPU-hungry step and everything after it is cheap to iterate on.
"""

import gc
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from whisper_subs.audio import SAMPLE_RATE
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


def transcribe(
    audio: np.ndarray,
    opts: Options,
    on_progress: Callable[[float, float], None] = lambda done, total: None,
    log: Callable[[str], None] = lambda msg: None,
) -> dict:
    """Returns {"options", "duration", "segments": [...]} with times in seconds."""
    device, compute = resolve_device(opts)
    from faster_whisper import WhisperModel

    common = {
        "language": opts.language,
        "beam_size": opts.beam_size,
        "word_timestamps": True,
        "condition_on_previous_text": opts.condition_on_previous_text,
        "initial_prompt": opts.initial_prompt,
        "no_speech_threshold": opts.no_speech_threshold,
        "log_prob_threshold": opts.log_prob_threshold,
        "hallucination_silence_threshold": opts.hallucination_silence_threshold,
    }
    duration = len(audio) / SAMPLE_RATE
    model = WhisperModel(opts.model, device=device, compute_type=compute)
    try:
        segments, info = model.transcribe(
            audio, vad_filter=opts.vad, vad_parameters={"threshold": opts.vad_threshold},
            **common,
        )
        out = []
        # The generator does the actual decoding; consuming it is the slow part.
        for s in segments:
            out.append(_segment_dict(s))
            on_progress(min(s.end, duration), duration)

        if opts.gap_fill:
            out = _gap_fill(model, audio, out, opts, common, log)
    finally:
        # Free VRAM before anything else (Ollama, later) wants it.
        del model
        gc.collect()

    return {
        "options": {**asdict(opts), "device": device, "compute_type": compute},
        "language": info.language,
        "duration": round(duration, 3),
        "segments": out,
    }


def _gap_fill(model, audio, segments, opts, common, log) -> list[dict]:
    """Second pass over speech the first pass skipped.

    Decoding a 30 s window of voice acting under music, Whisper sometimes jumps
    its timestamp past the speech: on NSY6YHXbxtA, 22% of the lines YouTube
    captioned had no words at all, and turning every skip threshold off changed
    nothing. Re-transcribing just those stretches, found with Silero VAD,
    recovers most of them (22% -> 10% missed) for ~30 s on a 22 min video.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    speech = [(r["start"] / SAMPLE_RATE, r["end"] / SAMPLE_RATE)
              for r in get_speech_timestamps(audio, VadOptions(
                  threshold=opts.gap_fill_vad_threshold, min_silence_duration_ms=500,
                  speech_pad_ms=200))]
    words = sorted((w["start"], w["end"]) for s in segments for w in s["words"])
    holes = find_holes(speech, words, opts.gap_fill_min_s)
    if not holes:
        return segments
    log(f"gap fill: re-transcribing {len(holes)} stretch(es) of missed speech, "
        f"{sum(b - a for a, b in holes):.0f}s in total")

    duration = len(audio) / SAMPLE_RATE
    added = []
    for a, b in holes:
        # A second of padding gives Whisper context; words in the padding are
        # already transcribed, so only words inside the hole are kept.
        a0, b0 = max(0.0, a - 1.0), min(duration, b + 1.0)
        found, _ = model.transcribe(
            audio[int(a0 * SAMPLE_RATE):int(b0 * SAMPLE_RATE)], vad_filter=False, **common)
        for s in found:
            seg = _segment_dict(s, offset=a0,
                                keep=lambda ws, we, a=a, b=b: we > a - 0.2 and ws < b + 0.2)
            if seg:
                seg["gap_fill"] = True
                added.append(seg)
    log(f"gap fill: recovered {len(added)} segment(s)")
    return sorted(segments + added, key=lambda s: s["start"])
