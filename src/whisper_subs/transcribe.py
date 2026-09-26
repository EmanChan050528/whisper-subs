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


def transcribe(
    audio: np.ndarray,
    opts: Options,
    on_progress: Callable[[float, float], None] = lambda done, total: None,
) -> dict:
    """Returns {"options", "duration", "segments": [...]} with times in seconds."""
    device, compute = resolve_device(opts)
    from faster_whisper import WhisperModel

    model = WhisperModel(opts.model, device=device, compute_type=compute)
    try:
        segments, info = model.transcribe(
            audio,
            language=opts.language,
            beam_size=opts.beam_size,
            word_timestamps=True,
            vad_filter=opts.vad,
            vad_parameters={"threshold": opts.vad_threshold},
            condition_on_previous_text=opts.condition_on_previous_text,
            initial_prompt=opts.initial_prompt,
        )
        duration = len(audio) / SAMPLE_RATE
        out = []
        # The generator does the actual decoding; consuming it is the slow part.
        for s in segments:
            out.append({
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
                "avg_logprob": round(s.avg_logprob, 4),
                "no_speech_prob": round(s.no_speech_prob, 4),
                "compression_ratio": round(s.compression_ratio, 3),
                "words": [
                    {"start": round(w.start, 3), "end": round(w.end, 3),
                     "word": w.word, "p": round(w.probability, 3)}
                    for w in (s.words or [])
                ],
            })
            on_progress(min(s.end, duration), duration)
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
