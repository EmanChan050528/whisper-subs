"""Any audio or video file -> 16 kHz mono float32, which is what Whisper takes.

Decoding goes through PyAV (bundled with faster-whisper), which carries its own
FFmpeg libraries, so no system ffmpeg is needed.

`stream()` decodes in fixed-size blocks, so a 10-hour archive never has to fit
in memory at once (16 kHz float32 is ~230 MB per hour). It uses faster-whisper's
own decode steps, so the samples are identical to `load()`'s.
"""

import gc
from collections.abc import Iterator
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class AudioError(Exception):
    pass


def check(path: str | Path) -> float | None:
    """Raise AudioError if the file has no readable audio; return its duration
    in seconds if the container knows it."""
    import av

    path = Path(path)
    if not path.is_file():
        raise AudioError(f"No such file: {path}")
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioError(f"{path.name} has no audio stream.")
            if container.duration:
                return container.duration / 1_000_000
            stream = container.streams.audio[0]
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
            return None
    except av.error.FFmpegError as err:
        raise AudioError(f"Cannot read {path.name}: {err}") from err


def load(path: str | Path) -> np.ndarray:
    """The whole file at once. Fine for short files and tools; the pipeline
    uses stream()."""
    from faster_whisper import decode_audio

    check(path)
    audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    if audio.size == 0:
        raise AudioError(f"{Path(path).name} decoded to zero samples.")
    return audio


def stream(path: str | Path, block_s: float = 60.0) -> Iterator[np.ndarray]:
    """Decode sequentially, yielding float32 blocks of exactly block_s seconds
    (the last may be shorter). Never seeks: seeking into compressed audio is
    where sample-accurate positions stop being guaranteed."""
    import av
    from faster_whisper.audio import _group_frames, _ignore_invalid_frames, _resample_frames

    check(path)
    block = int(block_s * SAMPLE_RATE)
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono",
                                                  rate=SAMPLE_RATE)
    pending: list[np.ndarray] = []
    held = 0
    total = 0
    try:
        with av.open(str(path), mode="r", metadata_errors="ignore") as container:
            frames = _resample_frames(_group_frames(_ignore_invalid_frames(
                container.decode(audio=0)), 500000), resampler)
            for frame in frames:
                samples = frame.to_ndarray().reshape(-1)
                pending.append(samples)
                held += samples.size
                while held >= block:
                    joined = np.concatenate(pending)
                    yield joined[:block].astype(np.float32) / 32768.0
                    total += block
                    pending, held = [joined[block:]], joined.size - block
    finally:
        del resampler
        gc.collect()  # as faster-whisper does: the resampler leaks otherwise
    if held:
        yield np.concatenate(pending).astype(np.float32) / 32768.0
        total += held
    if total == 0:
        raise AudioError(f"{Path(path).name} decoded to zero samples.")


def blocks_of(audio: np.ndarray, block_s: float = 60.0) -> Iterator[np.ndarray]:
    """An in-memory array as the same kind of block stream."""
    block = int(block_s * SAMPLE_RATE)
    for i in range(0, len(audio), block):
        yield audio[i:i + block]
