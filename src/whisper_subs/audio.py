"""Any audio or video file -> 16 kHz mono float32, which is what Whisper takes.

Decoding goes through PyAV (bundled with faster-whisper), which carries its own
FFmpeg libraries, so no system ffmpeg is needed.
"""

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class AudioError(Exception):
    pass


def load(path: str | Path) -> np.ndarray:
    import av
    from faster_whisper import decode_audio

    path = Path(path)
    if not path.is_file():
        raise AudioError(f"No such file: {path}")

    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioError(f"{path.name} has no audio stream.")
    except av.error.FFmpegError as err:
        raise AudioError(f"Cannot read {path.name}: {err}") from err

    audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    if audio.size == 0:
        raise AudioError(f"{path.name} decoded to zero samples.")
    return audio
