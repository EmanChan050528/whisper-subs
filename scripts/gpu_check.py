"""Milestone 0.2: can faster-whisper run on this GPU, and how fast?

    python scripts/gpu_check.py samples/clip.wav --models small large-v3
    python scripts/gpu_check.py samples/clip.mp4 --language ja --device cpu --compute-type int8

Prints one row per model: load time, transcribe time, real-time factor (audio
seconds per wall-clock second, higher is faster) and peak VRAM, then a sample
of the transcript so a broken run is obvious.
"""

import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_subs.gpu import enable_cuda_dlls

enable_cuda_dlls()

from faster_whisper import WhisperModel, decode_audio  # noqa: E402

SAMPLE_RATE = 16000


def vram_used_mib() -> int | None:
    """Whole-GPU usage from nvidia-smi. Includes Ollama if it has a model loaded."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
        return int(out.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def run(model_name, audio, args):
    before = vram_used_mib()
    t0 = time.perf_counter()
    model = WhisperModel(model_name, device=args.device, compute_type=args.compute_type)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        audio,
        language=args.language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    # transcribe() is lazy: nothing is decoded until the generator is consumed.
    segments = list(segments)
    transcribe_s = time.perf_counter() - t0
    peak = vram_used_mib()

    del model
    gc.collect()

    duration = len(audio) / SAMPLE_RATE
    return {
        "model": model_name,
        "load_s": load_s,
        "transcribe_s": transcribe_s,
        "rtf": duration / transcribe_s if transcribe_s else float("inf"),
        "vram_mib": (peak - before) if (peak is not None and before is not None) else None,
        "segments": segments,
        "language": info.language,
        "language_prob": info.language_probability,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("audio", type=Path)
    p.add_argument("--models", nargs="+", default=["small", "medium", "large-v3-turbo", "large-v3"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--language", default=None, help="e.g. ja; default auto-detects")
    p.add_argument("--show", type=int, default=3, help="segments of transcript to print")
    args = p.parse_args()

    t0 = time.perf_counter()
    audio = decode_audio(str(args.audio), sampling_rate=SAMPLE_RATE)
    print(f"{args.audio.name}: {len(audio) / SAMPLE_RATE:.1f}s decoded via PyAV "
          f"in {time.perf_counter() - t0:.2f}s")
    print(f"device={args.device} compute_type={args.compute_type}\n")

    rows = []
    for name in args.models:
        print(f"--- {name}")
        try:
            r = run(name, audio, args)
        except Exception as err:  # report and carry on to the next model
            print(f"    FAILED: {type(err).__name__}: {err}\n")
            continue
        rows.append(r)
        print(f"    language {r['language']} ({r['language_prob']:.2f}), "
              f"{len(r['segments'])} segments")
        for s in r["segments"][: args.show]:
            print(f"    [{s.start:6.2f} -> {s.end:6.2f}] {s.text.strip()}")
        print()

    if rows:
        print(f"{'model':<16}{'load s':>8}{'run s':>8}{'RTF':>8}{'VRAM MiB':>10}")
        for r in rows:
            vram = "?" if r["vram_mib"] is None else r["vram_mib"]
            print(f"{r['model']:<16}{r['load_s']:>8.1f}{r['transcribe_s']:>8.1f}"
                  f"{r['rtf']:>7.1f}x{vram:>10}")
    return 0 if len(rows) == len(args.models) else 1


if __name__ == "__main__":
    sys.exit(main())
