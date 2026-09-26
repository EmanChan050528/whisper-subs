# whisper-subs

Subtitles for any Japanese video or audio file, generated entirely on your own
machine. It needs no cloud services and no API key, and nothing is charged per
file.

```
whisper-subs input.mp4
```

1. **Transcribe.** faster-whisper produces Japanese text with timings for each
   word.
2. **Translate.** A two-pass local LLM pipeline on Ollama, from
   [jp-subs](https://github.com/EmanChan050528/jp-subs). Pass 1 reads the whole
   transcript and builds a glossary of names, terms and speech-recognition
   fixes. Pass 2 translates in chunks, with context on both sides of each chunk.
3. **Write.** It outputs an English `.srt`, and optionally a bilingual JP/EN
   `.srt`.

It works on the videos that jp-subs can't handle, the ones with no caption
track: local files, Niconico and Bilibili downloads, and stream archives.
Together the two projects form one local-AI subtitle toolkit.

## Status

In development. See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the step-by-step
build plan.

| Milestone | |
|---|---|
| 0. Setup and GPU check | ✅ |
| 1. CLI MVP: audio in, Japanese `.srt` out | ✅ |
| 2. Translation: port the jp-subs pipeline, English `.srt` | ✅ |
| 3. Quality: eval, timing, glossary, long files | ☐ |
| 4. GUI and packaging | ☐ |

## Usage

```bash
whisper-subs video.mp4               # video.en.srt, plus video.ja.srt and video.ja.json
whisper-subs video.mp4 --bilingual   # also video.ja-en.srt (Japanese above English)
whisper-subs video.mp4 --ja-only     # Japanese only; no Ollama needed
whisper-subs video.ja.json           # translate again without re-transcribing
whisper-subs video.mp4 --fast        # large-v3-turbo: ~4x faster, weaker on rare words
whisper-subs video.mp4 --limit 60    # translate only the first 60 cues, to try a model
```

A 22-minute video takes about 4 minutes on an RTX 5070: 2 to transcribe, 2 to
translate. Whisper and the LLM take turns on the GPU. See
[docs/benchmarks.md](docs/benchmarks.md).

The translation pipeline is a Python port of
[jp-subs](https://github.com/EmanChan050528/jp-subs), tested byte for byte
against the original. It adds one fix of its own. The model copies each
Japanese line before translating it, which stops translations sliding onto
neighbouring lines in fragmented conversation, and lets a mismatched copy be
caught and retried.

## Requirements

- Windows, with Python 3.11 or later
- [Ollama](https://ollama.com) with a translation model (`ollama pull qwen3.5:9b`)
- An NVIDIA GPU is recommended. CPU works with smaller Whisper models.

## Development setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,cuda]"
```

The `cuda` extra installs cuBLAS and cuDNN as pip wheels, so no system CUDA
toolkit is needed. Leave it out on a machine without an NVIDIA GPU.

Check the GPU and measure speed on any audio or video file:

```bash
.venv/Scripts/python scripts/gpu_check.py path/to/clip.mp4 --language ja
```

Whisper models download on first use (large-v3 is 2.9 GB). If a download hangs
at 0 bytes, set `HF_HUB_DISABLE_XET=1`. Results are in
[docs/benchmarks.md](docs/benchmarks.md).
