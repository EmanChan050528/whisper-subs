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

Planning. See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the step-by-step
build plan.

| Milestone | |
|---|---|
| 0. Setup and GPU check | ☐ |
| 1. CLI MVP: audio in, Japanese `.srt` out | ☐ |
| 2. Translation: port the jp-subs pipeline, English `.srt` | ☐ |
| 3. Quality: eval, timing, glossary, long files | ☐ |
| 4. GUI and packaging | ☐ |

## Requirements (planned)

- Python 3.11 or later
- [Ollama](https://ollama.com) with a translation model (`ollama pull qwen3.5:9b`)
- An NVIDIA GPU is recommended. CPU works with smaller Whisper models.
