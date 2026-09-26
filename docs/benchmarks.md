# Benchmarks

Measured with `scripts/gpu_check.py`, settings `beam_size=5`,
`word_timestamps=True`, `vad_filter=True`. The RTF column is audio seconds per
wall-clock second, so higher is faster. VRAM is the extra memory in use while
the model is loaded, measured with `nvidia-smi`.

## 2026-09-26: RTX 5070 (12 GB, Blackwell sm_120), driver 610.74

Stack: faster-whisper 1.2.1, CTranslate2 4.8.2, cuBLAS 12.9 and cuDNN 9.26
(from pip wheels, see `whisper_subs/gpu.py`), Python 3.11.9.

Clip: `samples/tts_en_check.wav`, 99.3 s of **English** Windows TTS. No
Japanese voice is installed, so this measures speed and whether the GPU works
at all, not Japanese accuracy. The encoder cost doesn't depend on language.
Decoding cost does depend somewhat on tokens per second, so re-measure on real
Japanese samples.

| Model | Compute | Load s | Run s | RTF | VRAM MiB |
|---|---|---:|---:|---:|---:|
| small | float16 | 0.8 | 3.5 | 28.7× | ~800 |
| medium | float16 | 1.2 | 6.1 | 16.3× | ~2,000 |
| large-v3-turbo | float16 | 1.2 | 2.1 | **46.5×** | ~2,100 |
| large-v3-turbo | int8_float16 | 1.8 | 2.5 | 40.3× | ~1,100 |
| large-v3 | float16 | 2.6 | 8.0 | 12.5× | ~4,100 |
| large-v3 | int8_float16 | 3.8 | 10.2 | 9.7× | ~2,100 |

The first run of the day was slower (`small` at 10.3×) because of CUDA
initialisation and cuDNN autotuning, so warm runs are the ones reported.

### What this means

- **The GPU works.** Blackwell runs fine on CTranslate2 4.8.2 once cuBLAS and
  cuDNN are available. There's no need for a CPU fallback on this machine.
- **Speed isn't the constraint.** Even `large-v3` does a 1-hour archive in
  about 5 minutes, and turbo in about 80 s. Pick the model on Japanese
  accuracy, which Milestone 1 measures on real samples.
- **int8 halves the VRAM and costs 15 to 20% speed.** It's only worth using
  when sharing the GPU.
- **Keep one model on the GPU at a time.** With `qwen3.5:9b` loaded in Ollama
  (5.5 GB at its default 4k context), `large-v3` fp16 still fit and ran at
  11.3×, with about 11.2 GB in use out of 12.2 GB. The translation pipeline
  runs Ollama at `num_ctx=16384`, which needs more KV cache than that, so
  running both at once risks Ollama spilling layers to the CPU. Unload Whisper
  before pass 1, as the plan says (Milestone 2.3).

### Gotchas hit during setup

1. `RuntimeError: Library cublas64_12.dll is not found`. CTranslate2's Windows
   wheel doesn't bundle cuBLAS or cuDNN. The fix is
   `pip install -e ".[cuda]"`, and `enable_cuda_dlls()` adds the wheels' `bin`
   directories to the DLL search path.
2. The `large-v3` download hung at 0 bytes through Hugging Face's Xet
   transfer. Setting `HF_HUB_DISABLE_XET=1` fixed it: 2.9 GB in about 6 minutes.
3. Hugging Face symlink warnings on Windows are harmless (the cache just uses
   more disk). Silence them with `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

## 2026-09-26: Japanese samples (same machine and stack)

All runs: `--language ja`, fp16. Transcripts are saved to `samples/out/`,
which is gitignored.

| Sample | Length | Content | large-v3 RTF | turbo RTF |
|---|---:|---|---:|---:|
| `minute-sample.mp3` | 51 s | Prepared solo speech, clean | 11.5× | 45.0× |
| `hour-sample.mp3` | 57.7 min | Two-person conversation, casual | **14.0×** (247 s) | **53.4×** (65 s) |
| `song-sample.mp3` | 4.3 min | Vocaloid song (Hatsune Miku) | 19.1×\* | 67.9×\* |

\* With VAD off. With VAD on, both models return **0 segments** for the song.

### Accuracy: large-v3 beats turbo, so large-v3 is the default

On the minute clip:

| | large-v3 | large-v3-turbo |
|---|---|---|
| 今、日本の東京に | ✓ | 今日本の (lost the comma, which changes the reading) |
| 三ヶ国語を喋れます | 三角語を喋れます | 三角語を**しております** |
| 和歌にも見られる | ✓ | **若**にも |
| これを見て | ✓ | **この世**を見て |
| Errors in 51 s | 1 | 4 |

On the hour clip, the two transcripts are almost identical line for line
(1,194 against 1,210 segments, and 11,265 against 11,195 characters). Casual
conversation is easy for both. Turbo's errors show up on less common words
(和歌, 三ヶ国語), which is exactly where names and subject terms occur.

**Decision:** `large-v3` is the default, with `large-v3-turbo` behind a `--fast`
flag. A 1-hour file takes 4 minutes with large-v3, which is fine. The
translation pass will take longer than that anyway.

### Finding 1: Whisper doesn't punctuate conversation

| | Segments ending in 。！？ | Sentence marks in the whole hour |
|---|---:|---:|
| large-v3 | 7 / 1,194 (1%) | 8 |
| large-v3-turbo | 3 / 1,210 (0%) | 4 |

The prepared speech in the minute clip was punctuated, although large-v3 still
lost it partway through. The casual conversation was not punctuated at all.

**This breaks jp-subs' main segmentation rule.** `segment.js` ends a unit at
sentence-final punctuation. With Whisper output that rule almost never fires,
so units would be cut only by the 2 s gap or the 64-character cap. Whisper's
own segments are already short and pause-aligned (mean 2.1 s, with roughly
one per utterance), which may be the better unit. **Step 3.2 has moved up into
Milestone 1**, because it decides how cues are built.

Options to measure:
1. Treat each Whisper segment as ending a sentence, and use `segment.py` only
   to merge very short fragments (for example そう, 外で).
2. Prime with a punctuated `initial_prompt` so Whisper writes 。 again.
3. Lower `gapMs` to about 500 to 800 ms and use word-level timings.

### Finding 2: VAD deletes songs, and Whisper hallucinates credits

- Silero VAD classes singing over music as non-speech, so it drops the whole
  song. For music content, VAD has to be off or tuned: add a `--music` preset.
- With VAD off, both models invent **作詞・作曲・編曲 初音ミク** (lyrics,
  composition and arrangement credits) over the instrumental intro. Turbo does
  it twice and adds an English phrase, "You are there". This is the phantom
  text problem from Milestone 3.4, and credit lines belong on the blocklist
  next to ご視聴ありがとうございました.
- Both models also miss the first verse. Treat song lyrics as best effort, not
  a supported case.

### Finding 3: the Windows console can't print Japanese

`print()` of Japanese raised `UnicodeEncodeError` (the console uses cp1252).
The CLI has to call `sys.stdout.reconfigure(encoding="utf-8")`, as
`gpu_check.py` now does.

### Other notes

- PyAV decoded mp3 fine with no system ffmpeg, including the whole hour in
  2.2 s. mp4 and mkv are still untested.
- Very short segments: 47 under 0.5 s with large-v3, and 61 with turbo. These
  are single words like そう and 外で. They need merging or a minimum display
  time (Milestone 3.3).
- No ご視聴ありがとうございました hallucinations appeared in the hour of
  conversation, with VAD on.

## To do

- [ ] Try mp4 and mkv input.
- [ ] Measure options 1 to 3 from Finding 1 on the hour clip.
