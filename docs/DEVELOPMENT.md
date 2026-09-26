# Development plan

Step-by-step build order for Whisper Subtitler. Each step ends in something you
can run and check. Don't start a step until the one before it passes its
**Done when** check.

The companion project is [jp-subs](https://github.com/EmanChan050528/jp-subs)
(local checkout: `D:\Desktop\Translator Project`). Most of the translation work
is already solved there. This project adds audio, transcription and timing.

---

## The key decision: how to reuse jp-subs

jp-subs' core is **dependency-free JavaScript** (`extension/src/core/*.js`,
about 700 lines). This project is Python. There are two ways to reuse it, and
the plan uses both, one after the other:

1. **Interop first.** Whisper output is written as a jp-subs `.ja.json`
   transcript. jp-subs' own CLI can translate that file without any changes:
   ```bash
   node "D:\Desktop\Translator Project\core\bin\jpsub.js" translate out/video.ja.json
   ```
   That gives English subtitles at the end of Milestone 1, before any
   translation code exists here.
2. **Port second.** The core is then ported to Python module by module, with
   the prompts copied **verbatim**. Parity tests run both implementations on
   jp-subs' eval fixtures and require identical results. The prompts were tuned
   against measured failure categories, so they should never be "improved"
   during the port.

### Why the `.ja.json` format fits so well

jp-subs cues look like this:

```json
{ "t_ms": 8400, "dur_ms": 3279, "ja": "…", "segs": [{ "text": "…", "t_ms": 8400 }, …] }
```

`segs` holds word-level timings. On YouTube only about half of all cues have
them, and `segment.js` estimates timings for the rest. faster-whisper's
`word_timestamps=True` gives a timing for every word. If each Whisper segment
becomes a cue and each Whisper word becomes a `seg`, then **every** sentence
break lands on a real timestamp. That was the measured timing win in jp-subs
(22 of 78 unit starts corrected, the worst by 4.3 s).

### Module map

| jp-subs (JS) | whisper-subs (Python) | Notes |
|---|---|---|
| `core/segment.js` | `whisper_subs/segment.py` | Needs retuning for Whisper (see step 3.2) |
| `core/chunk.js` | `whisper_subs/chunk.py` | Direct port |
| `core/prompt.js` | `whisper_subs/prompt.py` | **Verbatim.** Only the word "video" may become "audio/video" |
| `core/pipeline.js` | `whisper_subs/pipeline.py` | Direct port, including retries and the empty-glossary warning |
| `core/backends.js` (Ollama and `parseJson`) | `whisper_subs/ollama.py` | Keep `think: false`, `num_ctx`, and the tail-of-reply error |
| `core/srt.js` (`toSrt`, `wrap`) | `whisper_subs/srt.py` | Plus a bilingual writer |
| `background.js` channel glossary | `whisper_subs/glossary.py` | On disk, not in `chrome.storage` |
| — | `whisper_subs/audio.py`, `transcribe.py` | New |

---

## Milestone 0: setup (about half a day)

### 0.1 Tooling
- [ ] Create a venv with Python 3.11 (`py -3.11 -m venv .venv`). Use 3.11, not
      the 32-bit 3.13 that is also installed.
- [ ] Add `pyproject.toml` with a `whisper-subs` console script, the
      `faster-whisper` dependency, and `pytest` plus `ruff` as dev extras.
- [ ] Use a `src/whisper_subs/` layout.

### 0.2 Check the GPU. Do this first, because it decides everything else.
The RTX 5070 is Blackwell (sm_120). CTranslate2, which faster-whisper runs on,
only supports it in recent CUDA 12.8+ builds.
- [ ] `pip install faster-whisper`, then load `large-v3` with
      `device="cuda", compute_type="float16"` and transcribe 30 s of audio.
- [ ] If CUDA fails, record the error, try `compute_type="int8_float16"` and
      the latest `ctranslate2`, and as a last resort fall back to
      `device="cpu", compute_type="int8"` with `small` or `medium`.
- [ ] Record the real-time factor (audio seconds per wall-clock second) for
      `small`, `medium`, `large-v3` and `large-v3-turbo` in `docs/benchmarks.md`.

### 0.3 Audio decoding
faster-whisper decodes through **PyAV**, which bundles FFmpeg's libraries, so
mp4, mkv and mp3 files work **without a system ffmpeg**. That matters because
ffmpeg isn't installed on this machine, and it makes packaging much simpler.
- [ ] Use `faster_whisper.decode_audio()` first.
- [ ] Keep the ffmpeg CLI (`winget install Gyan.FFmpeg`) as an optional
      fallback only, for files PyAV can't open.

### 0.4 Test media
- [ ] Put 3 or 4 clips in `samples/` (it is gitignored), about 2 to 10 minutes
      each: one clean solo talker, one noisy stream with BGM, one with two
      speakers who overlap, and one long archive of at least an hour.
- [ ] Bonus: a YouTube VOD that jp-subs already has a fixture for
      (`EmteTL5Ij8g` 30 to 40 min, `NSY6YHXbxtA`). Then Whisper can be compared
      against YouTube ASR on the same speech, with a scored English reference.

**Done when** `large-v3` transcribes a 30 s clip on the GPU and the RTF is
written down.

---

## Milestone 1: CLI MVP, audio in and Japanese `.srt` out

### 1.1 `audio.py`
- [ ] `load(path) -> np.ndarray` at 16 kHz mono, through `decode_audio`.
- [ ] Give a clear error for files with no audio stream.

### 1.2 `transcribe.py`
- [ ] Call `WhisperModel.transcribe(audio, language="ja", word_timestamps=True, vad_filter=True, beam_size=5)`.
- [ ] Make `condition_on_previous_text=False` the default. With it on, Whisper
      tends to loop and repeat hallucinations on long, noisy streams.
- [ ] Stream the segments generator so progress shows while it runs. Use
      `info.duration` for the percentage.

### 1.3 Write `.ja.json` in jp-subs format
- [ ] One cue per Whisper segment: `t_ms`, `dur_ms`, `ja`, and `segs` built
      from words.
- [ ] Top-level fields: `title` (the file stem), `duration_s`,
      `source: "whisper <model>"`, `cue_count`.
- [ ] Also save the raw Whisper output (`.whisper.json`) so later steps can be
      re-run without transcribing again.

### 1.4 Japanese `.srt`
- [ ] Port `srt.js` `timestamp()` and write one block per Whisper segment. This
      gives a quick check on transcription quality.

### 1.5 CLI
```
whisper-subs input.mp4 [--model large-v3] [--device cuda|cpu] [--out DIR]
                       [--ja-only]
```
- [ ] Use `argparse`. Outputs go next to the input by default.

### 1.6 Interop check (the payoff)
- [ ] Run jp-subs' `jpsub.js segment` and then `translate` on the `.ja.json`.
      This is English subtitles from a local file, with no new translation code.

**Done when** `whisper-subs sample.mp4 --ja-only` writes a `.ja.srt` that plays
in sync in mpv or VLC, and `jpsub.js translate` accepts the `.ja.json`
unchanged.

---

## Milestone 2: translation, ported into Python

### 2.1 Port in dependency order, with a test for each module
1. [ ] `ollama.py`: `chat(prompt, json=True)` over `urllib` or `httpx` to
       `/api/chat`. Keep `think=False`, `num_ctx=16384`, `num_predict=8192`
       and `temperature=0.2`. Port every error message, because they came from
       real failures. Port `parse_json` too, including the tail-of-reply error.
2. [ ] `chunk.py`
3. [ ] `segment.py`
4. [ ] `prompt.py`: copy the text exactly and diff it against the JS output.
5. [ ] `pipeline.py`: `analyse` (sample down to 6000 chars, retry 3 times, loud
       warning when the glossary is empty) and `translate_units` (retry only the
       missing lines, `on_progress`).
6. [ ] `srt.py`: `wrap` (42 chars, 3 lines) and `to_srt` (700 ms minimum dwell).

### 2.2 Parity tests (`tests/parity/`)
- [ ] Copy `eval/fixtures/*.ja.json` from jp-subs into `tests/fixtures/`.
- [ ] Add a small Node script that dumps `segment()`, `chunk()` and both
      prompts to JSON for each fixture. pytest checks that the Python output
      matches it exactly.
- [ ] Port `srt-parse.test.mjs` and `glossary-apply.test.mjs` as pytest tests.

### 2.3 Wire it in
- [ ] Default pipeline: transcribe, then segment, then pass 1, then pass 2,
      then write `.en.srt`, `.en.json` and `.ja.srt`.
- [ ] **Free the VRAM between stages.** `large-v3` (about 3 to 4 GB in fp16)
      and `qwen3.5:9b` (6.6 GB) together are close to the 12 GB limit. Delete
      the Whisper model and call `gc.collect()` before pass 1 starts.
- [ ] New flags: `--llm-model qwen3.5:9b`, `--size`, `--context-before`,
      `--context-after`, `--limit`, `--bilingual`.
- [ ] `--from-json video.ja.json` skips transcription, for re-running
      translation cheaply.

### 2.4 Bilingual `.srt`
- [ ] Put Japanese on line 1 and English below it in each block. Wrap only the
      English.

**Done when** one command turns `sample.mp4` into an English `.srt`, and the
parity tests pass on both jp-subs fixtures.

---

## Milestone 3: quality

### 3.1 Evaluation harness (build this before tuning anything)
- [ ] `eval/` with a `score.py` that reports: cue count, mean and max
      chars/cue, how many cues exceed 17 chars/sec (reading speed), how many
      are shorter than 700 ms or longer than 7 s, and gaps under 80 ms.
- [ ] Reuse jp-subs' human-reference English (`eval/reference/`) for the
      YouTube-sourced clip, and score the same 7 failure categories.
- [ ] Record every tuning change as before and after numbers in `eval/README.md`.
      This is the same discipline jp-subs used.

### 3.2 Retune segmentation for Whisper
Whisper's output isn't shaped like YouTube's, so check both assumptions:
- [ ] **Punctuation.** `segment.py` splits on 。！？. Whisper's Japanese
      punctuation is inconsistent, and smaller models often drop it. Measure
      what share of units end in punctuation (`stats()` already reports this).
- [ ] **Gap threshold.** `gapMs=2000` was set for YouTube's coarse cues. With
      word-level timings a smaller pause (roughly 500 to 800 ms) is a reliable
      clause boundary. Sweep the value and score each setting.
- [ ] If punctuation is too sparse, try an `initial_prompt` written in
      punctuated Japanese (Whisper copies the style it is primed with), and
      fall back to gaps plus `maxChars`.

### 3.3 Cue timing polish (`timing.py`)
- [ ] Minimum duration of 700 ms, maximum of about 7 s. Split long units at
      the widest internal word gap.
- [ ] Close gaps under about 250 ms to prevent flicker, and never overlap cues.
- [ ] Measure reading speed on the **English** text. Stretch the end into
      following silence where there is room.
- [ ] Trim Whisper's trailing silence: word ends are more reliable than
      segment ends.

### 3.4 Hallucination guards
Whisper produces predictable junk on silence and music.
- [ ] Drop segments with `no_speech_prob > 0.6` and a low `avg_logprob`.
- [ ] Drop or collapse repeated lines, meaning the same text 3 or more times in
      a row.
- [ ] Blocklist the known phantom phrases (ご視聴ありがとうございました, チャンネル登録…)
      when they fall inside low-energy audio.
- [ ] Tune the VAD parameters (`min_silence_duration_ms`, `speech_pad_ms`) on
      the noisy BGM sample.

### 3.5 Glossary per channel or series
jp-subs keys glossaries by YouTube channel ID. Local files have no channel ID, so:
- [ ] `--glossary NAME` uses `%APPDATA%/whisper-subs/glossaries/NAME.json`.
      This is the same merge logic as `rememberChannelGlossary`: pass-1
      findings are added, hand edits are never overwritten.
- [ ] Optionally infer the name from the parent folder (for example
      `downloads/<streamer>/…`).
- [ ] Feed the glossary names into Whisper's `hotwords` / `initial_prompt`, so
      names are **recognised** correctly as well as translated consistently.
      This is a new win jp-subs couldn't get.

### 3.6 Long files
- [ ] Transcribe in about 10-minute windows (VAD-aligned) and checkpoint each
      one to `.whisper.json`, so a crash at 3 hours doesn't lose everything.
- [ ] Add `--resume`, which skips windows and translation chunks already done.
- [ ] Show progress with `tqdm` (percent of audio, then chunks).

### 3.7 Overlapping speech (keep this scope small)
- [ ] Record it as a known limitation first and measure it on the two-speaker
      sample.
- [ ] Optional stretch: diarization with `pyannote` (it needs a Hugging Face
      token, so keep it opt-in). Use it only to split cues at speaker changes,
      and prefix `-` for dialogue.

**Done when** the scores in `eval/README.md` show before and after
improvement, and a file over an hour long runs end to end with `--resume`
working.

---

## Milestone 4: GUI and packaging

### 4.1 GUI
- [ ] Use **PySide6** (it handles drag-and-drop natively and looks right on
      Windows), or `tkinterdnd2` if the bundle size matters.
- [ ] One window: a drop zone, model pickers (Whisper and Ollama, the latter
      filled from `/api/tags`), a glossary name, a bilingual checkbox, a
      progress bar and log, a Stop button, and "Open output folder".
- [ ] Run the pipeline on a `QThread` and use the existing `on_progress` and
      `should_stop` hooks. The core needs no changes.
- [ ] A queue for several files dropped at once.

### 4.2 Packaging
- [ ] PyInstaller one-folder build. Download Whisper models on first run (to
      the HF cache), and don't bundle them.
- [ ] A preflight screen that checks whether Ollama is reachable, whether the
      model is pulled, and whether CUDA is available, with the fix for each.

### 4.3 README and portfolio
- [ ] Demo GIF: drop a file, show progress, then the video playing with subs.
- [ ] Before and after table from the eval, with Whisper and the pipeline set
      against a naive per-line translation.
- [ ] Cross-link with jp-subs as a "local-AI subtitle toolkit".

---

## Later, if it earns it
- Share `prompt` and `segment` between the two repos from one source, for
  example by generating the JS from Python or the other way round, so they
  can't drift apart.
- Korean, reusing jp-subs' `korean-readiness` probe.
- `.vtt` and `.ass` output.
- Burn in subtitles with ffmpeg.
