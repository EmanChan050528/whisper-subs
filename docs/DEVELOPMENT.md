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
- [x] Create a venv with Python 3.11 (`py -3.11 -m venv .venv`). Use 3.11, not
      the 32-bit 3.13 that is also installed.
- [x] Add `pyproject.toml` with a `whisper-subs` console script, the
      `faster-whisper` dependency, and `pytest` plus `ruff` as dev extras.
- [x] Use a `src/whisper_subs/` layout.

### 0.2 Check the GPU. Do this first, because it decides everything else.

> **Done 2026-09-26.** It works on CUDA fp16. See [benchmarks.md](benchmarks.md):
> `large-v3` runs at 12.5× real time and `large-v3-turbo` at 46.5×.
The RTX 5070 is Blackwell (sm_120). CTranslate2, which faster-whisper runs on,
only supports it in recent CUDA 12.8+ builds.
- [x] `pip install faster-whisper`, then load `large-v3` with
      `device="cuda", compute_type="float16"` and transcribe 30 s of audio.
- [x] ~~If CUDA fails~~: the GPU works. The only failure was a missing
      `cublas64_12.dll`, fixed with the `[cuda]` extra. If it had failed, try `compute_type="int8_float16"` and
      the latest `ctranslate2`, and as a last resort fall back to
      `device="cpu", compute_type="int8"` with `small` or `medium`.
- [x] Record the real-time factor (audio seconds per wall-clock second) for
      `small`, `medium`, `large-v3` and `large-v3-turbo` in `docs/benchmarks.md`.

### 0.3 Audio decoding
faster-whisper decodes through **PyAV**, which bundles FFmpeg's libraries, so
mp4, mkv and mp3 files work **without a system ffmpeg**. That matters because
ffmpeg isn't installed on this machine, and it makes packaging much simpler.
- [x] Use `faster_whisper.decode_audio()` first. Confirmed on wav, mp3, mp4
      and mkv with no system ffmpeg.
- [ ] Keep the ffmpeg CLI (`winget install Gyan.FFmpeg`) as an optional
      fallback only, for files PyAV can't open.

### 0.4 Test media

> **Done 2026-09-26** with 3 of 4: a minute of solo speech, a 1-hour two-person
> conversation, and a song. There's no overlapping-speech clip yet. Results and
> the choice of large-v3 as the default are in [benchmarks.md](benchmarks.md).
- [x] Put 3 or 4 clips in `samples/` (it is gitignored), about 2 to 10 minutes
      each: one clean solo talker, one noisy stream with BGM, one with two
      speakers who overlap, and one long archive of at least an hour.
- [x] Bonus: a YouTube VOD that jp-subs already has a fixture for
      (`NSY6YHXbxtA` is `samples/clip-sample.mp3`; `EmteTL5Ij8g` was too long to
      download). Then Whisper can be compared
      against YouTube ASR on the same speech, with a scored English reference.

**Done when** `large-v3` transcribes a 30 s clip on the GPU and the RTF is
written down.

---

## Milestone 1: CLI MVP, audio in and Japanese `.srt` out

> **Built 2026-09-26.** `whisper-subs input.mp4` writes `.ja.srt`, `.ja.json`
> and `.whisper.json`. jp-subs translates the `.ja.json` unchanged, and the
> subtitles play in sync in VLC.
> Measurements are in [benchmarks.md](benchmarks.md#2026-09-26-milestone-1-cue-boundaries-and-vad).

### 1.1 `audio.py`
- [x] `load(path) -> np.ndarray` at 16 kHz mono, through `decode_audio`.
- [x] Give a clear error for files with no audio stream. Tested with a
      video-only mp4.
- [x] mp3, wav, mp4 (mpeg4 and AAC) and mkv (Opus) all decode with no system
      ffmpeg.

### 1.2 `transcribe.py`
- [x] `word_timestamps=True`, `beam_size=5`, `condition_on_previous_text=False`.
- [x] **VAD is off by default** (opt in with `--vad` and `--vad-threshold`). On
      `NSY6YHXbxtA`, VAD made Whisper miss 43% of the lines YouTube's captions
      have, against 22% without it. Without VAD the hour clip gained only real
      interjections and no hallucinations. This makes the 3.4 hallucination
      guards more important.
- [x] Stream the segments generator and show progress as a percentage of the
      audio's duration.
- [x] Free the model (`del` and `gc.collect()`) as soon as transcription ends.

### 1.3 Write `.ja.json` in jp-subs format
- [x] One cue per Whisper segment: `t_ms`, `dur_ms`, `ja`, and `segs` built
      from words. `ja` always equals the concatenated `segs`.
- [x] Top-level fields: `title`, `duration_s`, `source`, `captured_at`,
      `cue_count`.
- [x] Save the raw Whisper output (`.whisper.json`). It's reused if the source
      file (name, size and mtime) and every option that affects output are
      unchanged. `--force` redoes it.

### 1.3b Cue boundaries (moved up from 3.2)
Decided and measured:
- [x] **Split segments where words are more than 1.0 s apart.** Whisper
      sometimes pins a segment's first word tens of seconds before the rest
      (「え?」 at 160.9 s and the rest at 204.3 s). Cue times now come from word
      times, not segment times. This took the clip's units over 10 s from 13
      down to 1 (at a 400 ms gap).
- [x] ~~Punctuated `initial_prompt`~~: **rejected.** On the hour clip, units
      ending in punctuation only went from 8 to 16.
- [x] ~~Append 。 at the end of each segment~~: **not done.** The `.ja.json`
      stays faithful to what Whisper produced. Boundaries are the segmenter's
      job.
- [x] **The M2 `segment.py` needs `gapMs ≈ 500` for Whisper input**, against
      2000 for YouTube. At 500 ms, the mean unit length is 14.5 chars on the
      clip and 15.3 on the hour. YouTube's baseline is 13.6, and 2000 ms gives
      42.6 on the hour. Record this as a Whisper preset, not a changed default.

### 1.4 Japanese `.srt`
- [x] Port `timestamp()`. One block per cue, stretched to at least 700 ms but
      never into the next cue.

### 1.5 CLI
```
whisper-subs input.mp4 [--model M | --fast] [--device auto|cuda|cpu]
                       [--compute-type T] [--vad [--vad-threshold X]]
                       [--prompt TEXT] [--out DIR] [--force] [--ja-only]
```
- [x] `--fast` selects `large-v3-turbo`. `--music` was dropped because VAD is
      off by default.
- [x] UTF-8 stdout and stderr (the Windows console is cp1252).
- [x] `argparse`, with outputs next to the input by default.

### 1.6 Interop check (the payoff)
- [x] jp-subs' `segment.js` reads our `.ja.json` unchanged.
- [x] `jpsub.js translate` on the first 60 cues of `NSY6YHXbxtA`: 28 of 28
      units in 14.6 s on `qwen3.5:9b`, and pass 1 found 5 names, 5 terms and
      2 ASR corrections. That's English subtitles from a local file with no new
      translation code. Two issues showed, both covered later:
      - Units run up to 16 s and 3 lines, because of the 2 s gap. See 1.3b and
        M2.
      - A name comes out two ways (コロンビーナ and コロヴィーナ). Glossary
        hotwords (3.5) are the fix.

**Done when** `whisper-subs sample.mp4 --ja-only` writes a `.ja.srt` that plays
in sync in mpv or VLC, and `jpsub.js translate` accepts the `.ja.json`
unchanged.
- [x] **Watch one `.ja.srt` in a player.** Checked 2026-09-26 in VLC on
      `NSY6YHXbxtA`: the timing works well.

---

## Milestone 2: translation, ported into Python

> **Built 2026-09-26.** `whisper-subs video.mp4` goes from video to English
> `.srt` in one command. `NSY6YHXbxtA` (22 min) took 3¼ min end to end. The
> port matches jp-subs byte for byte on 5 fixtures × 2 presets. Measurements
> are in [benchmarks.md](benchmarks.md#2026-09-26-milestone-2-translation).

### 2.1 Port in dependency order, with a test for each module
1. [x] `ollama.py`: `urllib` only. `think=False`, `num_ctx=16384`,
       `num_predict=8192`, `temperature=0.2`, and every jp-subs error message.
       Plus `parse_json` with the tail-of-reply error, `check_model()` (a
       preflight before transcription) and `unload()`.
2. [x] `chunk.py`
3. [x] `segment.py`, with a `WHISPER` preset: `gap_ms=500` and
       `cue_end_min_chars=8` (a new option, see 2.5). The defaults are
       jp-subs'.
4. [x] `prompt.py`: text kept verbatim, including the JS `${...}` placeholders,
       which are filled in one pass.
5. [x] `pipeline.py`: `analyse` and `translate_units`, with retries,
       `should_stop` and `on_progress`.
6. [x] `srt.py`: `wrap` and `units_to_srt`.
7. [x] `_js.py`: the JS semantics a byte-for-byte port needs. `js_len` counts
       UTF-16 units (𠮷 is 2) and `js_round` rounds halves up. The sentence
       split also emulates a variable-width lookbehind that Python's `re`
       doesn't have.

### 2.2 Parity tests (`tests/parity/`, `tests/test_parity.py`)
- [x] Fixtures: jp-subs' two YouTube transcripts, our Whisper transcript of
      `NSY6YHXbxtA`, a synthetic 3× long file (for pass-1 sampling) and an
      edge-case file (closing brackets, 「!!」, 𠮷, tag-only cues, rounding).
- [x] `dump.mjs` runs jp-subs' **real** `segment`, `chunk`, `run` and `toSrt`
      with a deterministic fake model, which replies uselessly and then
      fenced on pass 1, drops lines to force retries, never answers some
      lines, and wraps some replies in prose. The goldens are committed, so
      the tests need neither Node nor jp-subs.
- [x] pytest compares units, chunks, every log line, every prompt (by hash,
      with a readable diff of the first of each kind), translations, failures
      and the `.srt`: 31 checks.
- [x] Checked that the tests actually fail: a one-space prompt edit failed 10,
      and UTF-16 length semantics failed 4. One deliberate mutant (splitting
      only at 。 and not after 」) survived, because it produces the same units
      on every input.
- [x] ~~Port `srt-parse.test.mjs`~~: not needed, because this project never
      reads `.srt` input. `glossary-apply` moves to 3.5 with the glossary.

### 2.3 Wire it in
- [x] Default pipeline: preflight Ollama, transcribe, segment, pass 1, pass 2,
      then write `.en.srt`, `.en.json` and `.ja.srt`.
- [x] **Free the VRAM between stages.** Whisper's real peak is **~10 GB**
      during transcription, not the ~4 GB measured after loading, so it can't
      share the card with `qwen3.5:9b` (~9 GB at 16k context). An LLM left over
      from an earlier run is unloaded before transcription, and Whisper is
      freed before pass 1. Checked with an `nvidia-smi` trace.
- [x] Flags: `--llm-model`, `--size`, `--context-before`, `--context-after`,
      `--limit`, `--bilingual` and `--gap-ms`.
- [x] ~~`--from-json`~~: passing a `.ja.json` as the input translates it
      directly. That works for our own output or for a jp-subs transcript,
      which gets jp-subs' defaults.

### 2.4 Bilingual `.srt`
- [x] `.ja-en.srt`: Japanese on the first line, wrapped English below.

### 2.5 Line-shift bug, found on the first full run
The model put translations under the wrong line numbers, shifted by 1 to 3
lines, and a subtitle file like that is worse than none. The model rebuilds
whole sentences from fragmented lines and then spreads the English back
across the numbers, putting the spill-over under the next number. This
happened in **5 of 6 runs** on the clip's chunk 1 and **3 of 3** on the hour
clip's chunk 34. Two fixes, both measured:
- [x] **Segmentation:** the `WHISPER` preset ends a unit at a Whisper segment
      boundary once the unit has 8 or more characters (`cue_end_min_chars`).
      Shorter fragments (そう, 外で) still merge. The clip's chunk 1 went from
      5 of 6 shifted to 0 of 12, but the hour clip, casual conversation made
      of short fragments, still shifted.
- [x] **Copy-then-translate replies (`echo`):** the model returns
      `{"n": {"ja": "<the line, copied>", "en": "..."}}`. Writing the source
      line immediately before its English anchors the translation, and a copy
      that doesn't match its line (`echo_matches`, ≥ 0.7 similarity) is
      rejected and retried. The hour clip's chunk 34 went from **3 of 3
      shifted to 3 of 3 aligned**, and the stress case (clip chunk 1 on the
      old segmentation) from 1 of 3 to 0 of 3. Fragments are now translated
      as fragments ("having a lot of it was / thought to be a bad thing.").
      The CLI turns it on, and parity tests still cover jp-subs' plain format.
- [ ] 3.1 still needs an alignment check in the eval, so a regression shows
      up in the numbers. Worth porting `echo` back to jp-subs.

**Done when** one command turns `sample.mp4` into an English `.srt`, and the
parity tests pass on both jp-subs fixtures. **Met.**

---

## Milestone 3: quality

### 3.1 Evaluation harness (build this before tuning anything)
- [ ] **Alignment check** (from 2.5): flag translated lines that plausibly
      belong to a neighbour. The line-shift bug is invisible in every other
      metric.
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
- [ ] **More urgent now that VAD is off by default (1.2).** Test on a stream
      archive with a long BGM-only waiting screen.
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
- [ ] Transcribe in about 10-minute windows, cut at word gaps since VAD is
      off. Checkpoint each one to `.whisper.json`, so a crash at 3 hours
      doesn't lose everything.
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
