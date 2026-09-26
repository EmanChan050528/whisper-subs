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

> **In progress.** Started 2026-09-26 from a viewer report: "a long while with no
> subtitles at all". Scores are in
> [benchmarks.md](benchmarks.md#2026-09-26-milestone-3-missing-speech-phantoms-and-timing).

### 3.0 Missing speech (added: the first thing a viewer noticed)
Only 512 s of the 1,312 s `NSY6YHXbxtA` video had a subtitle, and 148 s of
speech that YouTube captioned fell inside blank stretches of 10 s or more.
- [x] **Diagnosis:** Whisper wasn't discarding windows. Turning off every
      skip threshold (`no_speech_threshold`, `log_prob_threshold`) changed
      nothing (22% missed either way). Decoding 30 s windows of voice acting
      under music, it *jumps its timestamp past the speech*. VAD-on runs catch
      some of these spots but lose others (43% missed).
- [x] **Gap fill** (`transcribe._gap_fill`, on by default, `--no-gap-fill`
      turns it off). After the main pass, Silero VAD (threshold 0.2) finds
      speech with no word within 0.5 s. Stretches of 1 s or more, joined when
      less than 2 s apart and padded by 1 s, are re-transcribed with the same
      loaded model, and only words inside each stretch are kept. **Missed
      lines 22% → 10%**, and speech inside 10 s blank stretches 148 s → 39 s,
      for about 25 s of extra GPU time.
- [x] Rejected: also retrying every word-free stretch over 8 s regardless of
      VAD. It missed more (14%) and hallucinated more.
- [ ] What's left is mostly short interjections (え?, [笑い]) and a few lines
      under loud music. Worth another look after 3.5 (hotwords).

### 3.1 Evaluation harness (build this before tuning anything)
- [x] `eval/score.py` scores the `.srt` as shown: flashes under 1 s, cues over
      7 s, overlaps, reading speed over 20 characters/s, and against a
      reference transcript the lines missed and the speech inside 10 s+ blank
      stretches.
- [x] `eval/coverage.py`: coverage of a `.ja.json` against a reference, with
      the missed and extra lines listed.
- [ ] **Alignment check** (from 2.5): flag translated lines that plausibly
      belong to a neighbour. **Unsolved.** An English-length heuristic caught
      1 of 52 deliberately shifted lines, so it was removed. The next thing to
      try is a model-based back-check with the small `qwen3.5:2b`: "does this
      English translate this Japanese?", in batches.
- [ ] Reuse jp-subs' human-reference English (`eval/reference/`). It covers
      `EmteTL5Ij8g`, which we don't have audio for, so this needs that video
      or a new reference for `NSY6YHXbxtA`.
- [ ] Record every tuning change as before and after numbers. For now they go
      in `docs/benchmarks.md`.

### 3.2 Retune segmentation for Whisper
Done in Milestones 1 and 2: see 1.3b (gap 500 ms, the punctuated prompt
rejected) and 2.5 (`cue_end_min_chars=8`).

### 3.3 Cue timing polish (`timing.py`)
- [x] `display_times()` moves only the **end**, into following silence. The
      rules: at least 1 s, 17 English characters per second, at most 7 s,
      never overlapping, and gaps under 250 ms closed to two frames (83 ms).
      On `NSY6YHXbxtA`: flashes 41 → 5, overlaps 4 → 0, over 20 cps 42 → 20.
- [x] Word ends are already used instead of segment ends (1.3b).
- [ ] Split units longer than 7 s at the widest internal word gap. There are
      3 on the clip. It needs splitting the English too, so maybe pass it to
      the translator as two lines instead.

### 3.4 Hallucination guards (`filters.py`)
Whisper produces predictable junk on silence and music.
- [x] Phantom phrases (ご視聴ありがとうございました, チャンネル登録, 字幕…, 作詞・作曲…)
      are dropped only when the **whole** segment is the phrase **and**
      `no_speech_prob ≥ 0.5`, so a streamer really signing off is kept. On
      the clip it dropped 3, all over music.
- [x] A line of 6+ characters repeated 3+ times in a row is collapsed to its
      first copy. On the clip it collapsed one Whisper loop.
- [x] Dropped segments are listed with the reason in the `.ja.json`
      (`dropped`), so the filters can be audited.
- [ ] ~~Drop segments with `no_speech_prob > 0.6`~~: rejected. Real voice
      lines over music score 0.7 to 0.88.
- [ ] Test on a stream archive with a long BGM-only waiting screen. That's
      still the biggest risk with VAD off.

### 3.5 Glossary per channel or series (`glossary.py`)
jp-subs keys glossaries by YouTube channel ID. Local files have none, so the
user names one.
- [x] `--glossary NAME` uses `%APPDATA%/whisper-subs/glossaries/NAME.json`.
      It seeds pass 1 (jp-subs' "ALREADY ESTABLISHED FOR THIS CHANNEL"), and
      pass 1's findings are merged back, with **existing entries winning**, so
      hand edits stick. Up to 40 entries per category, written atomically.
- [x] Measured on `NSY6YHXbxtA`. Without a glossary, pass 1 renders
      コロンビーナ differently on every run ("Colonnella", then "Colonnibina").
      With a hand-corrected glossary it's **"Columbina" on all 14 lines**, and
      the next merge left the correction alone.
- [x] ~~Feed the names to Whisper as `hotwords`~~: **rejected.** Whisper
      recited the name list over music as 30 s cues, *in place of* the speech,
      and ran 3× slower (383 s, against 120 s). With the recitations filtered
      out, 28% of YouTube's lines were missing, against 10% without hotwords.
      Misheard names are left to pass 1's ASR corrections instead.
      `Options.hotwords` stays for experiments.
- [ ] Optionally infer the name from the parent folder (for example
      `downloads/<streamer>/…`). Not done: an explicit name is clearer.
- [ ] A way to see and edit a glossary without finding the file, perhaps in
      the Milestone 4 GUI.

### 3.6 Long files
- [x] **Windowed transcription.** About 10-minute windows, each cut at the
      middle of the pause nearest its mark (from the Silero map that gap fill
      already uses), with no tiny last window. On the hour clip the cuts fell
      at 598, 1201, 1801, 2400 and 2991 s. The windowed transcript is
      **97.9% identical** to the unwindowed one, with no words split at the cuts.
- [x] **Checkpoints, resumed automatically** (no `--resume` flag needed).
      `<name>.whisper.partial.json` is saved after each window, and
      `<name>.en.partial.json` after pass 1 and after every chunk. A rerun
      continues from them if the source file, options and units match;
      otherwise they're ignored. `--force` discards them. Writes go to a temp
      file and are renamed into place, so a crash can't leave a torn file.
- [x] **Tested by killing the process.** Transcription killed after 2 of 6
      windows resumed with the other 4. Translation killed after 200 of 1,047
      lines resumed without redoing pass 1 or those lines, and finished
      1,047 of 1,047.
- [x] **If Ollama dies mid-run**, every later chunk fails fast and the run
      ends with lines missing. The checkpoint is kept, and the same command
      retries only the missing lines. Tested.
- [x] A resumed run doesn't merge the glossary a second time.
- [ ] Progress is still the CLI's own line, not `tqdm`. That's enough for
      now, and the GUI will need its own anyway.
- [ ] Memory: the whole file is decoded into RAM (about 230 MB per hour).
      That's fine up to several hours, but for a 10-hour archive, decode per
      window instead.

### 3.8 Speed (added: asked for after 3.6)
- [x] Measured where the time goes: translation is bound by output tokens
      (93 tok/s), and echo doubles them.
- [x] `--parallel 2` (the default) keeps a second request queued: translation
      −36% on the clip and −38% on the hour.
- [x] Every transcription unloads whatever Ollama has resident. A leftover
      LLM had made an hour's transcription 22% slower.
- [x] ~~Echo only a prefix~~ and ~~batch gap fill~~: tried and rejected
      (shifted lines, and less speech recovered). See benchmarks.md.

### 3.7 Overlapping speech (keep this scope small)
- [ ] Record it as a known limitation first and measure it on the two-speaker
      sample.
- [ ] Optional stretch: diarization with `pyannote` (it needs a Hugging Face
      token, so keep it opt-in). Use it only to split cues at speaker changes,
      and prefix `-` for dialogue.

**Done when** the scores in `eval/README.md` show before and after
improvement, and a file over an hour long runs end to end with `--resume`
working. **Met** (the scores are in `docs/benchmarks.md`). Still open: the
alignment check (3.1) and overlapping speech (3.7).

---

## Milestone 4: GUI and packaging

> **Built 2026-09-26.** The window, a standalone build, and a README with a
> demo GIF recorded from a real run.

### 4.0 One pipeline, two front ends (`job.py`)
- [x] The CLI's pipeline moved into `job.run_job(src, settings, log,
      progress, should_stop)`. It reports through callbacks instead of
      `print`, and raises `JobError` for things the user can fix and `Stopped`
      when asked to stop. `cli.py` is now just argument parsing plus a printer.
      All existing tests passed unchanged.
- [x] Stop is checked between Whisper segments, gap-fill clips and
      translation chunks. With the 3.6 checkpoints, **Stop is a pause**.

### 4.1 GUI (`gui.py`, `whisper-subs-gui`)
- [x] **PySide6-Essentials** (the Qt Widgets subset). One window: a drop zone
      (files or whole folders, or click to choose), a queue with a status per
      file, Whisper and Ollama model pickers (the latter from `/api/tags`), a
      series glossary picker with **Edit…** (opens the folder), bilingual and
      Japanese-only switches, Start / Stop / Clear finished / Open output
      folder, a progress bar and a log.
- [x] A health line (the 4.2 "preflight"): CUDA ready or not, Ollama running
      or not, and models installed, each with the fix.
- [x] Runs on a `QThread`, and files can be added while it runs. One file's
      error is shown in the list and the queue carries on.
- [x] Preferences are kept in `%APPDATA%\whisper-subs\gui.ini`, **not the
      registry**. The first version used the registry, and its tests
      overwrote the real settings.
- [x] Tests (`tests/test_gui.py`, offscreen, with a fake job): the queue runs,
      errors stay per file, Stop pauses and Start carries on, Japanese-only
      disables the translation settings, and folders expand to their media.
- [x] **Real run through the window**: the minute clip in 27 s. On the
      22-minute clip, Stop during translation left it "Paused"; Start reused
      the transcription and resumed at line 120 of 244, then finished.

### 4.2 Packaging (`packaging/whisper-subs.spec`)
- [x] PyInstaller, one folder, **two programs sharing one set of libraries**:
      `Whisper Subtitler.exe` (windowed) and `whisper-subs.exe` (console).
      Whisper models aren't bundled; they download on first use.
- [x] **cuDNN is not needed.** Listing the DLLs a real transcription maps
      showed only `cublas64_12` and `cublasLt64_12` from the NVIDIA wheels.
      Transcription runs in fp16 and int8 with cuDNN and NVRTC hidden. The
      `cuda` extra is now cuBLAS alone: 1.3 GB less to install and bundle.
- [x] Bundle **1.1 GB**, of which cuBLAS is 736 MB. The rest: Qt 71 MB,
      FFmpeg libraries 63 MB, CTranslate2 59 MB, onnxruntime 36 MB.
- [x] The build tested end to end: the packaged CLI ran the minute clip on
      CUDA in 26 s, and the packaged window starts and stays up.
- [ ] Publish a zip as a GitHub release. It's under the 2 GB asset limit.
- [x] App icon: 字 over a yellow A on a diagonally split blue badge (the
      window's accent colour), picked from four drafts previewed at 256, 64,
      32 and 16 px on light and dark backgrounds. It was chosen because it
      reads best at small sizes. The lettering is Noto Sans JP Bold (SIL OFL
      1.1). `scripts/make_icon.py` redraws it and fetches the font into the
      gitignored `build/`. The icon is used for the window, the taskbar (its
      own AppUserModelID, so it isn't grouped under Python's) and both .exe
      files.
- [x] Version info on both .exe files (Explorer > Properties > Details),
      read from `__version__`.

### 4.3 README and portfolio
- [x] Rewritten as the project's front page: what it does, how it works,
      the measured decisions, speed, install, use, limitations and building.
- [x] Demo GIF (`docs/demo.gif`, 0.4 MB) and screenshot, recorded by
      `scripts/make_demo.py` from a real run of the window on the minute clip,
      which is the user's own sample, so there's no third-party footage.
      The finished frame shows real output lines.
- [x] Cross-linked with jp-subs as a local-AI subtitle toolkit.
- [ ] ~~The video playing with subs~~: the samples are audio. Adding it needs
      footage that can be published.
- [ ] Before and after table against a naive per-line translation. The
      nearest thing now is the line-shift table in benchmarks.md.

---

## Later, if it earns it
- Share `prompt` and `segment` between the two repos from one source, for
  example by generating the JS from Python or the other way round, so they
  can't drift apart.
- Korean, reusing jp-subs' `korean-readiness` probe.
- `.vtt` and `.ass` output.
- Burn in subtitles with ffmpeg.
