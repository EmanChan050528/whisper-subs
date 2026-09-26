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

## 2026-09-26: Milestone 1, cue boundaries and VAD

`NSY6YHXbxtA` is now in the samples as `clip-sample.mp3` (1,312 s). jp-subs
has YouTube's own ASR transcript of it (`eval/fixtures/NSY6YHXbxtA_full.ja.json`),
so for the first time Whisper can be compared with YouTube's captions on the
same speech. Coverage is measured with `eval/coverage.py`, and units with
jp-subs' own `segment.js`.

### VAD loses a lot of speech, so it's now off by default

A YouTube line counts as "missed" if no Whisper cue overlaps it in time.

| Setting | Whisper cues | YouTube lines missed | Characters missed | Whisper-only lines |
|---|---:|---:|---:|---:|
| VAD on (threshold 0.5) | 185 | 86 / 200 (43%) | 39% | 53 |
| VAD threshold 0.25 | 223 | 61 (30%) | 30% | 57 |
| VAD threshold 0.1 | 206 | 74 (37%) | 36% | 58 |
| **VAD off** | **241** | **43 (22%)** | **19%** | 64 |

- The missed lines are mostly quiet in-game voice acting under the streamer's
  voice, for example 84 to 109 s: やっぱり間に合わなかったのかな,
  名前を教えてくれる?
- Lowering the threshold doesn't behave predictably: 0.1 did worse than 0.25,
  because the chunking changes what Whisper decodes. Only turning VAD off
  helped reliably.
- The "Whisper-only" lines are mostly **real** game dialogue that YouTube's
  captions miss (時間とともに変化していく…). YouTube isn't ground truth.
- **Hour clip, VAD off:** 1,240 against 1,194 segments. There were no phantom
  phrases, no repeated lines, and nothing with a high `no_speech_prob` or
  compression ratio. The 23 new cues are all short real interjections
  (そう, まあね, ありがとうございます, …). Speed was the same.
- **Risk:** long silence or music with VAD off is where Whisper invents text
  (the song's credit lines). The Milestone 3.4 guards have to cover this, and
  `--vad` is still available.

### Whisper segments sometimes have wrong times

A segment can pin its first word far away from the rest. For example, the
segment 「え?俺じゃないって言った」 runs from 160.6 to 205.9 s, with 「え?」 at
160.9 s and 「俺」 at **204.3 s**. As a subtitle it would appear 45 s early and
stay up for 45 s. The clip had 13 segments over 10 s, and the longest was
80 s. **Fix:** `cues.py` splits segments where words are more than 1.0 s apart,
and times every cue from its words.

### Cue boundaries

jp-subs' `segment.js` merges cues into translation units until it sees
sentence-final punctuation, a silence of `gapMs` or longer, or 64 characters.
Whisper barely punctuates, so the gap does all the work:

| Input | gapMs | Units | Mean chars | ≥ 60 chars (cut mid-sentence) | Units > 10 s |
|---|---:|---:|---:|---:|---:|
| YouTube ASR, NSY6YHXbxtA (baseline) | 2000 | 235 | 13.6 | 2 | 14 |
| Whisper, clip (VAD off) | 2000 | 108 | 22.5 | 7 | 16 |
| Whisper, clip (VAD off) | **500** | 167 | **14.5** | 2 | 5 |
| Whisper, hour (VAD off) | 2000 | 263 | 42.6 | 57 | 162 |
| Whisper, hour (VAD off) | **500** | 732 | **15.3** | 3 | 26 |

- **A punctuated `initial_prompt` didn't help.** Units ending in punctuation
  went from 8 to 16 on the hour clip, and from 11 to 27 on the clip.
- **Decision:** the `.ja.json` stays faithful to Whisper. The Python
  `segment.py` (M2) gets a Whisper preset with `gapMs≈500`. Until then, jp-subs'
  CLI works on our files but produces long units, because it uses 2000.

### jp-subs interop

`node jpsub.js translate clip-sample.ja.json --limit 60` translated 28 of 28
units in 14.6 s on `qwen3.5:9b`, and pass 1 found 5 names, 5 terms and 2 ASR
corrections. The translations read well. The problems are the long units (up
to 16 s and 3 lines) and one name that Whisper spells two ways (コロンビーナ and
コロヴィーナ).

**Start Ollama through the app, not with `ollama serve` from a shell.** A bare
`ollama serve` started from Git Bash listed **no models**, because the app sets
the model location itself.

### Container checks

mp4 (mpeg4 video and AAC) and mkv (mpeg4 and Opus) test files, built from the
minute clip, both transcribe fine. A video-only mp4 gives
`error: silent.mp4 has no audio stream.`

## 2026-09-26: Milestone 2, translation

Stack: `qwen3.5:9b` on Ollama 0.33.3 (`num_ctx` 16384, `think` off,
temperature 0.2), with large-v3 for transcription. Everything runs on the
RTX 5070.

### End to end

| Input | Transcribe | Units | Translate | Total | Lines translated |
|---|---:|---:|---:|---:|---:|
| minute (51 s) | 10 s | 7 | 18 s | ~30 s | 7 / 7 |
| `NSY6YHXbxtA` (22 min) | 117 s | 220 | 118 s | ~4 min | 220 / 220 |
| hour (57.7 min, from `.ja.json`) | (cached) | 1,032 | 541 s | — | 1,032 / 1,032 |

The 22-minute and hour runs use the final settings: the `WHISPER`
segmentation preset and `echo`. Pass 1 on the hour sampled 1,032 units down
to 509 (5,992 characters).

Pass 1 repairs ASR errors from context. On the minute clip, Whisper's 三角語
(a mishearing of 三ヶ国語) came out as "trilingual". On `NSY6YHXbxtA`, the
glossary mapped the misheard コロヴィーナ to "Colombine".

### VRAM: stages can't share the card

An `nvidia-smi` trace every 3 s during the 22-minute run:

| Phase | VRAM used |
|---|---:|
| Start (qwen3.5:9b still loaded from the previous run) | 11.5 GB |
| After `unload()`, Whisper transcribing | 8.7–10.0 GB (peak) |
| Whisper freed | 2.4 GB (desktop baseline) |
| qwen3.5:9b translating | 8.4–8.6 GB |

Whisper's **peak** during transcription is about 7.5 GB above baseline, much
more than the ~4 GB measured after load in Milestone 0. So the two models
can't share 12 GB, and the unload before transcription and release after it
are both needed.

### Line-shift bug and its fixes

A shifted line is a translation filed under a neighbouring line's number. It
was found by reading the first full output: between 55 s and 118 s every
subtitle showed the line before's English, or the one before that. It's
invisible in every metric so far, since every line still has English.

The tests below repeat one chunk several times and check marker lines by hand.

| Chunk | Setting | Shifted runs |
|---|---|---:|
| clip, chunk 1 | jp-subs format, gap 500 | **5 / 6** |
| clip, chunks 1–2 | + `cue_end_min_chars=8` | 0 / 12 |
| hour, chunk 34 | + `cue_end_min_chars=8` | **3 / 3** |
| hour, chunk 34 | + `echo` | **0 / 3** |
| clip, chunk 1, old segmentation (stress) | jp-subs format | 1 / 3 |
| clip, chunk 1, old segmentation (stress) | `echo` | 0 / 3 |

Full runs with `echo`:

| | Retry events | Lines rejected by the copy check | Lines lost | Translate time |
|---|---:|---:|---:|---:|
| clip, without echo | 4 | — | 0 | 123 s |
| clip, with echo | **0** | 0 | 0 | 118 s |
| hour, without echo | 25 | — | 1 | 405 s |
| hour, with echo | **12** | **2** (both genuine shifts, retried) | **0** | 541 s |

Echo costs about a third more translation time on long conversation, and
nothing measurable on the clip. That's worth it: without it, the hour's
subtitles were wrong for whole stretches.

## 2026-09-26: Milestone 3, missing speech, phantoms and timing

This round started from a viewer report: "a long while with no subtitles at
all". The scores come from `eval/score.py` on the `.en.srt` as displayed, with
YouTube's captions as the reference. Those captions aren't ground truth, but
they're a fair check of speech we failed to show.

### NSY6YHXbxtA: before and after

| | M2 output | + gap fill + filters | + timing polish |
|---|---:|---:|---:|
| Subtitles | 220 | 247 | 247 |
| YouTube lines with no subtitle | 43 (22%) | 21 (10%) | 21 (10%) |
| YouTube speech inside 10 s+ blank stretches | 148 s | 42 s | **39 s** |
| Flashes under 1 s | 40 | 41 | **5** |
| Over 7 s | 2 | 3 | 3 |
| Overlapping the next subtitle | 3 | 4 | **0** |
| Faster than 20 characters/s | 40 | 42 | **20** |

End to end with everything on: 4 min 24 s for the 22-minute video. That's
transcription 120 s including gap fill, and translation 138 s for 247 units.

### Why the speech was missing

| Transcription setting | YouTube lines missed |
|---|---:|
| Default (VAD off) | 22% |
| `no_speech_threshold=None` | 24% |
| + `log_prob_threshold=None` | 22% |
| + `hallucination_silence_threshold=2` | 22% |
| VAD on (for comparison) | 43% |
| **Gap fill**, Silero 0.2, holes ≥ 1 s | **10%** |
| Gap fill, Silero 0.3, holes ≥ 1 s | 12% |
| Gap fill, Silero 0.15, holes ≥ 0.5 s | 12% |
| Gap fill, plus every word-free stretch > 8 s | 14% |

The thresholds make no difference, so Whisper isn't discarding windows. It
skips past speech while decoding: from 60.5 s it jumps straight to 86 s,
losing five lines of voice acting under music. Re-transcribing just the
stretches where Silero hears speech but we have no words recovers them. For
example, 64 s クータルが象徴していたすべてを捨てた and 70 s
私もこれからは彼女をコロンビーナと呼ぶべきだろう. The recovered real lines have
`no_speech_prob` of 0.7 to 0.88 because of the music, so that score can't be
used to filter.

### Hallucination filters

On the clip they dropped ご視聴 (134 s), ご視聴ありがとうございました (805 s) and
ご視聴 (1,107 s), all in gap-filled audio over music, plus 2 copies of a line
Whisper looped at 1,186 s. Nothing real was dropped.

### Gap fill on the hour of conversation

- 72 stretches were re-transcribed (126 s of audio), recovering 73 segments.
  They're nearly all real short interjections: そうですね, じゃあ, なるほど,
  うーん.
- Nothing was dropped as a phantom, and no invented sentences appeared.
- A couple of single-character fragments (シ, 最) got through.
- Cost: transcription 223 s → 329 s (+47%), from the many small re-runs. On
  the game clip it cost about 25 s.

### Glossary (3.5)

**Hotwords, tried and rejected.** A glossary seeded from the clip's own pass-1
names (ケンマシェ, コロンビーナ, サンドローネ, ファデュイ, アルレッキーノ…) was
passed to Whisper as `hotwords`:

| | No hotwords | Hotwords | Hotwords, recitations filtered |
|---|---:|---:|---:|
| Transcription time | 120 s | **383 s** | — |
| Cues | 271 | 174 | 148 |
| Cues over 10 s | 1 | **19** | 5 |
| Low-confidence segments (logprob < -1) | 5 | 29 | — |
| YouTube lines missed | 10% | 4% (inflated by long cues) | **28%** |
| コロンビーナ spelled consistently | 11 of 12 | 24 of 24 | — |

Whisper recited the list itself over music ("アルレッキーノ ファデュイ サンドローネ
ファデュイ アルレッキーノ ロザリン" for 29 s), 21 times, instead of the speech
underneath. Better name spelling isn't worth that, and pass 1 already repairs
misheard names.

**Seeding pass 1** is the part that works. Here is how コロンビーナ was
translated on the 14 lines that contain it:

| Run | Rendering |
|---|---|
| No glossary, run 1 | Colonnella ×14 |
| No glossary, run 2 | Colonnibina ×14 |
| `--glossary` with the entry hand-corrected to "Columbina" | **Columbina ×14** |

### Long files (3.6)

Tested on the hour clip, windowed at ~600 s:

| Test | Result |
|---|---|
| Windowed vs unwindowed transcript | 1,269 vs 1,313 segments, 11,398 vs 11,455 characters, **97.9% text similarity**; clean text at all 5 cuts |
| Transcription killed after 2 of 6 windows, rerun | "resuming: 2 of 6 window(s) already done", finished in 187 s |
| Translation killed after 200 of 1,047 lines, rerun | "resuming: pass 1 and 200/1047 lines already done", **1,047 / 1,047** in 383 s |

During the resumed translation Ollama's runner failed once (HTTP 500,
"connection was forcibly closed"). The chunk's automatic retry recovered all
20 lines.

## 2026-09-26: Speed

Asked: why is it slower than jp-subs? First, where the time goes.

### Translation is bound by output tokens

Three hour-clip chunks, with Ollama's own counters:

| Reply format | Wall time | Prompt read | Output written |
|---|---:|---:|---:|
| jp-subs plain | 16.1 s | 3,540 tokens in 0.8 s | 840 tokens in 9.0 s |
| echo (copy each line, then translate) | 25.2 s | 3,851 tokens in 0.9 s | 1,683 tokens in 18.1 s |

Output runs at **93 tokens/s** and reading the prompt is negligible, so time
follows what the model writes, and echo doubles that. Another ~2 s per request
is dead time between one reply and the next prompt.

### Three speedups tried

| Idea | Result | Kept? |
|---|---|---|
| Echo only the first 6 characters of each line | Hour chunk: **17 lines per run failed the copy check** (the model's fragments slid onto the previous line), one run shifted anyway, and it was *slower* (14–17 s against 8–13 s) | **No** |
| Batch the gap-fill re-runs (`BatchedInferencePipeline`) | Saves 8–22 s, but recovers less speech (13% missed against 10%): it has no temperature fallback | **No** |
| Keep a second translation request queued | 6 chunks: 48.0 s → 39.2 s | **Yes**, `--parallel 2` |

Ollama itself runs with `OLLAMA_NUM_PARALLEL=1`. Tested on a temporary second
server (same model files, port 11435), letting it decode 2 or 3 requests at
once gave nothing more. The GPU is nearly full with the model loaded (11.5 GB):

| Server setting | 6 chunks |
|---|---:|
| parallel 1, client sends 1 at a time | 48.0 s |
| parallel 1, client keeps 2 in flight | **39.2 s** |
| parallel 2 | 39.9 s |
| parallel 2 + flash attention | 38.1 s |
| parallel 3 + flash attention | 40.4 s |

So no Ollama settings need changing.

**Gap fill was not the hour's +47%.** It takes 14 s on the clip and 33 s on
the hour, and Silero's speech map takes 3 s. The 329 s hour run had
qwen3.5:9b still loaded (`--ja-only` skipped the unload). Clean, the hour
transcribes in **270 s**. Fix: every transcription now unloads whatever
Ollama has resident (`ollama.unload_all`).

### Before and after

| | Before | After |
|---|---:|---:|
| `NSY6YHXbxtA` (22 min), end to end | 4 min 24 s | **3 min 23 s** |
| of which transcription | 120 s | 104 s |
| of which translation | 132 s | **85 s** |
| Hour clip, translation | 541 s | **336 s** |
| Hour clip, end to end (clean) | ~14½ min | **~10 min** |

Quality is unchanged. On the clip: 11% of YouTube's lines missed (10%
before, within run-to-run noise), 4 flashes (5), 0 overlaps (0), 19 lines
over 20 cps (20), and 1,047 / 1,047 on the hour.

**Lesson from the measurement:** killing a test `ollama serve` doesn't kill the
`llama-server.exe` runners it started. Three orphans held the GPU and made the
first "after" run take 13 min 53 s (Whisper at 2.2× real time). Stop test
servers with their runners, and check `nvidia-smi` before timing anything.

## 2026-09-26: Milestone 4, packaging

### Which CUDA libraries Whisper actually uses

After a real transcription (large-v3, fp16, with gap fill), these are the DLLs
mapped into the process from `site-packages`:

| Library | Size | Loaded? |
|---|---:|---|
| `nvidia/cublas`: cublas64_12, cublasLt64_12 | 771 MB | **yes** |
| `ctranslate2`: ctranslate2.dll, its own small cudnn64_9.dll shim | 60 MB | yes |
| `nvidia/cudnn` (all of it) | 1.1 GB | **no** |
| `nvidia/cuda_nvrtc` | 179 MB | **no** |

With the cuDNN and NVRTC folders hidden, transcription still runs in fp16 and
in int8_float16 with identical output. So the `cuda` extra is now cuBLAS
alone. Milestone 0 installed cuDNN on common advice, but the error at the time
was only the missing cuBLAS.

### Build size

`dist/whisper-subs/` is **1.1 GB**: cuBLAS 736 MB, Qt 71 MB, FFmpeg
libraries 63 MB, CTranslate2 59 MB, onnxruntime 36 MB, NumPy's OpenBLAS
20 MB, and the rest small. The packaged CLI ran the minute clip end to end on
CUDA in 26 s, the same as from source.

## To do

- [ ] Re-run the Japanese RTF on real mp4/mkv downloads when available.
- [ ] Test VAD-off hallucinations on a stream archive with a long waiting
      screen or BGM-only section.
- [ ] An alignment check in the eval (3.1). The length heuristic failed.
- [ ] Gap fill: batch nearby holes to cut the cost on conversation, and drop
      single-character fragments.
