# Changelog

Versions were applied retroactively, in the way [jp-subs](https://github.com/EmanChan050528/jp-subs)
did it. Development ran as milestones (see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)), and each
version is the commit where a milestone landed. `0.1.0` is the first one that ships a window and a
standalone build; everything before it is `0.0.x`.

Tags are annotated, so `git show 0.1.0` explains why each one is where it is.

---

## Unreleased — a new window, line-shift check and streaming audio

The window has been rebuilt as a modern app, with file cards that show each stage's progress, switches
instead of checkboxes, and a drop target that covers the whole window. An optional `--check` pass finds
English subtitles that landed on a neighbouring line and re-translates them, and audio is now decoded in
a stream, so an hour-long file no longer needs the whole recording in memory.

The new window is a web page (HTML, CSS and JavaScript) shown by [pywebview](https://pywebview.flowrl.com)
in Windows' own WebView2, replacing the Qt widgets. It follows Windows' light or dark mode, title bar
included, and animates files in and out of the queue. The main button changes between Start, Pause and
Resume, and each finished file lists its subtitle files, which open in Explorer when clicked. The log folds
away to one line. Settings from the old window carry over. Dropping a folder no longer queues a
transcript (`.ja.json`) whose video is also in it. Without Qt the build is about 50 MB smaller.

The check asks a judge model (`qwen3.5:9b`, temperature 0) which neighbouring Japanese line each
English line actually translates. Runs of two or more lines pointing away from their own are shifts,
and `--check` re-translates those plus two lines either side, then judges again. On 342 clean lines it
raised no false alarms, and it found every planted and real shift. `eval/align.py` reports without
changing anything.

Long cues are tidied as well. A stretched final word is capped at 1.5 s (お疲れ様でした went from 9.6 s
to 4.4 s), and cues over 7 s split only at a real pause at a clause break. Single-character fragments from
gap fill are dropped. Audio memory for an hour of speech went from 222 MB to 18 MB, with samples
bit-identical to a whole-file decode. The echo preset is now checked byte for byte against jp-subs as
well, and `.gitattributes` keeps LF line endings in the repo and the working copy.

Before the rebuild, the Qt window was fixed for dark mode, where it painted light backgrounds under
white text. The app build
(`scripts/build_app.py`) keeps PyInstaller's scratch copies of the programs out of `build/`, and
adds a `README.txt` saying which of the two programs is the app, plus a `Whisper Subtitler.lnk`
shortcut in the repo folder for opening it.

## 0.1.0 — desktop app

A drag-and-drop window for anyone who would rather not use a terminal, and a standalone Windows build that
needs no Python installed.

The pipeline now lives in `job.py` as `run_job(...)`, so the command line and the window are two thin front
ends over the same code, and the CLI's behaviour did not change. The window (`whisper-subs-gui`) is a
PySide6 app with a drop zone, a queue, model and glossary pickers and a health line that says whether the
GPU and Ollama are ready. Stop is really pause: press Start again and it carries on from where it left off.

`packaging/whisper-subs.spec` builds a PyInstaller folder with both programs sharing their libraries
(about 1.1 GB). The `cuda` extra installs cuBLAS only, because a transcription never loads cuDNN, which
saves 1.3 GB. The app also gets an icon (字 over A on a split blue badge, lettered in Noto Sans JP Bold)
and version info on both executables. The README was rewritten around a demo GIF and a screenshot
recorded from a real run.

## 0.0.3 — quality and long files

Long recordings are safe to interrupt, and the subtitles are cleaner: missed speech is recovered, invented
sign-offs are dropped, and timing is tidier.

**Missed speech.** A second pass re-transcribes stretches where speech was detected but no words came back.
On the test video this took YouTube-captioned lines missed from 22% to 10%.

**Invented text.** Whisper writes things like "thanks for watching" over music. Those phantom sign-offs and
credits are dropped only when the audio is also non-speech, and looping repeats are collapsed. Dropped
lines are listed in the `.ja.json`, so nothing disappears silently.

**Timing.** Display times get a 1 s minimum and a 17 characters-per-second ceiling, no overlaps and no
flicker gaps. Flashes went from 40 to 5 and overlaps from 3 to 0. `eval/score.py` scores the `.srt` as
displayed, against a reference transcript.

**Series glossary.** `--glossary NAME` keeps names consistent across episodes. It seeds pass 1 and merges
its findings back, and existing entries win, so a hand edit sticks. On the test video, Colonnella and
Colonnibina both settled to Columbina, 14 times.

**Long files.** Transcription runs in windows of about 10 minutes, cut at pauses, with a checkpoint after
each. Translation checkpoints after pass 1 and after every chunk. A rerun resumes automatically when the
source, options and units match, and `--force` starts over. Writes are atomic, and this was tested by
killing the process mid-run.

**Speed.** Translation chunks now run on a thread pool (`--parallel`, default 2), which cut translation time
by 36% on a 22-minute clip and 38% on an hour. Every model Ollama has resident is unloaded before
transcription, because a leftover LLM had slowed an hour's transcription by 22%.

Several ideas were measured and rejected: feeding glossary names to Whisper as hotwords (it recited the
list over music, ran 3x slower and missed 28% of lines), prefix-only echo (lines shifted), batched gap
fill (recovered less speech), and tuning Ollama's parallelism and flash attention (no gain). The numbers
are in [docs/benchmarks.md](docs/benchmarks.md).

## 0.0.2 — translation

Adds English. The translation pipeline is ported from jp-subs to Python, with byte-for-byte parity against
the original JavaScript on 5 fixtures across 2 presets.

The port covers segmenting, chunking, prompts, the two-pass pipeline, the Ollama client and `.srt` output,
plus a small shim that reproduces the UTF-16 lengths and `Math.round` behaviour the JavaScript depends on.
The CLI checks that Ollama is up before starting, hands the GPU over between Whisper and the LLM (unload the
model, free Whisper), writes `.en.srt` and `.en.json`, and adds `--bilingual` for a file with Japanese above
English. It also accepts a `.ja.json` as input, so a transcript can be translated again without
re-transcribing.

The first full run exposed a real bug: translations shifting onto neighbouring lines. The fix was to end
units at Whisper's segment boundaries, and to have the model copy each line's Japanese before its English so
that a mismatched copy can be caught and retried.

## 0.0.1 — Japanese transcription

The first working command: audio or video in, Japanese `.srt` out, using faster-whisper on the GPU.

Audio is decoded with PyAV, so no system ffmpeg is needed, and errors say what went wrong. Whisper
`large-v3` was chosen as the default after benchmarking on an RTX 5070 (`scripts/gpu_check.py`). Word
timestamps are on and voice-activity detection is off by default, since VAD missed 43% of the test
video's YouTube-captioned lines against 22% without it. Whisper's segments sometimes start tens of seconds
before their first word, so cues split on gaps of more than 1 s between words.

The output is also written as a `.ja.json` in jp-subs' format, which means a Whisper transcript can be fed
straight into jp-subs' translator. `eval/coverage.py` compares a transcript against YouTube's own
speech-recognition text, and results are cached against the source file and the options used.
