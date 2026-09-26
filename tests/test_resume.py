"""Interrupted long runs pick up where they stopped."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from whisper_subs import cli, ollama
from whisper_subs import transcribe as T
from whisper_subs.transcribe import Options, plan_windows


def test_windows_are_cut_in_pauses_and_the_last_is_not_tiny():
    speech = [(0, 598), (603, 1190), (1195, 1300)]
    # The mark (600) falls inside the pause 598-603: cut right there.
    assert plan_windows(1300, speech, 600) == [(0.0, 600.0), (600.0, 1300)]
    # A pause just past the mark: its point nearest the mark, 0.25 s clear of speech.
    assert plan_windows(1300, [(0, 604), (606, 1300)], 600)[0] == (0.0, 604.25)
    # 1300 s with a pause near 1200 would leave a 100 s tail: folded in instead.
    assert plan_windows(1300, [(0, 1199), (1201, 1300)], 600)[-1][1] == 1300
    # No pause anywhere near the mark: cut on the mark.
    assert plan_windows(1250, [(0, 1250)], 600) == [(0.0, 600), (600, 1250)]


class Crash(BaseException):  # like Ctrl+C: not caught per chunk
    pass


class FakeWhisper:
    """Stands in for faster_whisper.WhisperModel: one segment per call."""
    calls = 0
    crash_on = None

    def __init__(self, *a, **k):
        pass

    def transcribe(self, audio, **k):
        FakeWhisper.calls += 1
        if FakeWhisper.calls == FakeWhisper.crash_on:
            raise Crash
        n = FakeWhisper.calls
        word = SimpleNamespace(start=0.1, end=0.5, word=f"語{n}", probability=0.9)
        seg = SimpleNamespace(start=0.1, end=0.5, text=f"語{n}", avg_logprob=-0.1,
                              no_speech_prob=0.1, compression_ratio=1.0, words=[word])
        return iter([seg]), SimpleNamespace(language="ja")


def test_transcription_resumes_from_its_last_finished_window(monkeypatch):
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisper)
    monkeypatch.setattr(T, "resolve_device", lambda opts: ("cpu", "int8"))
    monkeypatch.setattr(T, "speech_map", lambda audio, thr: [])
    audio = np.zeros(16000 * 30, dtype=np.float32)
    opts = Options(window_s=10, gap_fill=False)  # 30 s -> 3 windows

    saved = []
    FakeWhisper.calls, FakeWhisper.crash_on = 0, 3  # dies in the third window
    with pytest.raises(Crash):
        T.transcribe(audio, opts, on_checkpoint=lambda st: saved.append(json.loads(json.dumps(st))))
    assert len(saved[-1]["done"]) == 2

    FakeWhisper.calls, FakeWhisper.crash_on = 0, None
    out = T.transcribe(audio, opts, resume=saved[-1])
    assert FakeWhisper.calls == 1  # only the window that never finished
    assert [s["start"] for s in out["segments"]] == [0.1, 10.1, 20.1]


def test_translation_resumes_without_redoing_pass_1_or_finished_chunks(tmp_path, monkeypatch):
    cues = [{"t_ms": i * 2000, "dur_ms": 1000, "ja": f"これは{i}番目の文です"} for i in range(6)]
    src = tmp_path / "long.ja.json"
    src.write_text(json.dumps({"title": "t", "source": "whisper large-v3", "cues": cues},
                              ensure_ascii=False), encoding="utf-8")
    calls = {"analysis": 0, "chunks": 0}
    crash_after = {"chunks": 1}

    def backend(prompt, json_mode=False):
        if prompt.startswith("You are preparing"):
            calls["analysis"] += 1
            return json.dumps({"setting": "Test."})
        if crash_after["chunks"] is not None and calls["chunks"] >= crash_after["chunks"]:
            raise Crash
        calls["chunks"] += 1
        body = prompt.split("LINES TO TRANSLATE\n")[1].split("\n\nRules:")[0]
        answer = {}
        for line in body.split("\n"):
            n, ja = line.split("\t", 1)
            answer[n] = {"ja": ja, "en": f"Line {n}."}
        return json.dumps(answer, ensure_ascii=False)

    monkeypatch.setattr(ollama, "check_model", lambda *a, **k: None)
    monkeypatch.setattr(ollama, "ollama_backend", lambda **k: backend)

    # Crash after the first of three chunks (size 2).
    with pytest.raises(Crash):
        cli.main([str(src), "--size", "2"])
    assert (tmp_path / "long.en.partial.json").exists()
    assert calls == {"analysis": 1, "chunks": 1}

    crash_after["chunks"] = None
    assert cli.main([str(src), "--size", "2"]) == 0
    assert calls == {"analysis": 1, "chunks": 3}  # pass 1 not repeated, chunk 1 not re-sent
    en = json.loads((tmp_path / "long.en.json").read_text(encoding="utf-8"))
    assert [u["en"] for u in en["units"]] == [f"Line {n}." for n in range(1, 7)]
    assert not (tmp_path / "long.en.partial.json").exists()


def test_a_checkpoint_from_different_input_is_ignored(tmp_path, monkeypatch):
    (tmp_path / "x.en.partial.json").write_text(
        json.dumps({"key": "stale", "glossary": {}, "translations": ["old"]}), encoding="utf-8")
    src = tmp_path / "x.ja.json"
    src.write_text(json.dumps({"title": "t", "source": "whisper", "cues": [
        {"t_ms": 0, "dur_ms": 900, "ja": "はい"}]}, ensure_ascii=False), encoding="utf-8")

    def backend(prompt, json_mode=False):
        if prompt.startswith("You are preparing"):
            return json.dumps({"setting": "Test."})
        return json.dumps({"1": {"ja": "はい", "en": "Yes."}}, ensure_ascii=False)

    monkeypatch.setattr(ollama, "check_model", lambda *a, **k: None)
    monkeypatch.setattr(ollama, "ollama_backend", lambda **k: backend)
    assert cli.main([str(src)]) == 0
    en = json.loads((tmp_path / "x.en.json").read_text(encoding="utf-8"))
    assert en["units"][0]["en"] == "Yes."


def test_missing_lines_keep_the_checkpoint_so_a_rerun_retries_only_them(tmp_path, monkeypatch):
    src = tmp_path / "y.ja.json"
    src.write_text(json.dumps({"title": "t", "source": "whisper", "cues": [
        {"t_ms": 0, "dur_ms": 900, "ja": "はい"},
        {"t_ms": 3000, "dur_ms": 900, "ja": "いいえ"}]}, ensure_ascii=False), encoding="utf-8")
    ollama_up = {"ok": False}
    sent = []

    def backend(prompt, json_mode=False):
        if prompt.startswith("You are preparing"):
            return json.dumps({"setting": "Test."})
        body = prompt.split("LINES TO TRANSLATE\n")[1].split("\n\nRules:")[0]
        sent.append(body)
        if not ollama_up["ok"] and "いいえ" in body:
            raise ollama.BackendError("Cannot reach Ollama")  # died mid-run
        answer = {line.split("\t")[0]: {"ja": line.split("\t")[1], "en": "ok"}
                  for line in body.split("\n")}
        return json.dumps({k: v for k, v in answer.items()
                           if ollama_up["ok"] or v["ja"] == "はい"}, ensure_ascii=False)

    monkeypatch.setattr(ollama, "check_model", lambda *a, **k: None)
    monkeypatch.setattr(ollama, "ollama_backend", lambda **k: backend)
    assert cli.main([str(src), "--size", "1"]) == 0          # partial result, exit 0
    assert (tmp_path / "y.en.partial.json").exists()

    ollama_up["ok"], sent[:] = True, []
    assert cli.main([str(src), "--size", "1"]) == 0
    assert sent == ["2\tいいえ"]                              # only the missing line
    assert not (tmp_path / "y.en.partial.json").exists()


def test_streamed_audio_is_sample_identical_to_whole_file(tmp_path):
    import av

    from whisper_subs.audio import check, load, stream

    path = tmp_path / "tone.wav"
    rate, seconds = 44100, 2.5
    t = np.arange(int(rate * seconds)) / rate
    pcm = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    with av.open(str(path), "w") as out:
        s = out.add_stream("pcm_s16le", rate=rate)
        s.layout = "mono"
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.rate = rate
        for packet in s.encode(frame):
            out.mux(packet)
        for packet in s.encode():
            out.mux(packet)

    blocks = list(stream(path, block_s=1.0))
    assert [len(b) for b in blocks] == [16000, 16000, 8000]   # fixed blocks, short last
    assert np.array_equal(np.concatenate(blocks), load(path))
    assert abs(check(path) - seconds) < 0.05
