"""The window's behaviour, with the pipeline replaced by a fake job."""

import os
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from whisper_subs import gui, ollama
from whisper_subs.job import JobError, Result, Stopped


@pytest.fixture
def window(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_SUBS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(ollama, "list_models", lambda *a, **k: ["qwen3.5:9b"])
    app = QApplication.instance() or QApplication([])
    w = gui.MainWindow()
    w.prefs.clear()
    yield w
    w.close()
    app.processEvents()


def wait_until_idle(w, timeout=10):
    app = QApplication.instance()
    end = time.time() + timeout
    while w.running and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert not w.running


def statuses(w):
    return [w.queue.item(i).data(gui.Qt.ItemDataRole.UserRole + 1) for i in range(w.queue.count())]


def test_queue_runs_every_file_and_reports_errors_per_file(window, tmp_path, monkeypatch):
    files = [tmp_path / n for n in ("a.mp3", "bad.mp3", "c.mp3")]
    seen = []

    def fake_job(path, settings, log, progress, should_stop):
        seen.append((path.name, settings.llm_model, settings.bilingual))
        progress("transcribe", 30, 60)
        progress("translate", 1, 2)
        if path.name == "bad.mp3":
            raise JobError("bad.mp3 has no audio stream.")
        out = path.with_suffix(".en.srt")
        return Result(outputs=[out], units=2, translated=2)

    monkeypatch.setattr(gui, "run_job", fake_job)
    window.add_files(files)
    window.add_files(files[:1])  # duplicates are ignored
    window.bilingual.setChecked(True)
    window.start()
    wait_until_idle(window)

    assert statuses(window) == [gui.DONE, gui.FAILED, gui.DONE]
    assert seen == [("a.mp3", "qwen3.5:9b", True), ("bad.mp3", "qwen3.5:9b", True),
                    ("c.mp3", "qwen3.5:9b", True)]
    assert "no audio stream" in window.queue.item(1).text()
    assert window.open_btn.isEnabled() and "error" in window.status.text()

    window.clear_finished()
    assert statuses(window) == [gui.FAILED]


def test_stop_pauses_and_start_carries_on(window, tmp_path, monkeypatch):
    calls = []

    def fake_job(path, settings, log, progress, should_stop):
        calls.append(path.name)
        if len(calls) == 1:
            window.request_stop()  # the user presses Stop mid-file
            assert should_stop()
            raise Stopped
        return Result(outputs=[path.with_suffix(".en.srt")], units=1, translated=1)

    monkeypatch.setattr(gui, "run_job", fake_job)
    window.add_files([tmp_path / "long.mp3", tmp_path / "next.mp3"])
    window.start()
    wait_until_idle(window)
    assert statuses(window) == [gui.PAUSED, gui.WAITING]  # the second never started
    assert "Paused" in window.status.text() and window.start_btn.isEnabled()

    window.start()
    wait_until_idle(window)
    assert calls == ["long.mp3", "long.mp3", "next.mp3"]
    assert statuses(window) == [gui.DONE, gui.DONE]


def test_japanese_only_disables_translation_settings(window):
    window.ja_only.setChecked(True)
    assert not window.llm.isEnabled() and not window.bilingual.isEnabled()
    assert window.settings().ja_only


def test_dropping_a_folder_adds_its_media(window, tmp_path):
    (tmp_path / "ep1.mkv").write_bytes(b"")
    (tmp_path / "ep2.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    assert gui._is_media(tmp_path / "ep1.mkv") and not gui._is_media(tmp_path / "notes.txt")
    window.add_files(sorted(p for p in tmp_path.iterdir() if gui._is_media(p)))
    assert [Path(window.queue.item(i).data(gui.Qt.ItemDataRole.UserRole)).name
            for i in range(window.queue.count())] == ["ep1.mkv", "ep2.mp4"]
