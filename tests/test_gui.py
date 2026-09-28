"""The window's behaviour behind the page, with the pipeline replaced by a fake job."""

import json
import time
from pathlib import Path

import pytest

from whisper_subs import gui, ollama
from whisper_subs.job import JobError, Result, Stopped


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_SUBS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(ollama, "list_models", lambda *a, **k: ["qwen3.5:9b"])
    events = []
    a = gui.App(emit=lambda event, data: events.append((event, data)))
    a.events = events
    yield a
    a._shutdown()


def media(tmp_path, *names):
    paths = []
    for n in names:
        (tmp_path / n).write_bytes(b"")
        paths.append(str(tmp_path / n))
    return paths


def wait_until_idle(app, timeout=10):
    end = time.time() + timeout
    while app.running and time.time() < end:
        time.sleep(0.01)
    assert not app.running


def statuses(app):
    return [f["status"] for f in app._files]


def test_queue_runs_every_file_and_reports_errors_per_file(app, tmp_path, monkeypatch):
    files = media(tmp_path, "a.mp3", "bad.mp3", "c.mp3")
    seen = []

    def fake_job(path, settings, log, progress, should_stop):
        seen.append((path.name, settings.llm_model, settings.bilingual))
        progress("transcribe", 30, 60)
        progress("translate", 1, 2)
        if path.name == "bad.mp3":
            raise JobError("bad.mp3 has no audio stream.")
        return Result(outputs=[path.with_suffix(".en.srt")], units=2, translated=2)

    monkeypatch.setattr(gui, "run_job", fake_job)
    assert app.add_paths(files) == 3
    assert app.add_paths(files[:1]) == 0  # duplicates are ignored
    app.start({**app.load_prefs(), "bilingual": True})
    wait_until_idle(app)

    assert statuses(app) == [gui.DONE, gui.FAILED, gui.DONE]
    assert seen == [("a.mp3", "qwen3.5:9b", True), ("bad.mp3", "qwen3.5:9b", True),
                    ("c.mp3", "qwen3.5:9b", True)]
    assert "no audio stream" in app._files[1]["detail"]
    assert app._files[0]["outputs"] == [str(tmp_path / "a.en.srt")]
    last = [d for e, d in app.events if e == "state"][-1]
    assert not last["running"] and "error" in last["status"] and last["done"] == 2
    assert any(e == "log" and "wrote" in d for e, d in app.events)

    app.clear_finished()
    assert statuses(app) == [gui.FAILED]


def test_stop_pauses_and_start_carries_on(app, tmp_path, monkeypatch):
    calls = []

    def fake_job(path, settings, log, progress, should_stop):
        calls.append(path.name)
        if len(calls) == 1:
            app.stop()  # the user presses Pause mid-file
            assert should_stop()
            raise Stopped
        return Result(outputs=[path.with_suffix(".en.srt")], units=1, translated=1)

    monkeypatch.setattr(gui, "run_job", fake_job)
    app.add_paths(media(tmp_path, "long.mp3", "next.mp3"))
    app.start()
    wait_until_idle(app)
    assert statuses(app) == [gui.PAUSED, gui.WAITING]  # the second never started
    assert "Paused" in app._status

    app.start()
    wait_until_idle(app)
    assert calls == ["long.mp3", "long.mp3", "next.mp3"]
    assert statuses(app) == [gui.DONE, gui.DONE]


def test_prefs_round_trip_and_reach_the_job(app):
    app.save_prefs({**app.load_prefs(), "ja_only": True, "glossary": " my-show "})
    assert app.load_prefs()["ja_only"] is True
    s = app._settings()
    assert s.ja_only and s.glossary == "my-show"


def test_settings_from_the_old_qt_window_are_kept(app):
    home = gui.glossary.home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "gui.ini").write_text("[General]\nllm_model=gemma3:12b\nbilingual=true\n",
                                  encoding="utf-8")
    prefs = app.load_prefs()
    assert prefs["llm_model"] == "gemma3:12b" and prefs["bilingual"] is True
    assert prefs["check"] is False


def test_dropping_a_folder_adds_its_media(app, tmp_path):
    folder = tmp_path / "season"
    folder.mkdir()
    media(folder, "ep1.mkv", "ep1.ja.json", "ep2.mp4", "ep3.ja.json", "notes.txt")
    assert gui._is_media(folder / "ep1.mkv") and not gui._is_media(folder / "notes.txt")
    app.add_paths([str(folder)])
    # ep1's transcript is ep1's job already; ep3 has only its transcript.
    assert [f["name"] for f in app._files] == ["ep1.mkv", "ep2.mp4", "ep3.ja.json"]
    assert [f["kind"] for f in app._files] == ["video", "video", "transcript"]


def test_init_gives_the_page_everything_and_is_json(app, tmp_path):
    app.add_paths(media(tmp_path, "a.mp3"))
    d = app.init()
    json.dumps(d)  # it crosses into JavaScript
    assert d["models"] == ["qwen3.5:9b"] and d["state"]["files"][0]["kind"] == "audio"
    assert {h["name"] for h in d["health"]} == {"GPU", "Ollama"}


def test_the_page_can_only_see_the_public_methods():
    exposed = {n for n in dir(gui.App) if not n.startswith("_")}
    assert exposed == {"init", "load_prefs", "save_prefs", "add_paths", "browse", "remove",
                       "clear_finished", "start", "stop", "reveal", "open_glossaries",
                       "refresh_health", "running"}


def test_the_page_exists():
    page = gui.PAGE.read_text(encoding="utf-8")
    for name in ("init", "save_prefs", "start", "stop", "browse", "remove", "clear_finished",
                 "reveal", "open_glossaries", "refresh_health"):
        assert f"api.{name}(" in page, name
    assert Path(gui.ICON).exists()
