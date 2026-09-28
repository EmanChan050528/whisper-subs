"""The window: `whisper-subs-gui`.

Drop files in, press Start, get subtitles beside them. The window is a web
page (assets/ui/index.html) shown by pywebview in Windows' own WebView2, so it
looks and moves like a current app rather than a classic dialog. This module
is the Python side: `App` holds the queue and runs job.run_job on a worker
thread; the page calls App's public methods and redraws from the state App
sends back. Stop is a pause: each stage checkpoints, so Start carries on.
"""

import configparser
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from pathlib import Path

from whisper_subs import __version__, glossary, ollama
from whisper_subs.job import JobError, Settings, Stopped, run_job
from whisper_subs.transcribe import DEFAULT_MODEL, FAST_MODEL

ASSETS = Path(__file__).parent / "assets"
PAGE = ASSETS / "ui" / "index.html"
ICON = ASSETS / "icon.ico"
VIDEO = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".ts", ".m4v")
AUDIO = (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus")
WAITING, RUNNING, DONE, PAUSED, FAILED = "Waiting", "Running", "Done", "Paused", "Error"
PREFS = {"whisper_model": DEFAULT_MODEL, "llm_model": ollama.DEFAULT_MODEL, "glossary": "",
         "bilingual": False, "ja_only": False, "check": False}

Emit = Callable[[str, object], None]


def _is_media(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".ja.json")
                               or path.suffix.lower() in VIDEO + AUDIO)


def _kind(path: Path) -> str:
    if path.name.endswith(".ja.json"):
        return "transcript"
    return "audio" if path.suffix.lower() in AUDIO else "video"


class App:
    """Everything behind the window, without the window.

    The page reaches these methods as window.pywebview.api.<name>; pywebview
    runs each call on its own thread, hence the lock. Anything starting with
    an underscore stays hidden from the page. Changes go out through emit:
    ("state", snapshot) after every change, ("log", line) for the log.
    """

    def __init__(self, emit: Emit | None = None):
        self._emit = emit or (lambda event, data: None)
        self._lock = threading.RLock()
        self._files: list[dict] = []
        self._log: deque[str] = deque(maxlen=3000)  # replayed if the page reloads
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = "Add some files to begin."
        self._last_push = 0.0
        self._prefs_file = glossary.home() / "gui.json"
        self._window = None  # set by main(); None in tests

    # ------------------------------------------------------------ page API

    def init(self) -> dict:
        """Everything the page needs to draw itself the first time."""
        models = ollama.list_models()
        return {"version": __version__, "prefs": self.load_prefs(),
                "whisper": [{"value": DEFAULT_MODEL, "label": "Accurate", "hint": DEFAULT_MODEL},
                            {"value": FAST_MODEL, "label": "Fast", "hint": f"{FAST_MODEL}, ~4x"}],
                "models": models or [], "glossaries": self._glossaries(),
                "health": _health(models), "state": self._snapshot(), "log": list(self._log)}

    def load_prefs(self) -> dict:
        prefs = dict(PREFS)
        try:
            prefs.update(json.loads(self._prefs_file.read_text(encoding="utf-8")))
        except FileNotFoundError:
            prefs.update(_old_prefs(self._prefs_file.with_name("gui.ini")))
        except (OSError, ValueError):
            pass
        return {k: prefs[k] for k in PREFS}

    def save_prefs(self, prefs: dict) -> None:
        clean = {k: type(v)(prefs.get(k, v)) for k, v in PREFS.items()}
        self._prefs_file.parent.mkdir(parents=True, exist_ok=True)
        self._prefs_file.write_text(json.dumps(clean, indent=2), encoding="utf-8")

    def add_paths(self, paths: list[str]) -> int:
        files = []
        for p in map(Path, paths):  # a folder contributes the media files inside it
            if p.is_dir():
                found = sorted(f for f in p.iterdir() if _is_media(f))
                # Not a transcript whose video is also there: it's the same job.
                stems = {f.stem for f in found if not f.name.endswith(".ja.json")}
                files += [f for f in found if f.name.removesuffix(".ja.json") not in stems]
            elif p.is_file():
                files.append(p)
        with self._lock:
            known = {f["path"] for f in self._files}
            added = [f for f in dict.fromkeys(str(p) for p in files) if f not in known]
            self._files += [_row(Path(p)) for p in added]
            if added and not self.running:
                self._status = f"{self._count(WAITING)} file(s) ready. Press Start."
        self._push()
        return len(added)

    def browse(self) -> None:
        """The file picker, for people who don't drag."""
        import webview
        exts = ";".join(f"*{e}" for e in VIDEO + AUDIO)
        picked = self._window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=True,
            file_types=(f"Video or audio ({exts})", "Transcripts (*.ja.json)", "All files (*.*)"))
        if picked:
            self.add_paths(list(picked))

    def remove(self, path: str) -> None:
        with self._lock:
            self._files = [f for f in self._files if f["path"] != path or f["status"] == RUNNING]
        self._push()

    def clear_finished(self) -> None:
        with self._lock:
            self._files = [f for f in self._files if f["status"] != DONE]
        self._push()

    def start(self, prefs: dict | None = None) -> None:
        with self._lock:
            if self.running:
                return
            if prefs is not None:
                self.save_prefs(prefs)
            jobs = [f for f in self._files if f["status"] != DONE]
            if not jobs:
                self._status = "Nothing to do. Drop some files in first."
                self._push()
                return
            self._stop.clear()
            self._status = "Starting…"
            self._thread = threading.Thread(target=self._run, args=(jobs, self._settings()),
                                            daemon=True, name="whisper-subs job")
            self._thread.start()
        self._push()

    def stop(self) -> None:
        if self.running:
            self._stop.set()
            self._status = "Stopping after the current step…"
            self._push()

    def reveal(self, path: str) -> None:
        """Open the file's folder with the file selected."""
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(Path(path))])
        else:
            _open(Path(path).parent)

    def open_glossaries(self) -> None:
        folder = glossary.home() / "glossaries"
        folder.mkdir(parents=True, exist_ok=True)
        _open(folder)

    def refresh_health(self) -> dict:
        models = ollama.list_models()
        return {"health": _health(models), "models": models or [],
                "glossaries": self._glossaries()}

    # ------------------------------------------------------------ internals

    @property
    def running(self) -> bool:
        return self._thread is not None

    def _settings(self) -> Settings:
        p = self.load_prefs()
        return Settings(model=p["whisper_model"], glossary=p["glossary"].strip() or None,
                        llm_model=p["llm_model"].strip() or ollama.DEFAULT_MODEL,
                        bilingual=p["bilingual"], ja_only=p["ja_only"], check=p["check"])

    def _run(self, jobs: list[dict], settings: Settings) -> None:
        for row in jobs:
            if self._stop.is_set():
                break
            self._set(row, RUNNING, "", stage="transcribe", percent=None)
            self._status = f"Working on {row['name']}"
            self._push()
            self._say(f"— {row['name']}")
            try:
                result = run_job(Path(row["path"]), settings, log=self._say,
                                 progress=lambda s, d, t, row=row: self._progress(row, s, d, t),
                                 should_stop=self._stop.is_set)
            except Stopped:
                self._set(row, PAUSED, "Press Start to carry on")
                break
            except JobError as err:
                self._say(f"error: {err}")
                self._set(row, FAILED, str(err))
                continue
            except Exception as err:  # keep the window alive; show what happened
                self._say(traceback.format_exc())
                self._set(row, FAILED, f"{type(err).__name__}: {err}")
                continue
            detail = (f"{result.missing} line(s) untranslated. Start again to retry them."
                      if result.missing else "")
            row["outputs"] = [str(p) for p in result.outputs]
            for o in row["outputs"]:
                self._say(f"  wrote {o}")
            self._set(row, DONE, detail)
        with self._lock:
            self._thread = None
            left = self._count(WAITING) + self._count(PAUSED)
            failed = self._count(FAILED)
            if self._stop.is_set():
                self._status = "Paused. Press Start to carry on where it stopped."
            elif failed:
                self._status = f"Finished, with {failed} error(s). See the list and the log."
            else:
                self._status = "All done." if not left else f"{left} file(s) still waiting."
        self._push()

    def _progress(self, row: dict, stage: str, done: float, total: float) -> None:
        pct = 100 * done / total if total else None
        if stage == "transcribe":
            text = f"Transcribing {pct:.0f}%" if pct is not None else "Transcribing"
        else:
            text = f"Translating {int(done)} / {int(total)}"
        changed = row.get("stage") != stage
        self._set(row, RUNNING, text, stage=stage, percent=pct, push=False)
        self._status = f"{row['name']}: {text[0].lower()}{text[1:]}"
        self._push(force=changed)

    def _set(self, row: dict, status: str, detail: str, push: bool = True, **extra) -> None:
        with self._lock:
            row.update(status=status, detail=detail, **extra)
            if status != RUNNING:
                row.update(stage=None, percent=None)
        if push:
            self._push()

    def _say(self, line: str) -> None:
        self._log.append(line)
        self._emit("log", line)

    def _count(self, status: str) -> int:
        return sum(f["status"] == status for f in self._files)

    def _snapshot(self) -> dict:
        with self._lock:
            files = [dict(f) for f in self._files]
            done = self._count(DONE)
        return {"files": files, "running": self.running, "stopping": self._stop.is_set(),
                "status": self._status, "done": done, "total": len(files)}

    def _push(self, force: bool = True) -> None:
        # Progress arrives many times a second; the page needs ten at most.
        now = time.monotonic()
        if not force and now - self._last_push < 0.1:
            return
        self._last_push = now
        self._emit("state", self._snapshot())

    @staticmethod
    def _glossaries() -> list[str]:
        gdir = glossary.home() / "glossaries"
        return [g.stem for g in sorted(gdir.glob("*.json"))] if gdir.is_dir() else []

    def _shutdown(self) -> None:
        """Closing the window: pause the job (it checkpoints) and let it go."""
        thread = self._thread
        if thread:
            self._stop.set()
            thread.join(3)


def _row(path: Path) -> dict:
    return {"path": str(path), "name": path.name, "kind": _kind(path), "status": WAITING,
            "detail": "", "stage": None, "percent": None, "outputs": []}


def _old_prefs(ini: Path) -> dict:
    """Settings saved by the first (Qt) window, so an upgrade keeps them."""
    cp = configparser.ConfigParser()
    try:
        cp.read(ini, encoding="utf-8")
    except (OSError, configparser.Error):
        return {}
    old = dict(cp["General"]) if cp.has_section("General") else {}
    out = {}
    for k, default in PREFS.items():
        if k in old:
            out[k] = old[k].lower() == "true" if isinstance(default, bool) else old[k]
    return out


def _health(models: list[str] | None) -> list[dict]:
    """What will and won't work, before anyone presses Start."""
    try:
        from whisper_subs.gpu import enable_cuda_dlls
        enable_cuda_dlls()
        import ctranslate2
        gpu = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        gpu = False
    out = [{"name": "GPU", "ok": gpu,
            "text": "GPU ready" if gpu else "No GPU",
            "hint": "NVIDIA CUDA" if gpu else "Transcription will run on the CPU and be "
                                              "slow. The Fast model helps."}]
    if models is None:
        out.append({"name": "Ollama", "ok": False, "text": "Ollama not running",
                    "hint": "Start the Ollama app to translate, or turn on Japanese only."})
    elif not models:
        out.append({"name": "Ollama", "ok": False, "text": "No Ollama models",
                    "hint": f"Run: ollama pull {ollama.DEFAULT_MODEL}"})
    else:
        out.append({"name": "Ollama", "ok": True, "text": f"Ollama · {len(models)} models",
                    "hint": ", ".join(models)})
    return out


def _open(folder: Path) -> None:
    if sys.platform == "win32":
        os.startfile(folder)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(folder)])


# ------------------------------------------------------------------ window

def _windows_dark() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def _page_emitter(window) -> Emit:
    def emit(event: str, data: object) -> None:
        js = f"window.ui && window.ui.on({json.dumps(event)}, {json.dumps(data)})"
        try:
            window.run_js(js)
        except Exception:
            pass  # the page is loading or the window is closing
    return emit


def main() -> int:
    import webview
    from webview.dom import DOMEventHandler

    if sys.platform == "win32":
        # Its own taskbar identity, so Windows groups and labels it as this app.
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("whisper-subs.gui")
    dark = _windows_dark()  # paints the window before the page arrives
    app = App()
    window = webview.create_window(
        "Whisper Subtitler", url=PAGE.as_uri(), js_api=app, width=1040, height=720,
        min_size=(760, 540), background_color="#141417" if dark else "#f5f5f7")
    app._window = window
    app._emit = _page_emitter(window)

    def on_drop(event: dict) -> None:
        files = (event.get("dataTransfer") or {}).get("files") or []
        app.add_paths([f["pywebviewFullPath"] for f in files if f.get("pywebviewFullPath")])

    bound = []

    def on_loaded() -> None:
        # Dropped files only carry their full path through pywebview's own DOM
        # events. "loaded" fires again on a reload, so bind once.
        if not bound:
            window.dom.document.events.drop += DOMEventHandler(on_drop, True, True)
            bound.append(True)

    window.events.loaded += on_loaded
    window.events.closing += app._shutdown
    # pywebview sets the title bar to match Windows' theme and follows changes.
    webview.start(debug=bool(os.environ.get("WHISPER_SUBS_DEBUG")), icon=str(ICON))
    return 0


if __name__ == "__main__":
    sys.exit(main())
