"""Record docs/demo.gif and docs/screenshot.png from a real run of the window.

    python scripts/make_demo.py samples/minute-sample.mp3

Opens the real window on the real pipeline (Ollama must be running) and grabs
it from the screen every few hundred milliseconds, so leave it uncovered until
it closes. The file is copied to a temp folder first, so the run starts from
scratch and nothing is written beside the original.
"""

import ctypes
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Keep the demo's settings and glossary away from the real ones.
os.environ["WHISPER_SUBS_HOME"] = str(Path(tempfile.gettempdir()) / "ws-demo")

import webview
from PIL import Image, ImageGrab

from whisper_subs import gui

DOCS = Path(__file__).resolve().parents[1] / "docs"
FRAME_S = 0.4
WIDTH = 720  # GIF width; the window is grabbed larger and scaled down


class Rect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


def own_window(title: str) -> int:
    """This process's top-level window with the given title. (WinForms won't
    hand over its handle to another thread.)"""
    user32, found = ctypes.windll.user32, []
    pid = ctypes.c_ulong()

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(hwnd, _):
        user32.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(pid))
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(ctypes.c_void_p(hwnd), buf, 256)
        if pid.value == os.getpid() and buf.value == title:
            found.append(hwnd)
        return True

    user32.EnumWindows(visit, None)
    return found[0]


def record(window, app: gui.App, media: Path, frames: list) -> None:
    time.sleep(3)  # the page draws itself
    hwnd = own_window("Whisper Subtitler")
    rect = Rect()

    def grab(hold: float = FRAME_S) -> Image.Image:
        ctypes.windll.user32.SetForegroundWindow(ctypes.c_void_p(hwnd))
        # The frame without its invisible resize border (DWMWA_EXTENDED_FRAME_BOUNDS).
        ctypes.windll.dwmapi.DwmGetWindowAttribute(ctypes.c_void_p(hwnd), 9, ctypes.byref(rect),
                                                   ctypes.sizeof(rect))
        im = ImageGrab.grab((rect.left, rect.top, rect.right, rect.bottom), all_screens=True)
        frames.append((im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS),
                       hold))
        return im

    try:
        grab(1.2)                                       # empty window
        window.evaluate_js("document.body.classList.add('dragging')")
        time.sleep(0.3)
        grab(0.9)                                       # a file dragged over it
        window.evaluate_js("document.body.classList.remove('dragging')")
        app.add_paths([str(media)])
        window.evaluate_js("document.querySelector('#bilingual').checked = true")
        time.sleep(0.5)
        grab(0.8)
        window.evaluate_js("pywebview.api.start(prefs())")
        time.sleep(0.5)
        while app.running:
            start = time.time()
            grab()
            time.sleep(max(0.0, FRAME_S - (time.time() - start)))
        time.sleep(0.5)
        window.evaluate_js("document.querySelector('#logbox').open = true")
        time.sleep(0.5)
        grab(4.0).save(DOCS / "screenshot.png")
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        window.destroy()


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="ws-demo-"))
    media = work / Path(sys.argv[1]).name
    shutil.copy2(sys.argv[1], media)
    frames: list[tuple[Image.Image, float]] = []

    app = gui.App()
    window = webview.create_window("Whisper Subtitler", url=gui.PAGE.as_uri(), js_api=app,
                                   width=1040, height=720, min_size=(760, 540),
                                   background_color="#141417" if gui._windows_dark() else "#f5f5f7")
    app._window = window
    app._emit = gui._page_emitter(window)
    webview.start(record, (window, app, media, frames), icon=str(gui.ICON))
    shutil.rmtree(work, ignore_errors=True)

    # Drop near-identical consecutive frames (long quiet stretches of a run),
    # keeping their display time.
    kept: list[tuple[Image.Image, float]] = []
    for im, hold in frames:
        if kept and im.tobytes() == kept[-1][0].tobytes():
            kept[-1] = (kept[-1][0], kept[-1][1] + hold)
        else:
            kept.append((im, hold))
    palette = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im, _ in kept]
    palette[0].save(DOCS / "demo.gif", save_all=True, append_images=palette[1:],
                    duration=[round(1000 * min(h, 4.0)) for _, h in kept], loop=0,
                    optimize=True)
    size = (DOCS / "demo.gif").stat().st_size / 1e6
    print(f"{len(kept)} frames, {size:.1f} MB -> {DOCS / 'demo.gif'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
