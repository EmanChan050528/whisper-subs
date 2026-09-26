"""Record docs/demo.gif and docs/screenshot.png from a real run of the window.

    python scripts/make_demo.py samples/minute-sample.mp3

Drives the real window and the real pipeline (Ollama must be running),
rendering offscreen and grabbing a frame every few hundred milliseconds.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
# Keep the demo's settings and glossary away from the real ones.
os.environ["WHISPER_SUBS_HOME"] = str(Path(os.environ.get("TEMP", "/tmp")) / "ws-demo")

from PIL import Image
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from whisper_subs import gui

DOCS = Path(__file__).resolve().parents[1] / "docs"
FRAME_S = 0.4
WIDTH = 640  # GIF width; the window is rendered larger and scaled down


def main() -> int:
    media = Path(sys.argv[1])
    app = QApplication([])
    app.setFont(QFont("Segoe UI", 9))  # what Windows itself uses
    w = gui.MainWindow()
    w.resize(760, 700)
    w.show()
    frames: list[tuple[Image.Image, float]] = []

    def grab(hold: float = FRAME_S) -> None:
        app.processEvents()
        img = w.grab().toImage()
        path = DOCS / "_frame.png"
        img.save(str(path))
        im = Image.open(path).convert("RGB")
        frames.append((im.resize((WIDTH, round(im.height * WIDTH / im.width)),
                                 Image.LANCZOS), hold))

    grab(1.2)                          # empty window
    w.drop._set_hover(True)            # a file being dragged over
    grab(0.8)
    w.drop._set_hover(False)
    w.add_files([media])
    w.bilingual.setChecked(True)
    grab(0.8)
    w.start()
    while w.running:
        grab()
        end = time.time() + FRAME_S
        while time.time() < end:
            app.processEvents()
            time.sleep(0.02)
    grab(1.0)
    subs = media.with_suffix(".ja-en.srt").read_text(encoding="utf-8").strip().split("\n\n")
    for block in subs[:4]:             # show what came out
        w.append_log(block.replace("\n", "  |  ", 1).split("\n", 1)[-1].replace("\n", " / "))
    grab(4.0)
    w.grab().save(str(DOCS / "screenshot.png"))
    (DOCS / "_frame.png").unlink()

    # Drop near-identical consecutive frames (long quiet stretches of a run),
    # keeping their display time.
    kept: list[tuple[Image.Image, float]] = []
    for im, hold in frames:
        if kept and im.tobytes() == kept[-1][0].tobytes():
            kept[-1] = (kept[-1][0], kept[-1][1] + hold)
        else:
            kept.append((im, hold))
    palette = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im, _ in kept]
    palette[0].save(DOCS / "demo.gif", save_all=True, append_images=palette[1:],
                    duration=[round(1000 * min(h, 4.0)) for _, h in kept], loop=0,
                    optimize=True)
    size = (DOCS / "demo.gif").stat().st_size / 1e6
    print(f"{len(kept)} frames, {size:.1f} MB -> {DOCS / 'demo.gif'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
