"""Build the standalone app into dist/whisper-subs/.

    python scripts/build_app.py

Runs PyInstaller on packaging/whisper-subs.spec with its scratch files in the
system temp folder. PyInstaller's default puts them in build/, including
unfinished copies of both .exe files that don't run on their own, which is
easy to mistake for the real thing. Then writes a README.txt beside the
programs saying which is which, and a shortcut to the app in the repo root.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
APP = DIST / "whisper-subs"
SHORTCUT = ROOT / "Whisper Subtitler.lnk"

README = """Whisper Subtitler {version}
=========================

Japanese video or audio in, English subtitles out, all on this PC.

  Whisper Subtitler.exe   The app. Double-click it, drop files in, press Start.
  whisper-subs.exe        The same thing from a terminal, for scripts:
                            whisper-subs.exe video.mp4
                            whisper-subs.exe --help
  _internal\\              Libraries both programs share. Leave it where it is.

Before the first run
  - Install Ollama (https://ollama.com) and run:  ollama pull qwen3.5:9b
  - The first transcription downloads the Whisper model (about 3 GB).
  - An NVIDIA graphics card makes it much faster; without one it still works.
  - The window uses Microsoft Edge WebView2, which Windows 10 and 11 include.

Subtitles are saved next to each video: .en.srt (English), .ja.srt
(Japanese) and, if chosen, .ja-en.srt (both). Pause stops after the current
step; Resume carries on.

https://github.com/EmanChan050528/whisper-subs
"""


def make_shortcut(target: Path, link: Path) -> None:
    """A Windows shortcut to the app, for opening it from the repo folder.

    Not a copy of the .exe: PyInstaller's program only runs next to its
    _internal folder, and a one-file build would unpack ~1 GB to temp on every
    launch. A shortcut holds an absolute path, so it is gitignored.
    """
    ps = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK); "
        "$s.TargetPath = $env:TARGET; $s.WorkingDirectory = $env:WORKDIR; "
        "$s.IconLocation = $env:TARGET + ',0'; "
        "$s.Description = 'Whisper Subtitler: Japanese video or audio to English subtitles'; "
        "$s.Save()"
    )
    env = {**os.environ, "LNK": str(link), "TARGET": str(target),
           "WORKDIR": str(target.parent)}
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, env=env)


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from whisper_subs import __version__

    # Windows won't let PyInstaller replace the files of a program that is
    # running, and the failure it reports doesn't say so.
    running = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Whisper Subtitler.exe"],
                             capture_output=True, text=True).stdout
    if "Whisper Subtitler.exe" in running:
        print("Close Whisper Subtitler first: its files in dist/ can't be replaced while it runs.")
        return 1

    work = Path(tempfile.gettempdir()) / "whisper-subs-pyinstaller"
    cmd = [sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "whisper-subs.spec"),
           "--noconfirm", "--log-level", "WARN",
           "--workpath", str(work), "--distpath", str(DIST)]
    print("building...")
    if subprocess.run(cmd, cwd=ROOT).returncode:
        print("PyInstaller failed; its messages are above.")
        return 1
    shutil.rmtree(work, ignore_errors=True)

    (APP / "README.txt").write_text(README.format(version=__version__), encoding="utf-8")
    size = sum(f.stat().st_size for f in APP.rglob("*") if f.is_file())
    print(f"\n{APP}  ({size / 1e9:.2f} GB)")
    for exe in sorted(APP.glob("*.exe")):
        print(f"  {exe.name}")
    print("  README.txt")
    if sys.platform == "win32":
        make_shortcut(APP / "Whisper Subtitler.exe", SHORTCUT)
        print(f"\n{SHORTCUT}  (shortcut to the app)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
