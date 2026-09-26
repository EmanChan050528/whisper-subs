# PyInstaller build: one folder, two programs sharing one set of libraries.
#
#   .venv/Scripts/pyinstaller packaging/whisper-subs.spec --noconfirm
#
# -> dist/whisper-subs/Whisper Subtitler.exe   the window
#    dist/whisper-subs/whisper-subs.exe        the command line
#
# Whisper models are not bundled: faster-whisper downloads them to the
# Hugging Face cache on first use (large-v3 is 2.9 GB).

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

site = Path(SPECPATH).parent / ".venv" / "Lib" / "site-packages"

# cuBLAS only. A transcription never loads cuDNN or NVRTC (docs/benchmarks.md),
# which saves 1.3 GB. Kept at nvidia/cublas/bin so gpu.enable_cuda_dlls finds it.
cublas = [(str(dll), "nvidia/cublas/bin") for dll in (site / "nvidia" / "cublas" / "bin").glob("*.dll")]

binaries = cublas + collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("av")
datas = collect_data_files("faster_whisper")  # the Silero VAD model

# Qt modules the window does not use; PySide6-Essentials still ships them.
excludes = [
    "PySide6.QtNetwork", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtSql",
    "PySide6.QtTest", "PySide6.QtXml", "PySide6.QtOpenGL", "PySide6.QtPdf",
    "PySide6.QtConcurrent", "PySide6.QtDBus", "PySide6.QtHelp", "PySide6.QtDesigner",
    "tkinter", "matplotlib", "pytest", "IPython",
]

common = dict(
    binaries=binaries,
    datas=datas,
    excludes=excludes,
    pathex=[str(Path(SPECPATH).parent / "src")],
)

gui = Analysis([str(Path(SPECPATH) / "launch_gui.py")], **common)
cli = Analysis([str(Path(SPECPATH) / "launch_cli.py")], **common)

gui_exe = EXE(
    PYZ(gui.pure), gui.scripts, [],
    exclude_binaries=True, name="Whisper Subtitler", console=False,
)
cli_exe = EXE(
    PYZ(cli.pure), cli.scripts, [],
    exclude_binaries=True, name="whisper-subs", console=True,
)

COLLECT(
    gui_exe, gui.binaries, gui.datas,
    cli_exe, cli.binaries, cli.datas,
    name="whisper-subs",
)
