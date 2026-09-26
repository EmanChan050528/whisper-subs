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

import re

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct,
    VSVersionInfo,
)

here = Path(SPECPATH)
site = here.parent / ".venv" / "Lib" / "site-packages"
icon = str(here / "icon.ico")  # scripts/make_icon.py
version = re.search(r'__version__ = "([^"]+)"',
                    (here.parent / "src" / "whisper_subs" / "__init__.py").read_text()).group(1)


def version_info(description, filename):
    """What Explorer shows under Properties > Details."""
    nums = tuple(int(n) for n in (version.split(".") + ["0"] * 4)[:4])
    return VSVersionInfo(
        ffi=FixedFileInfo(filevers=nums, prodvers=nums),
        kids=[
            StringFileInfo([StringTable("040904B0", [
                StringStruct("ProductName", "Whisper Subtitler"),
                StringStruct("FileDescription", description),
                StringStruct("FileVersion", version),
                StringStruct("ProductVersion", version),
                StringStruct("OriginalFilename", filename),
                StringStruct("InternalName", filename.removesuffix(".exe")),
                StringStruct("LegalCopyright", "github.com/EmanChan050528/whisper-subs"),
            ])]),
            VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
        ],
    )

# cuBLAS only. A transcription never loads cuDNN or NVRTC (docs/benchmarks.md),
# which saves 1.3 GB. Kept at nvidia/cublas/bin so gpu.enable_cuda_dlls finds it.
cublas = [(str(dll), "nvidia/cublas/bin") for dll in (site / "nvidia" / "cublas" / "bin").glob("*.dll")]

binaries = cublas + collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("av")
datas = collect_data_files("faster_whisper")  # the Silero VAD model
datas += [(str(here.parent / "src" / "whisper_subs" / "assets" / "icon.png"),
           "whisper_subs/assets")]  # the window icon

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
    exclude_binaries=True, name="Whisper Subtitler", console=False, icon=icon,
    version=version_info("Whisper Subtitler", "Whisper Subtitler.exe"),
)
cli_exe = EXE(
    PYZ(cli.pure), cli.scripts, [],
    exclude_binaries=True, name="whisper-subs", console=True, icon=icon,
    version=version_info("Whisper Subtitler (command line)", "whisper-subs.exe"),
)

COLLECT(
    gui_exe, gui.binaries, gui.datas,
    cli_exe, cli.binaries, cli.datas,
    name="whisper-subs",
)
