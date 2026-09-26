"""Make CUDA libraries findable before faster-whisper loads.

CTranslate2's Windows wheel ships without cuBLAS and cuDNN. The usual advice is
to install the CUDA toolkit system-wide; instead we take them from the
`nvidia-cublas-cu12` / `nvidia-cudnn-cu12` pip wheels inside the venv, which
keeps the project self-contained and lets a PyInstaller build carry them.

Call `enable_cuda_dlls()` before importing faster_whisper.
"""

import os
import sys
from pathlib import Path


def cuda_dll_dirs() -> list[Path]:
    """`bin` directories of any nvidia-* wheels installed in this environment."""
    dirs = []
    for entry in sys.path:
        nvidia = Path(entry) / "nvidia"
        if nvidia.is_dir():
            dirs.extend(p for p in nvidia.glob("*/bin") if p.is_dir())
    return dirs


def enable_cuda_dlls() -> list[Path]:
    if sys.platform != "win32":
        return []
    dirs = cuda_dll_dirs()
    for d in dirs:
        os.add_dll_directory(str(d))
    # CTranslate2 loads cuDNN lazily through LoadLibrary, which honours PATH
    # rather than add_dll_directory, so set both.
    if dirs:
        os.environ["PATH"] = os.pathsep.join([*map(str, dirs), os.environ.get("PATH", "")])
    return dirs
