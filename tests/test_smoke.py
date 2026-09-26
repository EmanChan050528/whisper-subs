from whisper_subs import __version__
from whisper_subs.cli import main
from whisper_subs.gpu import enable_cuda_dlls


def test_version_flag(capsys):
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_enable_cuda_dlls_is_safe_to_call_twice():
    enable_cuda_dlls()
    enable_cuda_dlls()
