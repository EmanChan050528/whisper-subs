import pytest

from whisper_subs import __version__
from whisper_subs.cli import main
from whisper_subs.gpu import enable_cuda_dlls


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_enable_cuda_dlls_is_safe_to_call_twice():
    enable_cuda_dlls()
    enable_cuda_dlls()


def test_cache_is_tied_to_the_source_file(tmp_path):
    import json

    from whisper_subs.cli import load_cached, source_id
    from whisper_subs.transcribe import Options

    mp4, mkv = tmp_path / "clip.mp4", tmp_path / "clip.mkv"
    mp4.write_bytes(b"a")
    mkv.write_bytes(b"bb")
    cache = tmp_path / "clip.whisper.json"
    from dataclasses import asdict
    cache.write_text(json.dumps({"source": source_id(mp4), "options": asdict(Options())}))

    assert load_cached(cache, Options(), mp4) is not None
    assert load_cached(cache, Options(), mkv) is None           # same stem, other file
    assert load_cached(cache, Options(model="small"), mp4) is None


def test_missing_input_is_a_clean_error(tmp_path, capsys):
    assert main([str(tmp_path / "nope.mp4")]) == 1
    assert "no such file" in capsys.readouterr().err
