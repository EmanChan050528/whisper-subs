import io
import json

import pytest

from whisper_subs import cli, ollama
from whisper_subs._js import js_len, js_round
from whisper_subs.ollama import BackendError, _host_url, check_model, parse_json
from whisper_subs.srt import units_to_srt, wrap


def test_js_semantics():
    assert js_len("𠮷野家") == 4 and js_len("あいう") == 3
    assert [js_round(x) for x in (2.5, -2.5, 0.49999999999999994, 1.4999)] == [3, -2, 0, 1]


@pytest.mark.parametrize("reply", [
    '{"1": "a"}',
    'Sure! Here it is:\n```json\n{"1": "a"}\n```\nHope that helps.',
    'Here you go: {"1": "a"} done',
])
def test_parse_json_recovers_wrapped_replies(reply):
    assert parse_json(reply) == {"1": "a"}


def test_parse_json_error_shows_the_tail():
    with pytest.raises(ValueError, match=r"….*cut off here"):
        parse_json('{"1": "a", "2": "this reply got cut off here')


@pytest.mark.parametrize("host, url", [
    ("http://localhost:11434", "http://localhost:11434"),
    ("127.0.0.1:11434", "http://127.0.0.1:11434"),
    ("0.0.0.0:11434", "http://127.0.0.1:11434"),
])
def test_host_url(host, url):
    assert _host_url(host) == url


def fake_tags(monkeypatch, names):
    body = json.dumps({"models": [{"name": n} for n in names]}).encode()
    monkeypatch.setattr(ollama.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))


def test_check_model(monkeypatch):
    fake_tags(monkeypatch, ["qwen3.5:9b", "gemma3:latest"])
    check_model("qwen3.5:9b")
    check_model("gemma3")  # untagged means :latest
    with pytest.raises(BackendError, match=r"ollama pull qwen3.5:4b"):
        check_model("qwen3.5:4b")


def test_wrap_caps_lines():
    text = " ".join(["word"] * 40)
    assert all(len(line) <= 42 for line in wrap(text).split("\n")[:2])
    assert len(wrap(text).split("\n")) == 3


def test_bilingual_srt_puts_japanese_first_and_skips_untranslated():
    units = [{"ja": "こんにちは", "start_ms": 0, "end_ms": 1000},
             {"ja": "未翻訳", "start_ms": 2000, "end_ms": 3000},
             {"ja": "はい", "start_ms": 4000, "end_ms": 4100}]
    srt = units_to_srt(units, ["Hello", "", "Yes"], japanese=True)
    assert srt == ("1\n00:00:00,000 --> 00:00:01,000\nこんにちは\nHello\n\n"
                   "2\n00:00:04,000 --> 00:00:04,700\nはい\nYes\n")


def test_cli_translates_a_ja_json_with_whisper_preset(tmp_path, monkeypatch, capsys):
    transcript = {"title": "t", "source": "whisper large-v3 (cuda/float16)", "cues": [
        {"t_ms": 0, "dur_ms": 900, "ja": "こんにちは"},
        {"t_ms": 1500, "dur_ms": 900, "ja": "元気ですか"},   # 600 ms gap: a new unit at 500
    ]}
    src = tmp_path / "clip.ja.json"
    src.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")

    def backend(prompt, json_mode=False):
        if prompt.startswith("You are preparing"):
            return json.dumps({"setting": "A greeting."})
        return json.dumps({"1": "Hello", "2": "How are you?"})

    monkeypatch.setattr(ollama, "check_model", lambda *a, **k: None)
    monkeypatch.setattr(ollama, "ollama_backend", lambda **k: backend)

    assert cli.main([str(src), "--bilingual"]) == 0
    en = (tmp_path / "clip.en.srt").read_text(encoding="utf-8")
    assert "Hello" in en and "How are you?" in en
    assert (tmp_path / "clip.ja-en.srt").read_text(encoding="utf-8").count("\n\n") == 1
    meta = json.loads((tmp_path / "clip.en.json").read_text(encoding="utf-8"))
    assert meta["gap_ms"] == 500 and [u["en"] for u in meta["units"]] == ["Hello", "How are you?"]


def test_cli_stops_before_transcribing_when_ollama_is_missing(tmp_path, monkeypatch, capsys):
    media = tmp_path / "clip.mp3"
    media.write_bytes(b"not really audio")

    def missing(*a, **k):
        raise BackendError("Cannot reach Ollama")

    monkeypatch.setattr(ollama, "check_model", missing)
    assert cli.main([str(media)]) == 1
    assert "Cannot reach Ollama" in capsys.readouterr().err
    assert not (tmp_path / "clip.whisper.json").exists()


def test_echo_matches():
    from whisper_subs.pipeline import echo_matches

    assert echo_matches("良くないことだと思っていました", "良くないことだと思っていました")
    assert echo_matches("こんにちは、藤森翔です。", "こんにちは藤森翔です")   # punctuation dropped
    assert not echo_matches("良くないことだと思っていました", "どうしてそう思ってましたか")
    assert not echo_matches("お金持ちが", None)


def _chunk_units(*ja):
    return [{"ja": j, "start_ms": i * 1000, "end_ms": i * 1000 + 900} for i, j in enumerate(ja)]


def test_echo_mode_rejects_shifted_lines_and_retries_them():
    from whisper_subs.pipeline import translate_units

    units = _chunk_units("昔は", "ニュースを見て", "お金持ちが")
    calls = []

    def backend(prompt, json_mode=False):
        calls.append(prompt)
        if len(calls) == 1:  # shifted: line 2 carries line 3's Japanese
            return json.dumps({"1": {"ja": "昔は", "en": "Back then"},
                               "2": {"ja": "お金持ちが", "en": "rich people"},
                               "3": {"ja": "お金持ちが", "en": "rich people"}},
                              ensure_ascii=False)
        return json.dumps({"2": {"ja": "ニュースを見て", "en": "watching the news"}},
                          ensure_ascii=False)

    log = []
    out = translate_units(units, {}, backend, {"echo": True}, log.append)
    assert out["translations"] == ["Back then", "watching the news", "rich people"]
    assert "rejected 1 line(s)" in log[0] and len(calls) == 2
    assert '"ja": "昔は"' in calls[0]  # the example shows the copy format


def test_echo_mode_accepts_bare_strings_only_as_a_last_resort():
    from whisper_subs.pipeline import translate_units

    units = _chunk_units("はい")
    calls = []

    def backend(prompt, json_mode=False):
        calls.append(prompt)
        return json.dumps({"1": "Yes"})

    out = translate_units(units, {}, backend, {"echo": True})
    assert out["translations"] == ["Yes"] and len(calls) == 3
