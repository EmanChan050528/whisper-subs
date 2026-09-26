import json

import pytest

from whisper_subs import cli, glossary, ollama


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_SUBS_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_existing_entries_win_so_hand_edits_stick():
    glossary.remember("genshin", {"names": {"コロンビーナ": "Colombine"}})
    # A person corrects the file...
    p = glossary.path("genshin")
    data = json.loads(p.read_text(encoding="utf-8"))
    data["names"]["コロンビーナ"] = "Columbina"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    # ...and the next run's different rendering does not undo it.
    _, added = glossary.remember("genshin", {"names": {"コロンビーナ": "Colonnella",
                                                       "サンドローネ": "Sandrone"}})
    stored = glossary.load("genshin")
    assert stored["names"] == {"コロンビーナ": "Columbina", "サンドローネ": "Sandrone"}
    assert added == 1 and stored["files"] == 2


def test_merge_ignores_junk_and_caps_entries():
    glossary.remember("s", {"names": {"": "x", "a": 5, "b": "B"}, "terms": ["not", "a", "dict"]})
    assert glossary.load("s")["names"] == {"b": "B"}
    glossary.remember("big", {"terms": {f"k{i}": "v" for i in range(100)}})
    assert len(glossary.load("big")["terms"]) == glossary.MAX_ENTRIES


def test_seed_is_none_for_an_empty_glossary():
    assert glossary.seed(glossary.load("new")) is None


@pytest.mark.parametrize("name", ["../evil", "a/b", "", ".hidden"])
def test_bad_names_are_rejected(name):
    with pytest.raises(ValueError):
        glossary.path(name)


def test_second_file_is_seeded_with_the_first_files_names(tmp_path, monkeypatch):
    prompts = []

    def backend(prompt, json_mode=False):
        prompts.append(prompt)
        if prompt.startswith("You are preparing"):
            return json.dumps({"setting": "A stream.", "names": {"コロンビーナ": "Columbina"}},
                              ensure_ascii=False)
        return json.dumps({"1": {"ja": "コロンビーナ", "en": "Columbina."}}, ensure_ascii=False)

    monkeypatch.setattr(ollama, "check_model", lambda *a, **k: None)
    monkeypatch.setattr(ollama, "ollama_backend", lambda **k: backend)

    transcript = {"title": "t", "source": "whisper large-v3",
                  "cues": [{"t_ms": 0, "dur_ms": 900, "ja": "コロンビーナ"}]}
    for n in (1, 2):
        src = tmp_path / f"ep{n}.ja.json"
        src.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
        assert cli.main([str(src), "--glossary", "genshin"]) == 0

    analysis = [p for p in prompts if p.startswith("You are preparing")]
    assert "ALREADY ESTABLISHED" not in analysis[0]
    assert "ALREADY ESTABLISHED" in analysis[1] and '"コロンビーナ": "Columbina"' in analysis[1]
    assert glossary.load("genshin")["files"] == 2
