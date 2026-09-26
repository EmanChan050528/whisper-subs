"""The Python port must reproduce jp-subs' JS exactly.

Goldens in tests/parity/golden/ were produced by jp-subs' own segment(),
chunk(), run() and toSrt() (see tests/parity/dump.mjs). Here the Python port
runs on the same fixtures with the same fake model, and every unit, chunk, log
line, translation, prompt and .srt must match.

To regenerate after a deliberate change on the jp-subs side:
    node tests/parity/dump.mjs "<path to jp-subs checkout>"
"""

import copy
import difflib
import hashlib
import json
from pathlib import Path

import pytest

from whisper_subs._js import js_len
from whisper_subs.chunk import chunk
from whisper_subs.pipeline import run
from whisper_subs.segment import segment
from whisper_subs.srt import units_to_srt

HERE = Path(__file__).parent
GOLDEN = sorted((HERE / "parity" / "golden").glob("*.json"))

# JS option names -> Python option names.
OPTION_NAMES = {"gapMs": "gap_ms", "maxChars": "max_chars", "size": "size"}

GLOSSARY = {
    "setting": "A streamer reacts to a story cutscene in a game.",
    "speakers": "One streamer, plus in-game characters.",
    "names": {"コロンビーナ": "Columbina", "マシ": "Mashi"},
    "terms": {"配信": "stream"},
    "asr_corrections": {},
    "register": "Casual, energetic.",
}


def fake_backend():
    """Same behaviour as fakeBackend() in dump.mjs. Keep the two in step."""
    seen: dict[int, int] = {}
    analyses = 0

    def backend(prompt: str, json_mode: bool = False) -> str:
        nonlocal analyses
        if prompt.startswith("You are preparing"):
            analyses += 1
            if analyses == 1:
                return '{"setting": "  ", "names": {}}'
            return "```json\n" + json.dumps(GLOSSARY, ensure_ascii=False) + "\n```"
        body = prompt.split("LINES TO TRANSLATE\n")[1].split("\n\nRules:")[0]
        lines = []
        for line in body.split("\n"):
            n, ja = line.split("\t", 1)
            lines.append((int(n), ja))
        answer = {}
        for n, ja in lines:
            count = seen.get(n, 0)
            seen[n] = count + 1
            if n % 13 == 0:
                continue
            if n % 7 == 0 and count == 0:
                continue
            answer[str(n)] = f"line {n}: " + "lorem " * (n % 11) + f"({js_len(ja)})"
        text = json.dumps(answer, ensure_ascii=False)
        return f"Here you go:\n{text}\nDone." if lines[0][0] % 5 == 0 else text

    return backend


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def first_diff(expected: str, actual: str) -> str:
    return "\n".join(list(difflib.unified_diff(
        expected.splitlines(), actual.splitlines(), "js", "python", lineterm="", n=1))[:40])


@pytest.fixture(params=GOLDEN, ids=[g.stem for g in GOLDEN])
def case(request):
    golden = json.loads(request.param.read_text(encoding="utf-8"))
    transcript = json.loads((HERE / "fixtures" / golden["fixture"]).read_text(encoding="utf-8"))
    options = {OPTION_NAMES[k]: v for k, v in golden["options"].items()}
    return golden, transcript, options


def test_goldens_exist():
    assert len(GOLDEN) >= 10, "run tests/parity/dump.mjs to generate the goldens"


def test_segment(case):
    golden, transcript, options = case
    assert segment(transcript["cues"], options) == golden["units"]


def test_chunk(case):
    golden, transcript, options = case
    chunks = chunk(segment(transcript["cues"], options))
    summary = [{"index": c["index"], "first_unit": c["first_unit"], "before": len(c["before"]),
                "target": len(c["target"]), "after": len(c["after"])} for c in chunks]
    assert summary == golden["chunks"]


def test_pipeline_prompts_logs_and_srt(case):
    golden, transcript, options = case
    prompts, log = [], []
    backend = fake_backend()

    def recording(prompt, json_mode=False):
        prompts.append(prompt)
        return backend(prompt, json_mode)

    result = run(copy.deepcopy(transcript), recording, options, log.append)

    # Readable diffs first, for the prompt kinds most likely to drift.
    ours = next(p for p in prompts if p.startswith("You are preparing"))
    want = golden["first_analysis_prompt"]
    assert ours == want, first_diff(want, ours)
    ours = next(p for p in prompts if p.startswith("Translate Japanese"))
    assert ours == golden["first_translation_prompt"], first_diff(
        golden["first_translation_prompt"], ours)

    assert log == golden["log"]
    assert [sha(p) for p in prompts] == golden["prompt_sha256"]
    assert result["translations"] == golden["translations"]
    assert result["failures"] == golden["failures"]
    assert sha(units_to_srt(result["units"], result["translations"])) == golden["srt_sha256"]
