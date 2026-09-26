"""Find translations that landed on the wrong line.

Used by `whisper-subs --check` (which also repairs what it finds) and by
`eval/align.py` (which only reports).

The failure (DEVELOPMENT.md 2.5): on fragmented speech the translator rebuilds
whole sentences and spreads the English across neighbouring numbers, so a run
of lines each shows the English of the line before or after. Every line still
has a translation, so no other metric sees it.

The check asks a model, for each line, which of three Japanese lines (the
previous one, its own, the next one) its English translates. Short fragments
like そう are ambiguous one at a time, so single votes are noise; a shift is a
RUN of lines pointing the same wrong way (2 or more), and only runs are reported.
"""

import sys

from whisper_subs.ollama import ollama_backend, parse_json

PROMPT = """Each item below gives an English subtitle and three Japanese lines labelled A, B and C. Say which Japanese line the English is a translation of.

- A, B and C are consecutive lines of the same conversation, so they may be related. Pick the one whose meaning the English actually expresses.
- If the English covers two of them, pick the one it covers most.
- If it matches none, answer "none".

{items}

Return JSON mapping each item number to "A", "B", "C" or "none", and nothing else: {{"1": "B", ...}}"""  # noqa: E501

BATCH = 15
#: Shortest run reported. 2, not 3: on the test files a 2-line threshold
#: raised no false alarms (342 clean lines) and caught one more real shift.
MIN_RUN = 2
#: Lines either side of a run that are re-translated with it. A shift spreads:
#: on the hour clip, two lines between two flagged runs were still shifted
#: but too short a run to be flagged themselves.
MARGIN = 2


#: A 2B judge raised a false alarm on clean output and broke planted runs
#: apart; the 9B found every planted and real shift with no false alarms.
DEFAULT_JUDGE = "qwen3.5:9b"


def judge(units: list[dict], model: str = DEFAULT_JUDGE,
          only: list[int] | None = None) -> list[str]:
    """Per unit ({"ja", "en"}): 'prev', 'same', 'next', 'none', or '?' (not
    asked, or no answer). `only` limits the question to those indices."""
    # Temperature 0: the same file must get the same verdict every time. At
    # 0.2 one run flagged lines 84-86 of the hour clip and the next did not.
    # Same num_ctx as the translator: a different size makes Ollama reload the
    # model between translating and checking.
    backend = ollama_backend(model=model, temperature=0)
    verdicts = ["?"] * len(units)
    todo = [i for i in (only if only is not None else range(len(units))) if units[i].get("en")]
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        items = []
        for k, i in enumerate(batch, 1):
            def ja(j):
                return units[j]["ja"] if 0 <= j < len(units) else "(nothing)"
            items.append(f'{k}. English: "{units[i]["en"]}"\n'
                         f"   A: {ja(i - 1)}\n   B: {ja(i)}\n   C: {ja(i + 1)}")
        try:
            answer = parse_json(backend(PROMPT.format(items="\n\n".join(items)),
                                        json_mode=True))
        except Exception as err:  # one bad batch should not end the check
            print(f"  batch at line {batch[0] + 1}: {err}", file=sys.stderr)
            continue
        for k, i in enumerate(batch, 1):
            v = str(answer.get(str(k), "")).strip().upper()
            verdicts[i] = {"A": "prev", "B": "same", "C": "next", "NONE": "none"}.get(v, "?")
    return verdicts


def runs(verdicts: list[str], min_run: int = MIN_RUN) -> list[tuple[int, int, str]]:
    """(first, last, direction) for runs of min_run+ lines pointing the same way."""
    out, i = [], 0
    while i < len(verdicts):
        v = verdicts[i]
        j = i
        while j + 1 < len(verdicts) and verdicts[j + 1] == v:
            j += 1
        if v in ("prev", "next") and j - i + 1 >= min_run:
            out.append((i, j, v))
        i = j + 1
    return out


def check_and_repair(units: list[dict], translations: list[str], glossary: dict, backend,
                     options: dict, log=lambda m: None,
                     model: str = DEFAULT_JUDGE) -> tuple[list[str], dict]:
    """Find shifted runs, re-translate just those lines, and look again.

    -> (translations, {"found": runs before, "left": runs after repair}).
    The re-translation goes through translate_units with the flagged lines
    blanked, so only their chunks are asked again, and always with echo.
    """
    from whisper_subs.pipeline import translate_units

    def pairs(ts):
        return [{"ja": u["ja"], "en": t} for u, t in zip(units, ts, strict=True)]

    log(f"checking {sum(1 for t in translations if t)} lines for translations on the "
        f"wrong line ({model})")
    found = runs(judge(pairs(translations), model))
    if not found:
        log("check: no shifted lines")
        return translations, {"found": [], "left": []}
    flagged = sorted({i for a, b, _ in found
                      for i in range(max(0, a - MARGIN), min(len(units), b + 1 + MARGIN))})
    log(f"check: {len(found)} shifted run(s) ("
        + ", ".join(f"{a + 1}-{b + 1}" for a, b, _ in found)
        + f"); re-translating {len(flagged)} lines around them")

    blanked = [("" if i in set(flagged) else t) for i, t in enumerate(translations)]
    repaired = translate_units(units, glossary, backend, {**options, "echo": True}, log,
                               initial=blanked)["translations"]
    # A line the retry could not fill keeps its old translation: probably
    # shifted, but better than a gap.
    repaired = [r or t for r, t in zip(repaired, translations, strict=True)]

    near = sorted({j for i in flagged for j in (i - 1, i, i + 1) if 0 <= j < len(units)})
    left = runs(judge(pairs(repaired), model, only=near))
    log("check: repaired" if not left else
        f"check: {len(left)} run(s) still look shifted: "
        + ", ".join(f"{a + 1}-{b + 1}" for a, b, _ in left))
    return repaired, {"found": found, "left": left}
