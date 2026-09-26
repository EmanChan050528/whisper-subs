"""Two-pass orchestration. Port of jp-subs' core/pipeline.js.

Pass 1 reads the whole transcript and builds a glossary; pass 2 translates
chunks with that glossary and read-only context on both sides.
"""

import difflib
import re
import unicodedata
from collections.abc import Callable

from whisper_subs._js import js_len, js_round
from whisper_subs.chunk import chunk
from whisper_subs.ollama import Backend, parse_json
from whisper_subs.prompt import analysis_prompt, translation_prompt
from whisper_subs.segment import segment

Log = Callable[[str], None]

#: How much Japanese to show pass 1. A four-hour archive is ~75,000 characters
#: and will not fit any local model's context, and an over-long prompt fails as
#: a truncated, unparseable reply rather than as a clear error.
MAX_ANALYSIS_CHARS = 6000


def _nolog(_msg: str) -> None:
    pass


def analysis_text(units: list[dict], budget: int = MAX_ANALYSIS_CHARS) -> dict:
    """Evenly sample units across the whole recording rather than taking a
    prefix, so the glossary still sees names and vocabulary from the end."""
    all_ja = [u["ja"] for u in units]
    total = sum(js_len(s) + 1 for s in all_ja)
    if total <= budget:
        return {"text": "\n".join(all_ja), "sampled": False}

    step = max(1, js_round(total / budget))
    kept, used = [], 0
    for i in range(0, len(all_ja), step):
        if used + js_len(all_ja[i]) > budget:
            break
        kept.append(all_ja[i])
        used += js_len(all_ja[i]) + 1
    return {"text": "\n".join(kept), "sampled": True, "kept_units": len(kept)}


def _normalise(s: str) -> str:
    return re.sub(r"[\s、。，．！？!?「」『』…・〜ー~]", "", unicodedata.normalize("NFKC", s))


def echo_matches(source: str, copied, threshold: float = 0.7) -> bool:
    """Is `copied` this line's Japanese, allowing for a slightly garbled copy?

    Models copy imperfectly (dropped punctuation, a changed kana). A shifted
    line is a *different* line, which scores far below the threshold, except
    for a genuine repeat like コロンビーナ / コロンビーナ, where either
    translation is right anyway.
    """
    if not isinstance(copied, str):
        return False
    a, b = _normalise(source), _normalise(copied)
    if not a or not b:
        return a == b
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def _count(o) -> int:
    return len(o) if isinstance(o, dict | list) else 0


def _useful(g) -> bool:
    if not isinstance(g, dict):
        return False
    setting = g.get("setting")
    return (isinstance(setting, str) and len(setting.strip()) > 0) or (
        _count(g.get("names")) + _count(g.get("terms")) + _count(g.get("asr_corrections")) > 0
    )


def analyse(units: list[dict], backend: Backend, meta: dict | None = None,
            log: Log = _nolog, seed: dict | None = None) -> dict:
    """Pass 1: whole-transcript glossary and speaker model."""
    a = analysis_text(units)
    log(f"pass 1: analysing {len(units)} units ({js_len(a['text'])} chars"
        + (f", sampled down to {a['kept_units']} units" if a["sampled"] else "") + ")")

    glossary = None
    for attempt in range(1, 4):
        try:
            glossary = parse_json(backend(analysis_prompt(a["text"], meta, seed), json_mode=True),
                                  "analysis pass")
            if _useful(glossary):
                break
            log(f"pass 1: attempt {attempt} came back empty")
        except Exception as err:
            # A malformed or truncated reply must NOT end the run. Losing the
            # glossary costs quality; raising here costs every subtitle.
            first_line = (str(err).splitlines() or [""])[0]
            log(f"pass 1: attempt {attempt} failed — {first_line}")
            glossary = None

    # Pass 1 is the whole quality advantage over per-line translation. If it is
    # empty the run still "succeeds" and quietly produces worse subtitles, so
    # say so rather than letting it pass.
    if not _useful(glossary):
        log("pass 1: WARNING — no glossary. Names, domain terms and ASR")
        log("        corrections will NOT be applied. Translating anyway;")
        log("        expect roughly per-chunk quality.")
        glossary = {}
    else:
        log(f"pass 1: {_count(glossary.get('names'))} names, {_count(glossary.get('terms'))} "
            f"terms, {_count(glossary.get('asr_corrections'))} ASR corrections")
    return glossary


def translate_units(
    units: list[dict], glossary: dict, backend: Backend, options: dict | None = None,
    log: Log = _nolog, on_progress: Callable[[list[str], int, int], None] | None = None,
) -> dict:
    """Pass 2: translate every chunk. `translations` is parallel to `units`."""
    options = options or {}
    chunks = chunk(units, {k: options[k] for k in ("size", "context_before", "context_after")
                           if k in options})
    translations = [""] * len(units)
    failures: list[str] = []
    echo = bool(options.get("echo"))

    def request(c, lines, label, last_try=False):
        """Ask for a specific set of lines; write whatever comes back."""
        raw = backend(translation_prompt(c, glossary, lines, echo=echo), json_mode=True)
        answer = parse_json(raw, label)
        if not isinstance(answer, dict):
            return
        rejected = 0
        for line in lines:
            value = answer.get(str(line["n"]))
            if echo and isinstance(value, dict):
                # The copied Japanese must be this line's. If it is a
                # neighbour's, the model has shifted: leave the line
                # outstanding so the retry asks for it on its own.
                if not echo_matches(line["ja"], value.get("ja")):
                    rejected += 1
                    continue
                value = value.get("en")
            elif echo and not last_try:
                # A bare string cannot be checked. Accept one only when the
                # alternative is a missing subtitle.
                rejected += 1
                continue
            if isinstance(value, str) and value.strip():
                translations[line["n"] - 1] = value.strip()
        if rejected:
            log(f"{label}: rejected {rejected} line(s) whose copied Japanese did not match")

    def outstanding(c):
        lines = [{"n": c["first_unit"] + i + 1, "ja": u["ja"]} for i, u in enumerate(c["target"])]
        return [line for line in lines if not translations[line["n"] - 1]]

    should_stop = options.get("should_stop")
    for c in chunks:
        # Checked between chunks: a caller that has lost interest should not
        # keep occupying the GPU.
        if should_stop and should_stop():
            log(f"stopped after {c['index']} of {len(chunks)} chunks")
            return {"translations": translations, "failures": failures, "stopped": True}

        label = f"chunk {c['index'] + 1}/{len(chunks)}"
        total = len(c["target"])

        try:
            request(c, outstanding(c), label)
        except Exception as err:
            log(f"{label}: {err}")

        # Models drop keys from long JSON objects. Re-ask for only the missing
        # lines: a shorter request usually succeeds where the full one did not.
        for attempt in (1, 2):
            missing = outstanding(c)
            if not missing:
                break
            log(f"{label}: retrying {len(missing)} missing line(s)")
            try:
                request(c, missing, f"{label} retry {attempt}", last_try=attempt == 2)
            except Exception as err:
                log(f"{label}: retry {attempt} failed — {err}")

        if on_progress:
            on_progress(translations, c["index"] + 1, len(chunks))

        left = len(outstanding(c))
        if left:
            failures.append(f"{label}: {left} of {total} lines still missing after 2 retries")
        log(f"{label}: {total - left}/{total} lines")

    return {"translations": translations, "failures": failures}


def run(transcript: dict, backend: Backend, options: dict | None = None,
        log: Log = _nolog, on_progress=None, seed: dict | None = None) -> dict:
    options = options or {}
    cues = transcript.get("cues") or []
    if not cues:
        raise ValueError("Transcript contains no cues.")

    units = segment(cues, {k: options[k] for k in ("gap_ms", "max_chars", "drop_tag_only_cues",
                                                 "cue_end_min_chars")
                           if k in options})
    log(f"segmented {len(cues)} cues into {len(units)} units")

    glossary = analyse(units, backend, transcript, log, seed)
    result = translate_units(units, glossary, backend, options, log, on_progress)
    translations = result["translations"]
    return {
        "units": units, "glossary": glossary, "translations": translations,
        "failures": result["failures"], "translated": sum(1 for t in translations if t),
    }
