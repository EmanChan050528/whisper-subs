"""Glossaries that persist across files of the same channel or series.

jp-subs keeps one per YouTube channel. Local files have no channel ID, so the
user names the glossary: `--glossary genshin`. Each run seeds pass 1 with it,
so names keep the same English rendering, then merges pass 1's findings back.

Existing entries always win a merge. That is what stops a spelling drifting
between episodes, and it is also what makes hand edits stick: pass 1 rendered
コロンビーナ as "Colombine" on one run and "Colonnella" on the next, so the file
is meant to be opened and corrected, and a correction must survive every later
run.

The names are NOT given to Whisper as hotwords. Tried on NSY6YHXbxtA: Whisper
recited the list itself over music ("コロンビーナ クータル ハイポステレニア
サンドローネ ファデュイ ..." as 30 s cues), in place of the real speech, and ran
3x slower. With the recitations filtered out, 28% of the lines YouTube
captioned were missing, against 10% without hotwords. A misheard name is
cheaper to live with: pass 1 lists it as an ASR correction and the
translation repairs it.
"""

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

#: Per category, as jp-subs: keeps the pass-1 prompt small.
MAX_ENTRIES = 40

_NAME = re.compile(r"^[\w\-. ]+$")


def home() -> Path:
    """%APPDATA%/whisper-subs on Windows; WHISPER_SUBS_HOME overrides."""
    if os.environ.get("WHISPER_SUBS_HOME"):
        return Path(os.environ["WHISPER_SUBS_HOME"])
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "whisper-subs"
    return Path.home() / ".config" / "whisper-subs"


def path(name: str) -> Path:
    if not _NAME.match(name) or name.strip(". ") != name:
        raise ValueError(f"glossary name {name!r}: use letters, digits, - _ . and spaces")
    return home() / "glossaries" / f"{name}.json"


def load(name: str) -> dict:
    """The stored glossary, or an empty one."""
    p = path(name)
    if not p.is_file():
        return {"name": name, "names": {}, "terms": {}, "files": 0}
    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("names", {})
    data.setdefault("terms", {})
    return data


def seed(glossary: dict) -> dict | None:
    """What pass 1 is told is already established, or None for a new glossary."""
    if not glossary.get("names") and not glossary.get("terms"):
        return None
    return {"names": glossary["names"], "terms": glossary["terms"]}


def _merge(base: dict, add) -> dict:
    """Existing entries win; new ones are appended up to MAX_ENTRIES."""
    out = dict(base or {})
    for k, v in (add or {}).items() if isinstance(add, dict) else []:
        if len(out) >= MAX_ENTRIES:
            break
        if isinstance(k, str) and isinstance(v, str) and k.strip() and k not in out:
            out[k] = v
    return out


def remember(name: str, found: dict, source: str | None = None) -> tuple[Path, int]:
    """Merge one file's pass-1 glossary in. -> (path, number of new entries)."""
    stored = load(name)
    before = len(stored["names"]) + len(stored["terms"])
    stored["names"] = _merge(stored["names"], found.get("names"))
    stored["terms"] = _merge(stored["terms"], found.get("terms"))
    stored["files"] = stored.get("files", 0) + 1
    stored["updated"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    if source:
        stored["last_file"] = source
    p = path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)  # never leave a half-written file that a hand edit went into
    return p, len(stored["names"]) + len(stored["terms"]) - before
