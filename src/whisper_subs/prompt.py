"""Prompt construction for both passes. Port of jp-subs' core/prompt.js.

The rules below are not generic "translate well" advice: each one targets a
failure category measured on YouTube's own auto-translation in jp-subs'
eval/README.md. Change them only with a fixture run to back it up.

The template text is kept VERBATIM, including the JS `${...}` placeholders,
which are filled by plain replacement so that nothing needs escaping and a
diff against prompt.js stays readable. tests/test_parity.py checks the output
byte for byte against the JS.
"""

import json
import re


def _dumps(obj) -> str:
    """JSON.stringify(obj, null, 2)."""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _fill(template: str, values: dict[str, str]) -> str:
    # One pass, so a "${...}" inside the transcript itself is never expanded.
    return re.sub(r"\$\{(\w+)\}", lambda m: values[m.group(1)], template)


ANALYSIS = """You are preparing to translate a Japanese video transcript into English subtitles.
${seeded}
Before translating, read the whole transcript and build a reference sheet.

${title}${author}
TRANSCRIPT
${fullJapaneseText}

Produce JSON with exactly these keys:

{
  "setting": "One or two sentences: what is happening, what kind of video, what is being played or discussed. Be specific — this is used to disambiguate homophones later.",
  "speakers": "Who is talking. Note especially whether anyone refers to themselves in the third person by name or nickname, which is common for streamers.",
  "names": { "japanese term": "how to render it in English, consistently" },
  "terms": { "japanese term": "English meaning IN THIS CONTEXT, not the dictionary default" },
  "asr_corrections": { "misrecognised form": "what was almost certainly said, and why" },
  "register": "How this speaker sounds, and what English register matches. One or two sentences."
}

Guidance:
- "terms" is for words whose ordinary dictionary sense would be wrong here. For example 配信 is "stream", not "delivery", when the speaker is a streamer.
- "asr_corrections" is for speech-recognition errors you can identify from context. Japanese homophones are the usual cause. Only list ones you are confident about.
- If a category is empty, use an empty object. Do not invent entries.
- Output only the JSON object. No preamble, no code fence.

Hard limits — a reply that breaks these is useless:
- **This is a reference sheet, NOT a translation.** Do not translate the transcript. Do not add an entry per line.
- Keys in "names", "terms" and "asr_corrections" must be single words or short phrases. Never a whole sentence or a whole line of dialogue.
- At most 12 entries in "names", 15 in "terms", 10 in "asr_corrections". Choose the ones that matter most and leave the rest out.
- Keep the whole reply under 2000 characters."""  # noqa: E501


TRANSLATION = """Translate Japanese video dialogue into English subtitles.

REFERENCE SHEET
${glossary}

${before}
${after}
LINES TO TRANSLATE
${numbered}

Rules:
1. Japanese omits subjects constantly and has no verb agreement to recover them from. Decide who or what each line is about using the context above and the reference sheet. Do not default to "I" — it is frequently something on screen, or the person being spoken to.
2. Japanese puts negation, tense and politeness at the END of a clause. If a line's meaning depends on a clause that finishes in a later line, translate it so the pair reads correctly together. Never assert the opposite of what was meant.
3. Some lines are sentence fragments. Translate a fragment as a fragment that joins onto its neighbours. Do not inflate one into a standalone sentence, and never read a grammatical ending as a name.
4. Use the reference sheet for names and terms, every time, without variation.
5. Apply the ASR corrections from the reference sheet where the misrecognised form appears.
6. Match the register on the reference sheet. Keep it natural spoken English, not literal glosses.
7. Never invent content that is not in the source. If a line is genuinely unclear, translate the part you are sure of.
8. Subtitles are read at speed and get two lines on screen. Keep each translation close to the length of its Japanese source and never longer than about 100 characters. Do not explain, expand, add background, or spell out what is merely implied — a line that needs a footnote should still be translated as the line, not the footnote.
9. Translate ONLY the numbered lines. The context sections are for understanding; never fold their content into an answer.
10. Output plain sentences. No leading or trailing ellipses, no surrounding quotation marks, no speaker labels.

${reply}"""  # noqa: E501

#: jp-subs' reply format: {"12": "English", ...}. Kept for parity.
REPLY_PLAIN = """Return JSON mapping each line number to its English translation, and nothing else:

{${example}}

Every number listed above must appear exactly once. No preamble, no code fence."""

#: Not in jp-subs. On fragmented conversation the model rebuilds whole
#: sentences and redistributes them across the numbers, shifting every later
#: line (docs/benchmarks.md, milestone 2). Copying each line's Japanese right
#: before its English anchors the translation to that line, and a copy that
#: does not match the line reveals a shift, so the pipeline can reject it.
#: Copying only the first 6 characters (to halve the output tokens) was tried
#: and rejected: on the hour clip the model's fragments slid onto the previous
#: line, 17 lines per run failed the check, and it ran slower, not faster.
REPLY_ECHO = """Return JSON mapping each line number to an object holding that line's Japanese, copied exactly, and its English translation, and nothing else:

{${example}}

Every number listed above must appear exactly once, with its own Japanese copied into "ja". Each "en" translates only the Japanese in its own "ja". When a sentence runs across several lines, split the English at the same places. Never move words to a neighbouring line, and never leave a line's "en" empty because its meaning was folded into another line. No preamble, no code fence."""  # noqa: E501


def analysis_prompt(full_japanese_text: str, meta: dict | None = None,
                    seed: dict | None = None) -> str:
    """Pass 1. Read the whole transcript, produce a glossary and speaker model.

    This is where consistency comes from. It also repairs ASR errors:
    高感度イベント is a homophone of 好感度イベント and only one is meaningful in
    a farming sim, but you have to know it is a farming sim.
    """
    meta = meta or {}
    title = f"Video title: {meta['title']}\n" if meta.get("title") else ""
    author = f"Channel: {meta['author']}\n" if meta.get("author") else ""

    # Names and recurring terms belong to a channel or series, not one video.
    # Carrying them across is what stops a name drifting between episodes.
    seeded = ""
    if seed and (seed.get("names") or seed.get("terms")):
        seeded = (
            "\nALREADY ESTABLISHED FOR THIS CHANNEL. Reuse these spellings exactly, "
            "and add to them:\n"
            + _dumps({"names": seed.get("names") or {}, "terms": seed.get("terms") or {}})
            + "\n"
        )

    return _fill(ANALYSIS, {
        "seeded": seeded, "title": title, "author": author,
        "fullJapaneseText": full_japanese_text,
    })


def translation_prompt(chunk: dict, glossary: dict, lines: list[dict] | None = None,
                       echo: bool = False) -> str:
    """Pass 2. Translate one chunk, with surrounding units as read-only context.

    `lines` carries explicit numbers ({"n", "ja"}), so a retry can resend an
    arbitrary subset of a chunk without the numbering drifting. `echo` asks for
    {"n": {"ja", "en"}} instead of jp-subs' {"n": "en"}; see REPLY_ECHO.
    """
    def context(units, label):
        return f"{label}\n" + "\n".join(u["ja"] for u in units) + "\n" if units else ""

    # `lines is not None`, not `lines or`: an empty array is truthy in the JS.
    items = lines if lines is not None else [
        {"n": chunk["first_unit"] + i + 1, "ja": u["ja"]} for i, u in enumerate(chunk["target"])
    ]

    # The reply section is itself a template: splice it in before filling.
    template = TRANSLATION.replace("${reply}", REPLY_ECHO if echo else REPLY_PLAIN)
    return _fill(template, {
        "glossary": _dumps(glossary),
        "before": context(chunk["before"],
                          "CONTEXT — the lines immediately before (do not translate):"),
        "after": context(chunk["after"],
                         "CONTEXT — the lines immediately after (do not translate). Use these: "
                         "in Japanese, what comes next often reveals who or what the current "
                         "line is about."),
        "numbered": "\n".join(f"{it['n']}\t{it['ja']}" for it in items),
        "example": ", ".join(
            f'"{it["n"]}": {{"ja": {json.dumps(it["ja"], ensure_ascii=False)}, "en": "..."}}'
            if echo else f'"{it["n"]}": "..."'
            for it in items[:2]
        ),
    })
