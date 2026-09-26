// Regenerate the parity goldens from jp-subs' own JS.
//
//   node tests/parity/dump.mjs "D:/Desktop/Translator Project"
//
// For every fixture and preset this runs jp-subs' real segment(), chunk(),
// run() and toSrt() with a deterministic fake model, and writes what they
// produced to tests/parity/golden/. tests/test_parity.py runs the Python port
// with the same fake and requires identical output, so the goldens (not
// jp-subs) are what the Python tests depend on.
//
// The fake is duplicated in tests/test_parity.py. Keep the two in step.

import { readFileSync, writeFileSync, mkdirSync, readdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const jpSubs = process.argv[2];
if (!jpSubs) {
  console.error('usage: node tests/parity/dump.mjs "<path to jp-subs checkout>"');
  process.exit(1);
}
const core = (name) => pathToFileURL(join(jpSubs, "extension/src/core", name)).href;
const { segment } = await import(core("segment.js"));
const { chunk } = await import(core("chunk.js"));
const { run } = await import(core("pipeline.js"));
const { toSrt } = await import(core("srt.js"));

const here = dirname(fileURLToPath(import.meta.url));
const fixtures = join(here, "..", "fixtures");
const out = join(here, "golden");
mkdirSync(out, { recursive: true });

export const PRESETS = {
  default: {},
  whisper: { gapMs: 500 },
  echo: { gapMs: 500, echo: true },  // copy-then-translate replies
};

const GLOSSARY = {
  setting: "A streamer reacts to a story cutscene in a game.",
  speakers: "One streamer, plus in-game characters.",
  names: { "コロンビーナ": "Columbina", "マシ": "Mashi" },
  terms: { "配信": "stream" },
  asr_corrections: {},
  register: "Casual, energetic.",
};

/** Deterministic stand-in for the model. See the header comment. */
function fakeBackend() {
  const seen = new Map();
  let analyses = 0;
  return async (prompt) => {
    if (prompt.startsWith("You are preparing")) {
      analyses += 1;
      // First reply is useless, to exercise the retry; the second is fenced,
      // to exercise parseJson's recovery.
      if (analyses === 1) return '{"setting": "  ", "names": {}}';
      return "```json\n" + JSON.stringify(GLOSSARY) + "\n```";
    }
    const body = prompt.split("LINES TO TRANSLATE\n")[1].split("\n\nRules:")[0];
    const lines = body.split("\n").map((l) => {
      const tab = l.indexOf("\t");
      return { n: Number(l.slice(0, tab)), ja: l.slice(tab + 1) };
    });
    const answer = {};
    for (const { n, ja } of lines) {
      const count = seen.get(n) || 0;
      seen.set(n, count + 1);
      if (n % 13 === 0) continue;                // never answered: failure path
      if (n % 7 === 0 && count === 0) continue;  // answered on retry
      answer[String(n)] = `line ${n}: ` + "lorem ".repeat(n % 11) + `(${ja.length})`;
    }
    const json = JSON.stringify(answer);
    // Some replies arrive wrapped in prose.
    return lines[0].n % 5 === 0 ? `Here you go:\n${json}\nDone.` : json;
  };
}

const sha = (s) => createHash("sha256").update(s, "utf8").digest("hex");

for (const file of readdirSync(fixtures).filter((f) => f.endsWith(".ja.json")).sort()) {
  const transcript = JSON.parse(readFileSync(join(fixtures, file), "utf8"));
  for (const [preset, options] of Object.entries(PRESETS)) {
    const units = segment(transcript.cues, options);
    const chunks = chunk(units).map((c) => ({
      index: c.index, first_unit: c.firstUnit,
      before: c.before.length, target: c.target.length, after: c.after.length,
    }));

    const prompts = [];
    const backend = fakeBackend();
    const recording = async (prompt, opts) => { prompts.push(prompt); return backend(prompt, opts); };
    const log = [];
    const result = await run(structuredClone(transcript), recording, options, (m) => log.push(m));
    const srt = toSrt(result.units, result.translations);

    const golden = {
      fixture: file, preset, options,
      units,
      chunks,
      log,
      failures: result.failures,
      translations: result.translations,
      srt_sha256: sha(srt),
      prompt_sha256: prompts.map(sha),
      // Full text of the first prompt of each kind, for a readable diff.
      first_analysis_prompt: prompts.find((p) => p.startsWith("You are preparing")),
      first_translation_prompt: prompts.find((p) => p.startsWith("Translate Japanese")),
    };
    const name = `${file.replace(/\.ja\.json$/, "")}.${preset}.json`;
    writeFileSync(join(out, name), JSON.stringify(golden, null, 1) + "\n");
    console.log(`${name}: ${units.length} units, ${prompts.length} prompts, ` +
                `${result.failures.length} failures`);
  }
}
