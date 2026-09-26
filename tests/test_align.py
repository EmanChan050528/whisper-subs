import json

from whisper_subs import align
from whisper_subs.align import runs


def test_only_runs_count_as_shifts():
    v = ["same", "prev", "same", "next", "next", "next", "same", "prev", "prev", "?", "prev"]
    assert runs(v) == [(3, 5, "next"), (7, 8, "prev")]   # a lone vote is noise
    assert runs(v, min_run=3) == [(3, 5, "next")]
    assert runs(["none"] * 5) == []              # "matches nothing" is not a direction


def test_check_and_repair_retranslates_only_the_flagged_run(monkeypatch):
    units = [{"ja": f"文{i}", "start_ms": i * 1000, "end_ms": i * 1000 + 900} for i in range(8)]
    shifted = ["S0", "S0", "S1", "S2", "S3", "S5", "S6", "S7"]  # lines 1-4 shifted by one

    def fake_judge(pairs, model=None, only=None):
        idx = only if only is not None else range(len(pairs))
        out = ["?"] * len(pairs)
        for i in idx:
            out[i] = "same" if pairs[i]["en"] == f"S{i}" else "prev"
        return out

    asked = []

    def backend(prompt, json_mode=False):
        body = prompt.split("LINES TO TRANSLATE\n")[1].split("\n\nRules:")[0]
        lines = [line.split("\t") for line in body.split("\n")]
        asked.extend(int(n) for n, _ in lines)
        return json.dumps({n: {"ja": ja, "en": f"S{int(n) - 1}"} for n, ja in lines},
                          ensure_ascii=False)

    monkeypatch.setattr(align, "judge", fake_judge)
    fixed, report = align.check_and_repair(units, shifted, {}, backend, {"size": 20})
    assert [(a, b) for a, b, _ in report["found"]] == [(1, 4)]
    assert sorted(asked) == [1, 2, 3, 4, 5, 6, 7]  # the run 2-5, plus 2 either side
    assert fixed == [f"S{i}" for i in range(5)] + ["S5", "S6", "S7"]
    assert report["left"] == []
