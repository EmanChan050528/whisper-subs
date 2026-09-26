from whisper_subs.cues import to_cues, to_transcript
from whisper_subs.srt import cues_to_srt, timestamp


def seg(start, end, words, text=None):
    return {
        "start": start, "end": end,
        "text": text if text is not None else "".join(w for _, w in words),
        "words": [{"start": t, "end": t + 0.1, "word": w, "p": 0.9} for t, w in words],
    }


def test_timestamp():
    assert timestamp(0) == "00:00:00,000"
    assert timestamp(3_723_456) == "01:02:03,456"
    assert timestamp(-5) == "00:00:00,000"


def test_cue_text_matches_concatenated_segs():
    cues = to_cues([seg(1.23, 2.61, [(1.23, " こんにちは"), (1.8, "、藤森"), (2.2, "です。")])])
    assert cues == [{
        # Timed from the first word's start to the last word's end.
        "t_ms": 1230, "dur_ms": 1070, "ja": "こんにちは、藤森です。",
        "segs": [{"text": "こんにちは", "t_ms": 1230}, {"text": "、藤森", "t_ms": 1800},
                 {"text": "です。", "t_ms": 2200}],
    }]
    assert "".join(s["text"] for s in cues[0]["segs"]) == cues[0]["ja"]


def test_segment_split_where_words_are_far_apart():
    # Measured on NSY6YHXbxtA: one segment, first word 43 s before the rest.
    cues = to_cues([seg(160.6, 205.9, [(160.9, "え?"), (204.3, "俺"), (204.5, "じゃない")])])
    assert [(c["t_ms"], c["ja"]) for c in cues] == [(160900, "え?"), (204300, "俺じゃない")]
    assert cues[0]["dur_ms"] == 100


def test_empty_segments_are_dropped_and_wordless_ones_kept():
    cues = to_cues([seg(0, 1, [(0, " ")]), seg(2, 3, [], text=" そう ")])
    assert cues == [{"t_ms": 2000, "dur_ms": 1000, "ja": "そう"}]


def test_transcript_has_jp_subs_shape():
    whisper = {
        "options": {"model": "large-v3", "device": "cuda", "compute_type": "float16"},
        "duration": 51.1,
        "segments": [seg(0, 1, [(0, "はい")])],
    }
    t = to_transcript(whisper, title="clip")
    assert t["title"] == "clip" and t["duration_s"] == 51 and t["cue_count"] == 1
    assert isinstance(t["cues"], list) and t["source"].startswith("whisper large-v3")


def test_srt_stretches_short_cues_but_not_into_the_next():
    cues = [
        {"t_ms": 0, "dur_ms": 200, "ja": "そう"},          # stretched to 700
        {"t_ms": 1000, "dur_ms": 100, "ja": "外で"},        # stretch capped at next start
        {"t_ms": 1400, "dur_ms": 2000, "ja": "家でゲームしたり"},
    ]
    srt = cues_to_srt(cues)
    assert "00:00:00,000 --> 00:00:00,700\nそう" in srt
    assert "00:00:01,000 --> 00:00:01,400\n外で" in srt
    assert "00:00:01,400 --> 00:00:03,400\n家でゲームしたり" in srt
    assert srt.startswith("1\n") and "\n3\n" in srt


def test_whisper_preset_ends_units_at_cue_ends_but_merges_short_fragments():
    from whisper_subs.segment import WHISPER, segment

    cues = [  # back to back, so the gap rule never fires
        {"t_ms": 0, "dur_ms": 2000, "ja": "人生という長い旅路の中では"},
        {"t_ms": 2000, "dur_ms": 900, "ja": "その人たちが"},       # 6 chars: merges on
        {"t_ms": 2900, "dur_ms": 1500, "ja": "違う名前で呼んだとしても"},
        {"t_ms": 4400, "dur_ms": 300, "ja": "そう"},
    ]
    # jp-subs' default: no punctuation and no gap, so it all becomes one unit.
    assert [u["ja"] for u in segment(cues)] == [
        "人生という長い旅路の中ではその人たちが違う名前で呼んだとしてもそう"]
    assert [u["ja"] for u in segment(cues, WHISPER)] == [
        "人生という長い旅路の中では", "その人たちが違う名前で呼んだとしても", "そう"]


def w(start, end, word):
    return {"start": start, "end": end, "word": word, "p": 0.9}


def test_a_stretched_last_word_no_longer_holds_the_subtitle_up():
    # 「でした」 stretched over 6.7 s of music (measured on NSY6YHXbxtA)
    cues = to_cues([{"start": 870.2, "end": 879.8, "text": "お疲れ様でした", "words": [
        w(870.2, 871.6, "お疲れ"), w(871.6, 873.1, "様"), w(873.1, 879.8, "でした")]}])
    assert cues[0]["t_ms"] == 870200 and cues[0]["dur_ms"] == 4400  # 873.1 + 1.5 - 870.2


def test_long_cues_split_only_at_a_clause_break():
    from whisper_subs.cues import _breakable, split_long

    assert _breakable("見て", "日本") and _breakable("けど、", "今")
    assert not _breakable("奥", "ゆかしい")        # mid-word, after kanji
    assert not _breakable("ちょ", "っと")           # before a small kana
    words = [w(0, 3, "私の話聞いて"), w(3.6, 5, "ますかって"), w(5, 8, "何回も言われました")]
    # 8 s long. The 0.6 s pause after 聞いて (ending in hiragana) is a clause
    # break, and both halves keep 6+ characters, so it splits there.
    assert [len(p) for p in split_long(words)] == [1, 2]
    dense = [w(0, 4, "やっぱり和歌にも見られる"), w(4.05, 8, "ような表現力")]
    assert len(split_long(dense)) == 1           # no real pause: left whole
