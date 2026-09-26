from whisper_subs.filters import clean, is_phantom
from whisper_subs.transcribe import find_holes


def seg(text, nsp=0.9, start=0.0):
    return {"start": start, "end": start + 1, "text": text, "no_speech_prob": nsp, "words": []}


def test_phantoms_need_both_the_phrase_and_weak_speech():
    assert is_phantom(seg("ご視聴ありがとうございました"))
    assert is_phantom(seg("作詞・作曲・編曲 初音ミク"))
    assert is_phantom(seg("チャンネル登録よろしくお願いします"))
    # A streamer really signing off: clear speech.
    assert not is_phantom(seg("ご視聴ありがとうございました", nsp=0.1))
    # The phrase inside a longer real sentence.
    assert not is_phantom(seg("ご視聴ありがとうございましたって言うの忘れてた"))


def test_repeated_lines_collapse_but_short_and_double_repeats_survive():
    segs = [seg("おやすみなさいませ", start=i) for i in range(4)]
    segs += [seg("そう", start=10 + i) for i in range(4)]
    segs += [seg("本当にそうなんですか", start=20 + i) for i in range(2)]
    kept, dropped = clean(segs)
    assert [s["text"] for s in kept] == ["おやすみなさいませ", "そう", "そう", "そう", "そう",
                                         "本当にそうなんですか", "本当にそうなんですか"]
    assert len(dropped) == 3 and dropped[0]["dropped"] == "repeated line"


def test_find_holes():
    speech = [(10.0, 20.0), (30.0, 31.0)]
    words = [(10.2, 11.0), (11.1, 12.0), (18.0, 19.0)]
    # 12.5-17.5 is speech with no words (0.5 s margin either side); 19.5-20 is
    # too short; 30-31 is exactly 1 s.
    assert find_holes(speech, words, min_hole=1.0) == [(12.5, 17.5), (30.0, 31.0)]
    # Holes closer than 2 s merge into one re-transcription.
    assert find_holes([(0, 3), (4, 7)], [], min_hole=1.0) == [(0, 7)]


def test_display_times():
    from whisper_subs.timing import display_times

    units = [
        {"start_ms": 0, "end_ms": 300},        # そう: flashes without help
        {"start_ms": 5000, "end_ms": 5800},    # long English, room to extend
        {"start_ms": 9000, "end_ms": 9500},    # next starts 100 ms after: flicker gap
        {"start_ms": 9600, "end_ms": 12000},
        {"start_ms": 13000, "end_ms": 14000},  # untranslated
    ]
    en = ["Yeah.", "x" * 68, "Hm.", "And then we went home.", ""]
    t = display_times(units, en)
    assert t[0] == (0, 1000)                  # stretched to the 1 s minimum
    assert t[1] == (5000, 8917)               # wants 4 s, stops 2 frames before the next
    assert t[2] == (9000, 9517)               # gap closed, two frames left
    assert t[3][1] >= 12000 and t[4] is None  # never shortened below the speech
