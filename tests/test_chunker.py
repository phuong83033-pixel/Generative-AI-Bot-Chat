"""Characterisation tests for ClauseChunker.

These pin what the chunker does TODAY, so the M8 change to the first-clause rule can
be made deliberately: any test that changes outcome afterwards is a decision, not an
accident. Where current behaviour is a defect, the test states the behaviour we want
and is marked xfail with the reason, so the defect is on record and flips to a pass
when it is fixed.

Plain sync tests around asyncio.run(), so no pytest-asyncio mode needs configuring.
"""

import asyncio

import pytest

from amadeus.synthesis.chunker import ClauseChunker


def chunk(tokens, stop=None, **chunker_kwargs):
    """Feed `tokens` through a fresh chunker and return the emitted clauses."""

    async def source():
        for t in tokens:
            yield t

    async def run():
        return [c async for c in ClauseChunker(**chunker_kwargs).process(source(), stop)]

    return asyncio.run(run())


# ----------------------------------------------------------------- basics

def test_empty_stream_yields_nothing():
    assert chunk([]) == []


def test_whitespace_only_tokens_never_become_empty_clauses():
    assert chunk(["  ", "\n", " "]) == []


def test_splits_at_sentence_end():
    tokens = ["Hello", " there", ",", " friend", ".", " How", " are", " you", "?"]
    assert chunk(tokens) == ["Hello there, friend.", "How are you?"]


def test_trailing_text_without_punctuation_is_flushed():
    assert chunk(["No", " full", " stop", " here"]) == ["No full stop here"]


def test_clauses_are_stripped():
    out = chunk(["  Padded", " sentence", " here", ".  "])
    assert out == ["Padded sentence here."]


# ----------------------------------------------------------------- length rules

def test_short_opening_is_swallowed_into_the_next_sentence():
    """'Hello!' is 6 chars, under MIN_CLAUSE_LENGTH (10), so it is NOT emitted alone.

    This is the behaviour M8.5 changes. It is why the first clause of a greeting is
    the whole first sentence ("Hello! I'm Amadeus, ... explore.") and costs ~1650 ms
    of synthesis before the user hears anything.
    """
    out = chunk(["Hello", "!", " I'm", " Amadeus", ",", " glad", " to", " help", "."])
    assert out == ["Hello! I'm Amadeus, glad to help."]


def test_comma_does_not_break_a_short_clause():
    """Soft breaks (comma, dashes) need 30+ characters."""
    out = chunk(["Well", ",", " yes", ".", " Sure", " thing", "."])
    assert out == ["Well, yes.", "Sure thing."]


def test_comma_breaks_once_the_clause_is_long_enough():
    long_part = "This is a rather long opening part"  # 34 chars; with the comma, 35 (>= 30)
    out = chunk([long_part, ",", " and", " then", " it", " ends", "."])
    assert out[0] == long_part + ","
    assert out[1] == "and then it ends."


def test_forced_break_at_max_length():
    # ~10-char tokens with no punctuation at all: only MAX_CLAUSE_LENGTH can split it.
    tokens = ["abcdefghi "] * 20  # 200 chars
    out = chunk(tokens)
    assert len(out) >= 2
    assert all(len(c) <= ClauseChunker.MAX_CLAUSE_LENGTH + 10 for c in out)
    assert len(out[0]) >= ClauseChunker.MAX_CLAUSE_LENGTH - 10


# ----------------------------------------------------------------- cancellation

def test_stop_event_set_up_front_yields_nothing():
    stop = asyncio.Event()
    stop.set()
    assert chunk(["Hello", " world", "."], stop) == []


def test_stop_event_mid_stream_halts_output():
    stop = asyncio.Event()

    async def source():
        yield "First sentence here."
        stop.set()
        yield " Second sentence here."

    async def run():
        return [c async for c in ClauseChunker().process(source(), stop)]

    out = asyncio.run(run())
    assert out == ["First sentence here."]


# ----------------------------------------------------------------- known defects

@pytest.mark.xfail(
    reason="A decimal point is treated as a sentence end, so '3.5' is spoken as "
    "'three point' [pause] 'five'. Needs a digit-before-and-after check.",
    strict=True,
)
def test_decimal_number_is_not_split():
    out = chunk(["It", " costs", " about", " 3", ".", "5", " dollars", " today", "."])
    assert out == ["It costs about 3.5 dollars today."]


@pytest.mark.xfail(
    reason="A colon inside a time ('10:30') is treated as a clause end.",
    strict=True,
)
def test_time_of_day_is_not_split():
    out = chunk(["We", " meet", " at", " 10", ":", "30", " tomorrow", "."])
    assert out == ["We meet at 10:30 tomorrow."]


# ----------------------------------------------------------------- first_clause_min (M8)

GREETING = ["Hello", "!", " I'm", " Amadeus", ",", " glad", " to", " help", "."]


def test_default_is_the_original_behaviour():
    assert chunk(GREETING) == ["Hello! I'm Amadeus, glad to help."]
    assert chunk(GREETING, first_clause_min=None) == ["Hello! I'm Amadeus, glad to help."]


def test_zero_breaks_at_the_first_punctuation():
    assert chunk(GREETING, first_clause_min=0) == ["Hello!", "I'm Amadeus, glad to help."]


def test_threshold_lets_a_soft_break_end_the_first_clause():
    """'Hello! I'm Amadeus,' is 19 chars: past 18, so the comma ends clause 1 even
    though the usual soft-break minimum is 30."""
    out = chunk(GREETING, first_clause_min=18)
    assert out == ["Hello! I'm Amadeus,", "glad to help."]


def test_threshold_holds_a_short_opener_back():
    """Below the threshold a hard end does NOT break; the clause keeps growing."""
    out = chunk(["Sure", " thing", ".", " I", " can", " help", " with", " that", " today", "."],
                first_clause_min=20)
    assert out == ["Sure thing. I can help with that today."]


def test_threshold_applies_to_the_first_clause_only():
    """Clause 2 uses the ordinary rules: 'glad,' (5 chars) must not break on its comma."""
    tokens = ["Hello", "!", " I'm", " Amadeus", ",", " glad", ",", " to", " help", " you", "."]
    assert chunk(tokens, first_clause_min=18) == ["Hello! I'm Amadeus,", "glad, to help you."]


def test_max_length_still_forces_a_break_with_a_huge_threshold():
    out = chunk(["abcdefghi "] * 20, first_clause_min=500)
    assert len(out) >= 2 and len(out[0]) >= ClauseChunker.MAX_CLAUSE_LENGTH - 10


def test_first_clause_rule_resets_for_every_reply():
    """One chunker serves many replies; 'first' must be per process() call."""
    chunker = ClauseChunker(first_clause_min=0)

    async def one_reply():
        async def source():
            for t in GREETING:
                yield t
        return [c async for c in chunker.process(source())]

    async def both():
        return await one_reply(), await one_reply()

    a, b = asyncio.run(both())
    assert a == b == ["Hello!", "I'm Amadeus, glad to help."]


# ----------------------------------------------------------------- first_clause_max (M8, opt-in)

NO_COMMA = ["Sure", "!", " Did", " you", " know", " that", " the", " brain", " uses",
            " about", " twenty", " percent", " of", " your", " energy", "?"]


def test_cap_is_off_by_default():
    assert chunk(NO_COMMA) == ["Sure! Did you know that the brain uses about twenty percent of your energy?"]


def test_cap_ends_the_first_clause_at_a_word_boundary():
    out = chunk(NO_COMMA, first_clause_max=30)
    assert out == ["Sure! Did you know that the brain", "uses about twenty percent of your energy?"]


def test_cap_never_splits_inside_a_word():
    """LLM tokens are sub-word pieces; only a token that STARTS with a space marks a
    finished word, so 'Extraordinary' must stay whole even though the cap (5) is
    reached after its first piece."""
    out = chunk(["Ext", "ra", "ord", "inary", " claims", " need", " evidence", "."],
                first_clause_max=5)
    assert out == ["Extraordinary", "claims need evidence."]


def test_punctuation_before_the_cap_still_wins():
    out = chunk(GREETING, first_clause_min=18, first_clause_max=40)
    assert out == ["Hello! I'm Amadeus,", "glad to help."]


def test_cap_applies_to_the_first_clause_only():
    tokens = ["Hello", "!"] + [" I'm", " Amadeus", " and", " I", " will", " happily", " explain",
                              " how", " every", " single", " part", " of", " this", " works", "."]
    out = chunk(tokens, first_clause_min=0, first_clause_max=10)
    assert out[0] == "Hello!"
    assert len(out) == 2 and out[1].startswith("I'm Amadeus") and out[1].endswith("works.")


def test_cap_longer_than_the_reply_changes_nothing():
    assert chunk(["Hi", " there", "."], first_clause_max=30) == ["Hi there."]


# ----------------------------------------------------------------- cap seam placement (function words)

def words_as_tokens(text):
    """Split like an LLM stream: every word after the first carries a leading space."""
    ws = text.split(" ")
    return [w if i == 0 else " " + w for i, w in enumerate(ws)]


def test_cap_cuts_before_a_dangling_function_word():
    out = chunk(words_as_tokens("Coffee is a beloved stimulant that energizes many people!"),
                first_clause_max=30)
    assert out == ["Coffee is a beloved stimulant", "that energizes many people!"]


def test_cap_moves_a_run_of_function_words_together():
    out = chunk(words_as_tokens("Let me tell you all about the amazing things."), first_clause_max=24)
    assert out == ["Let me tell you all", "about the amazing things."]


def test_cap_backs_up_no_further_than_the_floor():
    """All function words: the opener must not collapse to almost nothing, or clause 2
    would not be ready when clause 1 finishes."""
    text = "It is a an the of to in on at for by"
    out = chunk(words_as_tokens(text) + ["."], first_clause_max=30)
    assert len(out[0]) >= 15


@pytest.mark.parametrize("cap", [12, 20, 30, 45])
@pytest.mark.parametrize("minimum", [None, 0, 16])
def test_cap_never_loses_or_duplicates_text(minimum, cap):
    replies = [
        "Sure! Did you know that the brain uses about twenty percent of your energy?",
        "As an AI, I don't have personal preferences or emotions, so I don't have a favorite color.",
        "Good morning! I'm as active and curious as ever. How about you? Everything okay?",
        "The sky appears blue due to a phenomenon called Rayleigh scattering and it is lovely.",
        "It is a an the of to in on at for by with.",
    ]
    for reply in replies:
        out = chunk(words_as_tokens(reply), first_clause_min=minimum, first_clause_max=cap)
        assert " ".join(out) == reply, (minimum, cap, out)
