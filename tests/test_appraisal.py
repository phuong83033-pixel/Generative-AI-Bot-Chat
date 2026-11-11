"""Appraisal: user text -> ΔPAD. Rules are data, so every rule needs a sentence that fires it."""

import pytest

from amadeus.emotion.appraisal import _RULES, appraise
from amadeus.emotion.pad import PAD, PADState

BASELINE = PAD(0.05, 0.05, 0.25)

# (sentence, the tag that must fire, the label she ends up with from baseline)
FIRES = [
    ("You're really smart.", "praise", "flustered"),
    ("Good job on that.", "praise", "flustered"),
    ("I love you.", "affection", "flustered"),
    ("You're so cute.", "affection", "flustered"),
    ("You're stupid.", "insult", "annoyed"),
    ("Shut up!", "hostility", "angry"),
    ("I hate you.", "hostility", "angry"),
    ("I don't like you.", "dislike", "annoyed"),
    ("Hey Christina.", "nickname_christina", "annoyed"),
    ("Thanks a lot.", "gratitude", "content"),
    ("Sorry about that.", "apology", "content"),
    ("I'm so sad today.", "user_distress", "concerned"),
    ("I had a bad day.", "user_distress", "concerned"),
    ("You're wrong about that.", "challenge", "annoyed"),
    ("What do you think about time travel?", "science_topic", "curious"),
    ("Do you like Dr Pepper?", "favorite_drink", "happy"),
    ("haha that's funny", "humor", "happy"),
    ("Hello there", "greeting", "neutral"),     # a greeting is too small to be a feeling
    ("Goodbye", "farewell", "neutral"),
]


def label_after(text):
    state = PADState(BASELINE, PAD(90, 45, 120), clock=lambda: 0.0)
    state.apply(appraise(text).delta)
    return state.label().name


@pytest.mark.parametrize("text, tag, label", FIRES)
def test_rule_fires_and_gives_the_expected_feeling(text, tag, label):
    a = appraise(text)
    assert tag in a.tags
    assert label_after(text) == label


def test_every_rule_is_exercised_by_a_test_sentence():
    """A rule nobody tests is a rule that can silently stop matching."""
    covered = {tag for _, tag, _ in FIRES}
    assert {r.tag for r in _RULES} <= covered


# ------------------------------------------------------------------- negation

@pytest.mark.parametrize("text", [
    "You're not stupid.",
    "You aren't stupid.",
    "I don't think you're stupid.",
    "I don't hate you.",
    "I'm not sad.",
])
def test_negated_bad_things_change_nothing(text):
    a = appraise(text)
    assert a.delta == PAD()
    assert any("ignored" in t for t in a.tags)  # and the reason is on record


def test_negated_praise_stings_instead_of_flattering():
    a = appraise("You're not smart.")
    assert "praise(negated)" in a.tags
    assert a.delta.pleasure < 0 and a.delta.dominance >= 0
    assert label_after("You're not smart.") == "annoyed"


def test_cant_believe_is_not_a_negation():
    """'I can't believe you're so smart' is admiration."""
    assert appraise("I can't believe you're so smart.").tags == ("praise",)


# ------------------------------------------------------------- everything else

@pytest.mark.parametrize("text", ["", "   ", "What's the weather like?", "Okay.", "Tell me a story."])
def test_unrecognised_input_changes_nothing(text):
    a = appraise(text)
    assert a.delta == PAD() and a.tags == () and not a


def test_shouting_raises_arousal_not_pleasure():
    calm, loud = appraise("You are wrong."), appraise("YOU ARE WRONG!")
    assert loud.delta.arousal > calm.delta.arousal
    assert loud.delta.pleasure == calm.delta.pleasure
    assert "emphasis" in loud.tags and "emphasis" not in calm.tags


def test_emphasis_alone_is_not_an_emotion():
    assert appraise("What time is it!").delta == PAD()


def test_strength_scales_every_delta_and_zero_disables():
    full, half, off = appraise("You're stupid.", 1.0), appraise("You're stupid.", 0.5), appraise("You're stupid.", 0.0)
    assert half.delta.pleasure == pytest.approx(full.delta.pleasure / 2)
    assert off.delta == PAD()


def test_one_utterance_cannot_swing_an_axis_past_the_cap():
    a = appraise("You're brilliant and I love you, you're so cute! Good job! Well done!", strength=5.0)
    for v in a.delta.as_tuple():
        assert abs(v) <= 0.8 + 1e-9


def test_mixed_signals_add_up():
    a = appraise("Thanks, but you're wrong.")
    assert {"gratitude", "challenge"} <= set(a.tags)


def test_curly_apostrophes_from_stt_are_understood():
    assert "insult" in appraise("You’re stupid.").tags
