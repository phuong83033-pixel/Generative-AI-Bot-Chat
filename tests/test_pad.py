"""PAD state: arithmetic, decay toward the baseline, and the label read-out (README section 15)."""

import math

import pytest

from amadeus.emotion.labels import LABELS, NEUTRAL_RADIUS, classify, hint_for, EmotionLabel
from amadeus.emotion.pad import PAD, PADState

BASELINE = PAD(0.05, 0.05, 0.25)
HALF_LIVES = PAD(90.0, 45.0, 120.0)
FLUSTERED_DELTA = PAD(0.20, 0.40, -0.50)  # what one compliment does (appraisal.py)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def make_state(clock=None, **kw):
    clock = clock or Clock()
    return PADState(baseline=BASELINE, half_life_s=HALF_LIVES, clock=clock, **kw), clock


# ------------------------------------------------------------------ arithmetic

def test_pad_add_scale_and_clamp():
    a, b = PAD(0.5, -0.5, 0.2), PAD(0.7, -0.7, 0.1)
    s = a + b
    assert (s.pleasure, s.arousal) == pytest.approx((1.2, -1.2))
    assert s.clamped().pleasure == 1.0 and s.clamped().arousal == -1.0
    assert a.scaled(2).pleasure == 1.0


def test_pad_distance_is_euclidean():
    assert PAD(0, 0, 0).distance(PAD(0.3, 0.4, 0.0)) == pytest.approx(0.5)


def test_state_starts_at_baseline_and_is_neutral():
    state, _ = make_state()
    assert state.current == BASELINE
    assert state.label().name == "neutral"


# ----------------------------------------------------------------------- decay

def test_distance_to_baseline_halves_every_half_life():
    """The definition of half-life, checked per axis against the exact formula."""
    state, clock = make_state()
    state.apply(FLUSTERED_DELTA)
    start = state.current
    clock.advance(45.0)  # one arousal half-life
    state.tick()
    now = state.current
    assert now.arousal - BASELINE.arousal == pytest.approx((start.arousal - BASELINE.arousal) / 2)
    # Pleasure (90 s) has only gone 0.5 of a half-life: factor 0.5 ** 0.5
    assert now.pleasure - BASELINE.pleasure == pytest.approx(
        (start.pleasure - BASELINE.pleasure) * 0.5 ** 0.5
    )


def test_state_always_returns_to_baseline():
    state, clock = make_state()
    state.apply(PAD(0.8, 0.8, -0.8))
    clock.advance(3600)
    assert state.label().name == "neutral"
    assert state.current.distance(BASELINE) < 1e-6


def test_apply_decays_first_so_an_old_mood_is_not_revived():
    """Anger at t=0, then 10 half-lives later a compliment: only the compliment is left."""
    state, clock = make_state()
    state.apply(PAD(-0.8, 0.8, 0.5))
    clock.advance(900)
    state.apply(FLUSTERED_DELTA)
    fresh, _ = make_state()
    fresh.apply(FLUSTERED_DELTA)  # the same compliment, from a calm start
    assert state.current.distance(fresh.current) < 0.01


def test_values_never_leave_minus_one_to_one():
    state, _ = make_state()
    for _ in range(50):
        state.apply(PAD(0.9, 0.9, -0.9))
    for v in state.current.as_tuple():
        assert -1.0 <= v <= 1.0
    assert state.current.dominance < -0.99 and state.current.arousal > 0.99  # but it does get there


def test_repeated_events_have_diminishing_returns():
    """Regression: two arousing events used to pin arousal at +1.0, after which every
    mood looked 'curious'. Each push must be smaller than the last."""
    state, _ = make_state()
    gains, prev = [], state.current.arousal
    for _ in range(6):
        state.apply(PAD(0.0, 0.4, 0.0))
        gains.append(state.current.arousal - prev)
        prev = state.current.arousal
    assert all(b < a for a, b in zip(gains, gains[1:]))
    assert state.current.arousal < 1.0


def test_an_off_centre_baseline_never_enlarges_a_delta():
    """Dominance rests at +0.25. A -0.5 delta has 1.25 of room below it; it must still be
    exactly -0.5, not -0.625."""
    state, _ = make_state()
    state.apply(PAD(0.0, 0.0, -0.5))
    assert state.current.dominance == pytest.approx(BASELINE.dominance - 0.5)


def test_zero_half_life_snaps_back_and_infinite_never_decays():
    clock = Clock()
    snap = PADState(BASELINE, PAD(0.0, 0.0, 0.0), clock)
    snap.apply(FLUSTERED_DELTA)
    clock.advance(1)
    snap.tick()
    assert snap.current == BASELINE

    clock2 = Clock()
    stuck = PADState(BASELINE, PAD(math.inf, math.inf, math.inf), clock2)
    stuck.apply(FLUSTERED_DELTA)
    before = stuck.current
    clock2.advance(10_000)
    stuck.tick()
    assert stuck.current == before


def test_tick_with_no_time_elapsed_is_a_noop():
    state, _ = make_state()
    state.apply(FLUSTERED_DELTA)
    assert state.tick() is False


# ---------------------------------------------------------------------- labels

def test_a_compliment_makes_her_flustered():
    state, _ = make_state()
    state.apply(FLUSTERED_DELTA)
    assert state.label().name == "flustered"


@pytest.mark.parametrize("delta, expected", [
    (PAD(0.20, 0.40, -0.50), "flustered"),
    (PAD(-0.40, 0.45, 0.25), "annoyed"),
    (PAD(-0.55, 0.65, 0.30), "angry"),
    (PAD(0.12, 0.30, 0.10), "curious"),
    (PAD(0.30, 0.20, 0.05), "happy"),
    (PAD(-0.25, 0.10, 0.15), "concerned"),
])
def test_label_for_a_single_event(delta, expected):
    state, _ = make_state()
    state.apply(delta)
    assert state.label().name == expected


def test_label_identity_is_frozen_while_fading():
    """Regression: a fading 'flustered' used to pass through 'curious' and 'concerned'
    because the straight-line path to the baseline is near those prototypes."""
    state, clock = make_state()
    state.apply(FLUSTERED_DELTA)
    seen = []
    for _ in range(40):
        clock.advance(5)
        lab = state.label()
        seen.append(lab.name)
    non_neutral = {n for n in seen if n != "neutral"}
    assert non_neutral == {"flustered"}
    assert seen[-1] == "neutral"


def test_intensity_only_falls_while_fading():
    state, clock = make_state()
    state.apply(FLUSTERED_DELTA)
    last = state.label().intensity
    for _ in range(30):
        clock.advance(5)
        now = state.label().intensity
        assert now <= last + 1e-9
        last = now


def test_a_new_event_reclassifies_freely():
    state, clock = make_state()
    state.apply(FLUSTERED_DELTA)
    assert state.label().name == "flustered"
    clock.advance(1)
    state.apply(PAD(-0.9, 0.2, 0.9))  # insult lands on top of the flustered state
    assert state.label().name != "flustered"


ANNOYED_DELTA = PAD(-0.35, 0.50, 0.30)   # "Hey Christina."
SCIENCE_DELTA = PAD(0.12, 0.30, 0.10)    # "...time travel?"


def test_flustered_then_a_science_question_then_christina_ends_annoyed():
    """The live bug. Accumulated state after these three events sat nearest 'curious'
    (arousal pinned), so 'Hey Christina' made her curious. The label now follows the event."""
    state, clock = make_state()
    for delta in (FLUSTERED_DELTA, SCIENCE_DELTA, ANNOYED_DELTA):
        clock.advance(20)
        state.apply(delta)
    assert state.label().name == "annoyed"


def test_a_weaker_event_adds_to_the_mood_without_renaming_it():
    state, clock = make_state()
    state.apply(FLUSTERED_DELTA)
    before = state.label().intensity
    clock.advance(1)
    state.apply(SCIENCE_DELTA)  # |Δ| 0.34 vs a mood of ~0.67 in force: below the 60% takeover
    after = state.label()
    assert after.name == "flustered"
    assert after.intensity != before  # but it did move the state


def test_a_stronger_event_renames_the_mood():
    state, clock = make_state()
    state.apply(FLUSTERED_DELTA)
    clock.advance(1)
    state.apply(ANNOYED_DELTA)
    assert state.label().name == "annoyed"


def test_a_faint_event_does_not_name_a_mood_at_all():
    state, _ = make_state()
    state.apply(PAD(0.10, 0.10, 0.0))  # a greeting
    assert state.label().name == "neutral"


def test_neutral_radius_boundary():
    inside = BASELINE + PAD(NEUTRAL_RADIUS * 0.9, 0, 0)
    outside = BASELINE + PAD(NEUTRAL_RADIUS * 1.1, 0, 0)
    assert classify(inside, BASELINE).name == "neutral"
    assert classify(outside, BASELINE).name != "neutral"


def test_every_label_has_a_prompt_hint_and_neutral_has_none():
    for name in LABELS:
        text = hint_for(EmotionLabel(name, 0.6))
        assert text and "[Mood]" in text and "Do not name the feeling" in text
    assert hint_for(EmotionLabel("neutral", 0.0)) is None


def test_hint_degree_tracks_intensity():
    assert "slightly" in hint_for(EmotionLabel("flustered", 0.3))
    assert "very" in hint_for(EmotionLabel("flustered", 0.9))
    mid = hint_for(EmotionLabel("flustered", 0.6))
    assert "slightly" not in mid and "very" not in mid
