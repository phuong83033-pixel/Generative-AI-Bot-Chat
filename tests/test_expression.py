"""ExpressionModel (emotion/state/speech -> face) and AvatarDriver (the frame loop)."""

import asyncio
import random

import pytest

from amadeus.bus import EventBus
from amadeus.emotion.engine import EmotionEngine
from amadeus.emotion.pad import PAD
from amadeus.config import EmotionConfig
from amadeus.expression.driver import AvatarDriver
from amadeus.expression.frame import MOUTH_FULL_LEVEL, ExpressionFrame, ExpressionModel
from amadeus.state import SystemState

BASE = PAD(0.05, 0.05, 0.25)


def settled(model, seconds=3.0, dt=1 / 30):
    f = None
    for _ in range(int(seconds / dt)):
        f = model.step(dt)
    return f


def with_emotion(label, delta, intensity=0.7, seed=1):
    """A model that has settled into the face an event with this delta + label produces."""
    m = ExpressionModel(random.Random(seed))
    pad = BASE + delta
    m.set_emotion(pad, BASE, label, intensity)
    return m, settled(m)


# ------------------------------------------------------------------ the faces

def test_resting_face_is_neutral():
    m = ExpressionModel(random.Random(1))
    m.set_emotion(BASE, BASE, "neutral", 0.0)
    f = settled(m, 4)
    assert abs(f.mouth_smile) < 0.02 and f.blush == 0.0 and abs(f.brow_inner) < 0.02
    assert f.mouth_open == 0.0
    assert f.label == "neutral" and f.state == "IDLE"


def test_flustered_blushes_raises_brows_and_looks_away():
    _, f = with_emotion("flustered", PAD(0.2, 0.4, -0.5))
    assert f.blush > 0.4
    assert f.brow_raise > 0.2
    assert f.gaze_x > 0.15          # too embarrassed to meet your eyes
    assert f.head_pitch < 0         # chin down


def test_annoyed_furrows_frowns_and_narrows_the_eyes():
    rest, _ = with_emotion("neutral", PAD(0, 0, 0), 0.0)
    f = settled(rest)
    _, a = with_emotion("annoyed", PAD(-0.35, 0.5, 0.3))
    assert a.brow_inner < -0.2 and a.mouth_smile < -0.2
    assert a.eye_open < f.eye_open or a.eye_open <= 0.9


def test_concerned_lifts_the_inner_brows_not_furrows_them():
    """Regression, found by looking at the rendered avatar: 'concerned' came out furrowed because
    her resting dominance is high, so dominance alone read it as anger."""
    _, f = with_emotion("concerned", PAD(-0.25, 0.10, 0.15))
    assert f.brow_inner > 0.1
    _, sad = with_emotion("sad", PAD(-0.5, -0.3, -0.3))
    assert sad.brow_inner > 0.1


def test_angry_is_furrowed_even_with_low_dominance():
    _, f = with_emotion("angry", PAD(-0.55, 0.65, -0.2))
    assert f.brow_inner < -0.2


def test_happy_smiles():
    _, f = with_emotion("happy", PAD(0.35, 0.2, 0.0))
    assert f.mouth_smile > 0.3


@pytest.mark.parametrize("state, expect", [
    ("THINKING", lambda f: f.gaze_x < -0.2 and f.gaze_y > 0.2),   # looks up and away
    ("LISTENING", lambda f: f.head_roll > 1.5 and f.brow_raise > 0.05),
    ("IDLE", lambda f: abs(f.gaze_x) < 0.2),
])
def test_system_state_changes_what_the_face_does(state, expect):
    m = ExpressionModel(random.Random(3))
    m.set_emotion(BASE, BASE, "neutral", 0.0)
    m.set_state(state)
    f = settled(m)
    assert f.state == state and expect(f)


# ------------------------------------------------------------------- lip-sync

def test_mouth_opens_fast_with_speech_and_closes_when_it_stops():
    m = ExpressionModel(random.Random(1))
    m.set_level(MOUTH_FULL_LEVEL)
    assert settled(m, 0.2).mouth_open > 0.9
    m.set_level(0.0)
    assert settled(m, 0.5).mouth_open < 0.05


def test_mouth_is_proportional_to_loudness():
    opens = []
    for level in (0.0, 0.02, 0.06, 0.12, 0.3):
        m = ExpressionModel(random.Random(1))
        m.set_level(level)
        opens.append(settled(m, 0.5).mouth_open)
    assert opens == sorted(opens) and opens[0] == 0.0 and opens[-1] == 1.0


def test_mouth_attack_is_faster_than_release():
    """Lip-sync must not lag the sound that just started, but may ease off after it."""
    up = ExpressionModel(random.Random(1)); up.set_level(MOUTH_FULL_LEVEL)
    t_open = next(i for i in range(200) if up.step(1 / 60).mouth_open > 0.8)
    down = ExpressionModel(random.Random(1)); down.set_level(MOUTH_FULL_LEVEL); settled(down, 0.5)
    down.set_level(0.0)
    t_close = next(i for i in range(200) if down.step(1 / 60).mouth_open < 0.2)
    assert t_open < t_close


# ----------------------------------------------------------- motion and blink

def test_she_blinks_and_the_eyes_come_back():
    m = ExpressionModel(random.Random(7))
    m.set_emotion(BASE, BASE, "neutral", 0.0)
    eyes = [m.step(1 / 30).eye_open for _ in range(30 * 20)]
    blinks = sum(1 for a, b in zip(eyes, eyes[1:]) if a >= 0.2 > b)
    assert 3 <= blinks <= 9, f"{blinks} blinks in 20 s"
    assert eyes[-1] > 0.7 or max(eyes[-5:]) > 0.7


def test_idle_head_moves_a_little_but_not_a_lot():
    m = ExpressionModel(random.Random(1))
    frames = [m.step(1 / 30) for _ in range(30 * 20)]
    yaws = [f.head_yaw for f in frames]
    assert 1.0 < max(yaws) - min(yaws) < 8.0


def test_no_jump_when_the_mood_switches():
    """Smoothing: a sudden change of emotion must glide, not snap."""
    m = ExpressionModel(random.Random(1))
    m.set_emotion(BASE, BASE, "neutral", 0.0)
    settled(m)
    m.set_emotion(BASE + PAD(0.35, 0.2, 0.0), BASE, "happy", 0.7)
    smiles = [m.step(1 / 30).mouth_smile for _ in range(60)]
    steps = [b - a for a, b in zip(smiles, smiles[1:])]
    assert max(steps) < 0.15                         # no snap
    assert all(s >= -1e-9 for s in steps)            # and no overshoot / wobble
    assert smiles[-1] > smiles[0] + 0.3              # but it does get there


def test_a_stalled_loop_does_not_make_the_face_jump():
    a = ExpressionModel(random.Random(1)); b = ExpressionModel(random.Random(1))
    for m in (a, b):
        m.set_emotion(BASE + PAD(0.35, 0.2, 0.0), BASE, "happy", 0.7)
    assert a.step(10.0).mouth_smile == b.step(0.25).mouth_smile   # a long stall counts as at most 0.25 s


def test_every_value_stays_in_its_documented_range():
    rng = random.Random(0)
    m = ExpressionModel(random.Random(5))
    for _ in range(600):
        m.set_emotion(PAD(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)), BASE,
                      rng.choice(["flustered", "annoyed", "concerned", "happy", "neutral", "sad"]), rng.random())
        m.set_state(rng.choice(["IDLE", "LISTENING", "THINKING", "SPEAKING"]))
        m.set_level(rng.choice([0.0, 0.05, 0.12, 0.5, 2.0]))
        f = m.step(rng.choice([1 / 30, 1 / 60, 0.1]))
        assert 0.0 <= f.mouth_open <= 1.0 and 0.0 <= f.eye_open <= 1.0 and 0.0 <= f.blush <= 1.0
        for v in (f.mouth_smile, f.brow_raise, f.brow_inner, f.gaze_x, f.gaze_y):
            assert -1.0 <= v <= 1.0
        for v in (f.head_yaw, f.head_pitch, f.head_roll):
            assert abs(v) < 30.0


def test_the_same_seed_gives_the_same_animation():
    def run(seed):
        m = ExpressionModel(random.Random(seed)); m.set_level(0.1)
        return [m.step(1 / 30).to_dict() for _ in range(120)]
    assert run(4) == run(4) and run(4) != run(5)


# ----------------------------------------------------------------------- driver

class Sink:
    def __init__(self, fail=False):
        self.frames, self.closed, self.fail = [], False, fail

    async def send(self, frame):
        if self.fail:
            raise RuntimeError("renderer exploded")
        self.frames.append(frame)

    async def close(self):
        self.closed = True


def test_driver_turns_bus_events_into_frames_without_touching_cognition():
    async def go():
        bus = EventBus(); bus.bind_loop()
        sink = Sink()
        d = AvatarDriver(bus, [sink], level_source=lambda: 0.0)
        bus.subscribe  # (the driver subscribes itself in start())
        await d.start()
        engine = EmotionEngine(EmotionConfig(), bus=bus)
        await engine.on_user_utterance("You're really smart.")          # -> EmotionChanged
        await bus.publish("StateChanged", {"old": SystemState.IDLE, "new": SystemState.SPEAKING})
        await asyncio.sleep(0.5)
        await d.stop()
        return sink

    sink = asyncio.run(go())
    last = sink.frames[-1]
    assert last.label == "flustered" and last.state == "SPEAKING" and last.blush > 0.2
    assert sink.closed


def test_driver_reads_the_speaker_level_for_lipsync():
    async def go():
        bus = EventBus(); bus.bind_loop()
        sink, level = Sink(), {"v": 0.0}
        d = AvatarDriver(bus, [sink], level_source=lambda: level["v"], fps=60)
        await d.start()
        await asyncio.sleep(0.15); quiet = sink.frames[-1].mouth_open
        level["v"] = 0.12
        await asyncio.sleep(0.25); talking = sink.frames[-1].mouth_open
        await d.stop()
        return quiet, talking

    quiet, talking = asyncio.run(go())
    assert quiet < 0.05 and talking > 0.8


def test_a_failing_sink_neither_stops_the_loop_nor_starves_the_others():
    async def go():
        bus = EventBus(); bus.bind_loop()
        bad, good = Sink(fail=True), Sink()
        d = AvatarDriver(bus, [bad, good], fps=100)
        await d.start()
        await asyncio.sleep(0.3)
        await d.stop()
        return len(good.frames)

    assert asyncio.run(go()) >= 15


def test_driver_holds_its_frame_rate_and_stays_cheap():
    async def go():
        bus = EventBus(); bus.bind_loop()
        sink = Sink()
        d = AvatarDriver(bus, [sink], fps=30)
        await d.start()
        await asyncio.sleep(1.0)
        await d.stop()
        return len(sink.frames), d.max_tick_ms

    frames, worst = asyncio.run(go())
    assert 22 <= frames <= 36
    assert worst < 20.0   # one tick never costs a meaningful slice of the 33 ms frame


def test_stop_unsubscribes_from_the_bus():
    async def go():
        bus = EventBus(); bus.bind_loop()
        d = AvatarDriver(bus, [Sink()])
        await d.start(); await d.stop()
        before = d.model._in.state
        await bus.publish("StateChanged", {"old": SystemState.IDLE, "new": SystemState.THINKING})
        return before, d.model._in.state

    before, after = asyncio.run(go())
    assert before == after == "IDLE"


def test_frame_serialises_to_plain_json_types():
    import json
    d = ExpressionFrame().to_dict()
    assert json.loads(json.dumps(d)) == d


# ------------------------------------------- the face must agree with the label

def _drive(engine, model, clock, seconds, fps=30):
    """Feed the model the engine's current mood every frame, the way AvatarDriver does."""
    f = None
    for _ in range(int(seconds * fps)):
        clock.t += 1 / fps
        snap = engine.snapshot()
        model.set_emotion(snap.pad, snap.baseline, snap.label, snap.intensity)
        f = model.step(1 / fps)
    return f


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_annoyed_after_flustered_looks_annoyed_not_embarrassed():
    """Regression, seen in the live recording: 14 s after a compliment, 'Hey Christina' gave the
    label 'annoyed 0.78' but a face with blush 0.57, a smile, and barely furrowed brows, because
    the face followed the accumulated PAD (still holding the compliment) while the label followed
    the event."""
    clock = _Clock()
    engine = EmotionEngine(EmotionConfig(), clock=clock)
    model = ExpressionModel(random.Random(1))

    asyncio.run(engine.on_user_utterance("You're really smart."))
    flustered = _drive(engine, model, clock, 14)
    assert flustered.label == "flustered" and flustered.blush > 0.4          # the compliment did land

    snap = asyncio.run(engine.on_user_utterance("Hey Christina."))
    assert snap.label == "annoyed"
    f = _drive(engine, model, clock, 3)
    assert f.brow_inner < -0.3, f"brows not furrowed: {f.brow_inner}"
    assert f.mouth_smile < -0.15, f"still smiling: {f.mouth_smile}"
    assert f.blush < 0.35, f"still blushing: {f.blush}"


def test_flustered_after_annoyed_blushes_again():
    clock = _Clock()
    engine = EmotionEngine(EmotionConfig(), clock=clock)
    model = ExpressionModel(random.Random(1))
    asyncio.run(engine.on_user_utterance("Hey Christina."))
    _drive(engine, model, clock, 14)
    asyncio.run(engine.on_user_utterance("I love you."))
    f = _drive(engine, model, clock, 3)
    assert f.label == "flustered" and f.blush > 0.4 and f.brow_inner > -0.2


def test_every_label_has_an_expression_preset():
    """A label with no preset silently falls back to raw PAD, which is the bug above."""
    from amadeus.emotion.labels import LABELS
    from amadeus.expression.frame import _PRESETS
    assert set(LABELS) <= set(_PRESETS)
