"""EmotionEngine and how it reaches the prompt, the voice and the bus."""

import asyncio
from pathlib import Path

import pytest
import yaml

from amadeus.bus import EventBus
from amadeus.cognition.prompt import PromptBuilder
from amadeus.config import EmotionConfig, load_config
from amadeus.emotion.engine import EmotionEngine, EmotionSnapshot
from amadeus.emotion.pad import PAD
from amadeus.synthesis.base import TtsStyle
from test_respond_pipeline import FakeBrain, FakeTTS, make_app, respond


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def engine(bus=None, clock=None, **overrides):
    cfg = EmotionConfig(**overrides) if overrides else EmotionConfig()
    return EmotionEngine(cfg, bus=bus, clock=clock or Clock())


# ------------------------------------------------------- the neutral guarantee

def test_neutral_means_no_hint_and_exactly_normal_speed():
    """Until she actually feels something, behaviour is identical to before M9."""
    e = engine()
    assert e.prompt_hint() is None
    assert e.tts_style() == TtsStyle(speed=1.0)
    assert e.snapshot().label == "neutral"


def test_unrecognised_speech_leaves_her_neutral():
    e = engine()
    asyncio.run(e.on_user_utterance("What's the weather like?"))
    assert e.prompt_hint() is None and e.tts_style().speed == 1.0


# ---------------------------------------------------------------------- prompt

def test_a_compliment_produces_a_flustered_hint_and_a_faster_voice():
    e = engine()
    snap = asyncio.run(e.on_user_utterance("You're really smart."))
    assert snap.label == "flustered" and snap.reasons == ("praise",)
    hint = e.prompt_hint()
    assert hint and "flustered" in hint and "Do not name the feeling" in hint
    assert e.tts_style().speed > 1.0  # arousal went up


def test_a_mood_too_faint_for_the_threshold_adds_nothing():
    e = engine(hint_min_intensity=0.99)
    asyncio.run(e.on_user_utterance("You're really smart."))
    assert e.prompt_hint() is None


def test_the_hint_goes_away_as_the_mood_fades():
    clock = Clock()
    e = engine(clock=clock)
    asyncio.run(e.on_user_utterance("You're really smart."))
    assert e.prompt_hint()
    clock.t += 600
    assert e.prompt_hint() is None and e.tts_style().speed == pytest.approx(1.0, abs=1e-3)


def test_prompt_builder_puts_the_hint_before_the_last_user_message_only():
    pb = PromptBuilder("SYSTEM")
    pb.add_user_message("earlier question")
    pb.add_assistant_message("earlier answer")
    plain = pb.build("now")
    moody = pb.build("now", mood_hint="[Mood] hint")

    assert plain == [m for m in moody if m["content"] != "[Mood] hint"]  # nothing else moved
    assert moody[-2] == {"role": "system", "content": "[Mood] hint"}
    assert moody[-1] == {"role": "user", "content": "now"}
    assert moody[0] == {"role": "system", "content": "SYSTEM"}  # stable prefix untouched


def test_the_hint_is_never_stored_in_history():
    pb = PromptBuilder("SYSTEM")
    pb.build("hello", mood_hint="[Mood] hint")
    pb.add_user_message("hello")
    pb.add_assistant_message("hi")
    assert all(m["content"] != "[Mood] hint" for m in pb.build("next"))


# ----------------------------------------------------------------------- voice

@pytest.mark.parametrize("delta, direction", [
    (PAD(0.0, 0.6, 0.0), "faster"),     # agitated
    (PAD(-0.5, -0.6, 0.0), "slower"),   # low and flat
])
def test_voice_speed_follows_arousal_and_pleasure(delta, direction):
    e = engine()
    e.state.apply(delta)
    s = e.tts_style().speed
    assert (s > 1.0) if direction == "faster" else (s < 1.0)


def test_voice_speed_is_clamped():
    e = engine()
    for _ in range(10):
        e.state.apply(PAD(1.0, 1.0, 0.0))
    assert e.tts_style().speed == pytest.approx(EmotionConfig().speed_max)
    e2 = engine()
    for _ in range(10):
        e2.state.apply(PAD(-1.0, -1.0, 0.0))
    assert e2.tts_style().speed == pytest.approx(EmotionConfig().speed_min)


# ------------------------------------------------------------------------- bus

def test_utterance_publishes_emotion_changed_with_a_snapshot():
    async def go():
        bus = EventBus()
        bus.bind_loop()
        seen = []
        bus.subscribe("EmotionChanged", lambda s: seen.append(s))
        e = engine(bus=bus)
        await e.on_user_utterance("Hey Christina.")
        return seen

    (snap,) = asyncio.run(go())
    assert isinstance(snap, EmotionSnapshot)
    assert snap.label == "annoyed" and "nickname_christina" in snap.reasons


def test_ticker_publishes_while_fading_then_stops_when_settled_and_exits_on_stop():
    async def go():
        clock = Clock()
        bus = EventBus()
        bus.bind_loop()
        seen = []
        bus.subscribe("EmotionChanged", lambda s: seen.append(s))
        e = engine(bus=bus, clock=clock, tick_interval_s=0.01)
        await e.on_user_utterance("You're really smart.")
        n_after_event = len(seen)

        stop = asyncio.Event()
        task = asyncio.create_task(e.run_ticker(stop))
        clock.t += 60                       # time passes: the mood fades
        await asyncio.sleep(0.05)
        n_fading = len(seen)
        faded_pad = seen[-1].pad

        clock.t += 100_000                  # fully settled, then nothing more to say
        await asyncio.sleep(0.05)
        n_settled = len(seen)
        await asyncio.sleep(0.05)
        n_quiet = len(seen)

        stop.set()
        await asyncio.wait_for(task, 1.0)   # exits promptly
        return n_after_event, n_fading, faded_pad, n_settled, n_quiet, seen[-1]

    n0, n1, faded, n2, n3, last = asyncio.run(go())
    assert n1 > n0, "avatar must be told the mood is fading"
    assert faded.distance(PAD(0.05, 0.05, 0.25)) < PAD(0.25, 0.45, -0.25).distance(PAD(0.05, 0.05, 0.25))
    assert n3 == n2, "a settled mood must not be re-published forever"
    assert last.label == "neutral"


# ---------------------------------------------------------------------- config

def test_shipped_emotion_yaml_matches_the_dataclass_defaults():
    """If someone edits one and not the other, a missing file would behave differently."""
    shipped = load_config().emotion
    assert shipped == EmotionConfig()


def test_missing_emotion_yaml_falls_back_to_defaults(tmp_path):
    root = Path(__file__).resolve().parent.parent / "config"
    for name in ("settings.yaml", "persona.yaml"):
        (tmp_path / name).write_text((root / name).read_text(encoding="utf-8"), encoding="utf-8")
    assert load_config(tmp_path).emotion == EmotionConfig()


def test_partial_emotion_yaml_overrides_only_what_it_names(tmp_path):
    root = Path(__file__).resolve().parent.parent / "config"
    for name in ("settings.yaml", "persona.yaml"):
        (tmp_path / name).write_text((root / name).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "emotion.yaml").write_text(yaml.safe_dump({"appraisal_strength": 2.0}), encoding="utf-8")
    cfg = load_config(tmp_path).emotion
    assert cfg.appraisal_strength == 2.0
    assert cfg.baseline == EmotionConfig().baseline


# --------------------------------------------------- through the real _respond

def test_respond_hands_the_mood_to_the_prompt_and_the_voice():
    tts = FakeTTS(synth_s=0.0, audio_s=0.05)
    app = make_app(FakeBrain("Hmph. Fine, thanks."), tts)
    app.emotion = engine()
    app.emotion.state.apply(PAD(0.20, 0.40, -0.50))      # as if she had just been complimented

    respond(app)
    assert "flustered" in app.prompt.last_mood_hint
    assert tts.styles and all(s is not None and s.speed > 1.0 for s in tts.styles)


def test_respond_without_emotion_is_unchanged():
    tts = FakeTTS(synth_s=0.0, audio_s=0.05)
    app = make_app(FakeBrain("Hmph. Fine, thanks."), tts)
    assert app.emotion is None
    respond(app)
    assert app.prompt.last_mood_hint is None
    assert all(s is None for s in tts.styles)


def test_kokoro_scales_its_configured_speed_by_the_style():
    """KokoroTTS.synth multiplies, and clamps to Kokoro's accepted 0.5-2.0."""
    from amadeus.synthesis.kokoro import KokoroTTS

    seen = []

    class FakeKokoro:
        def create(self, text, voice, speed, lang):
            seen.append(speed)
            import numpy as np
            return np.zeros(100, dtype=np.float32), 24000

    k = KokoroTTS.__new__(KokoroTTS)            # skip loading the model
    k._kokoro, k._voice, k._speed, k._lang = FakeKokoro(), "af_heart", 1.1, "en-us"
    from amadeus.trace import get_tracer
    k._tracer = get_tracer("test")

    async def go():
        await k.synth("Hello there.")
        await k.synth("Hello there.", style=TtsStyle(speed=1.1))
        await k.synth("Hello there.", style=TtsStyle(speed=5.0))

    asyncio.run(go())
    assert seen == [pytest.approx(1.1), pytest.approx(1.21), 2.0]
