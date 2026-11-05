"""Tests for the pipelined Amadeus._respond.

The claim under test: synthesis of clause N+1 overlaps playback of clause N, so the
listener hears no silence between clauses unless synthesis is genuinely slower than
speech. Real models are replaced by fakes with controllable timing; the real chunker,
NullPlayer, state machine and _respond run unmodified.

Timings use tens of milliseconds, so thresholds leave generous margin for a loaded
machine; every test is also wrapped in a hard timeout, because the failure mode of a
broken producer/consumer is a hang, not an assertion error.
"""

import asyncio
import time

import numpy as np
import pytest

from amadeus.config import load_config
from amadeus.main import Amadeus
from amadeus.state import SystemState
from amadeus.synthesis.audio_out import NullPlayer
from amadeus.synthesis.base import TTSProvider
from amadeus.trace import TurnTrace

SR = 24000
HARD_TIMEOUT = 10.0


class FakeBrain:
    """Streams a fixed reply one word at a time."""

    def __init__(self, text, token_delay=0.0):
        self.words = text.split(" ")
        self.token_delay = token_delay

    async def stream(self, messages, stop=None):
        for i, w in enumerate(self.words):
            if stop is not None and stop.is_set():
                return
            await asyncio.sleep(self.token_delay)
            yield w if i == 0 else " " + w


class FakeTTS(TTSProvider):
    """Takes `synth_s` to produce `audio_s` seconds of audio, whatever the text."""

    def __init__(self, synth_s, audio_s, fail_on=None):
        self.synth_s, self.audio_s, self.fail_on = synth_s, audio_s, fail_on
        self.calls = []

    async def synth(self, text, stop=None):
        self.calls.append(text)
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("synth exploded")
        await asyncio.sleep(self.synth_s)
        return np.zeros(int(SR * self.audio_s), dtype=np.float32)

    @property
    def sample_rate(self):
        return SR


class FakePrompt:
    def build(self, user_input=None):
        return [{"role": "user", "content": user_input}]


def make_app(brain, tts, first_clause_min=None):
    from amadeus.synthesis.chunker import ClauseChunker

    app = Amadeus(load_config(), text_mode=True, mute=True)
    app.prompt = FakePrompt()
    app.brain = brain
    app.tts = tts
    app.chunker = ClauseChunker(first_clause_min=first_clause_min)
    app.player = NullPlayer(sample_rate=SR)
    return app


def respond(app, stop=None):
    """Run one _respond on a bound bus. Returns (reply, trace, wall_seconds)."""

    async def go():
        app.bus.bind_loop()  # StateManager.transition() publishes synchronously
        app.state.transition(SystemState.THINKING)
        trace = TurnTrace(1)
        trace.stamp("speech_end")
        s = stop or asyncio.Event()
        t0 = time.perf_counter()
        reply = await asyncio.wait_for(app._respond("hi", trace, s), HARD_TIMEOUT)
        return reply, trace, time.perf_counter() - t0

    return asyncio.run(go())


# Three clauses, each ending in a full stop and longer than the 10-char minimum.
THREE = "First clause is here. Second clause is here. Third clause is here."


def test_all_clauses_are_spoken_in_order():
    tts = FakeTTS(synth_s=0.01, audio_s=0.05)
    reply, trace, _ = respond(make_app(FakeBrain(THREE), tts))
    assert tts.calls == ["First clause is here.", "Second clause is here.", "Third clause is here."]
    assert reply == "First clause is here. Second clause is here. Third clause is here."
    assert trace.has("first_audio_out")


def test_synthesis_overlaps_playback_so_there_is_no_gap():
    """Synthesis (0.15 s) is faster than playback (0.40 s): clause 2 and 3 are ready
    before the previous clip ends, so the listener should hear no silence."""
    tts = FakeTTS(synth_s=0.15, audio_s=0.40)
    _, trace, wall = respond(make_app(FakeBrain(THREE), tts))

    assert len(trace.gaps_ms) == 2
    assert trace.max_gap_ms < 60, f"unexpected silence: {trace.gaps_ms}"
    # Sequential would take 3*(0.15+0.40) = 1.65 s. Pipelined: 0.15 + 3*0.40 = 1.35 s.
    assert wall < 1.55, f"no overlap: took {wall:.2f}s"


def test_gap_metric_reports_a_real_hole_when_synthesis_is_slower_than_speech():
    """Synthesis (0.30 s) slower than playback (0.10 s): about 0.20 s of silence
    before each later clause. The metric must show it, or it is useless."""
    tts = FakeTTS(synth_s=0.30, audio_s=0.10)
    _, trace, _ = respond(make_app(FakeBrain(THREE), tts))

    assert len(trace.gaps_ms) == 2
    for g in trace.gaps_ms:
        assert 120 < g < 320, f"expected ~200 ms hole, got {trace.gaps_ms}"


def test_first_audio_waits_only_for_the_first_clause():
    """First sound must not depend on how long the LATER clauses take to synthesise."""
    tts = FakeTTS(synth_s=0.20, audio_s=0.30)
    _, trace, _ = respond(make_app(FakeBrain(THREE), tts))
    assert trace.first_syllable_ms < 330, trace.first_syllable_ms  # ~one synth, not three


def test_first_clause_min_makes_the_opener_play_sooner():
    """With a short threshold the opening 'Hello!' is spoken on its own."""
    reply_text = "Hello! I am Amadeus and I am happy to help you today."
    slow = FakeTTS(synth_s=0.10, audio_s=0.10)
    app = make_app(FakeBrain(reply_text), slow, first_clause_min=0)
    respond(app)
    assert slow.calls[0] == "Hello!"

    plain = FakeTTS(synth_s=0.10, audio_s=0.10)
    respond(make_app(FakeBrain(reply_text), plain, first_clause_min=None))
    assert plain.calls[0] == "Hello! I am Amadeus and I am happy to help you today."


def test_tts_failure_propagates_without_hanging():
    """A synth error on clause 2 must surface as an exception, not deadlock the
    player on an empty queue or the synthesiser on a full one."""
    tts = FakeTTS(synth_s=0.02, audio_s=0.05, fail_on=2)
    with pytest.raises(RuntimeError, match="synth exploded"):
        respond(make_app(FakeBrain(THREE), tts))


def test_stop_mid_reply_ends_promptly():
    """Setting stop while speaking must return quickly and not play the rest."""
    tts = FakeTTS(synth_s=0.02, audio_s=2.0)  # 2 s clips: playing all would take ~6 s
    stop = asyncio.Event()

    async def go():
        app = make_app(FakeBrain(THREE), tts)
        app.bus.bind_loop()
        app.state.transition(SystemState.THINKING)
        trace = TurnTrace(1)
        trace.stamp("speech_end")
        asyncio.get_running_loop().call_later(0.3, stop.set)
        t0 = time.perf_counter()
        await asyncio.wait_for(app._respond("hi", trace, stop), HARD_TIMEOUT)
        return time.perf_counter() - t0

    assert asyncio.run(go()) < 1.5


def test_empty_audio_is_skipped_not_fatal():
    class SilentTTS(FakeTTS):
        async def synth(self, text, stop=None):
            self.calls.append(text)
            return np.array([], dtype=np.float32)

    reply, trace, _ = respond(make_app(FakeBrain(THREE), SilentTTS(0, 0)))
    assert not trace.has("first_audio_out")  # nothing was ever audible
    assert reply.startswith("First clause is here.")
