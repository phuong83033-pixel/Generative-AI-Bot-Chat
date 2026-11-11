"""The avatar wired into the real Amadeus app: a spoken reply must move a connected face,
and a broken avatar must never break the voice path."""

import asyncio
import dataclasses
import socket

import aiohttp
import numpy as np

from amadeus.config import AvatarConfig, load_config
from amadeus.emotion.engine import EmotionEngine
from amadeus.main import Amadeus
from amadeus.state import SystemState
from amadeus.synthesis.audio_out import NullPlayer
from amadeus.synthesis.base import TTSProvider
from amadeus.synthesis.chunker import ClauseChunker
from amadeus.trace import TurnTrace
from test_respond_pipeline import FakeBrain, SR, THREE


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LoudTTS(TTSProvider):
    """Every clause is 0.4 s of a constant loud tone, so the lip-sync has something to follow."""

    async def synth(self, text, stop=None, style=None):
        await asyncio.sleep(0.01)
        return np.full(int(SR * 0.4), 0.2, dtype=np.float32)

    @property
    def sample_rate(self):
        return SR


def make_app(**avatar_kw):
    cfg = dataclasses.replace(load_config(), avatar=AvatarConfig(enabled=True, fps=60, **avatar_kw))
    app = Amadeus(cfg, text_mode=True, mute=True)
    app.prompt = type("P", (), {"build": lambda self, u=None, mood_hint=None: [{"role": "user", "content": u}]})()
    app.brain = FakeBrain(THREE)
    app.tts = LoudTTS()
    app.chunker = ClauseChunker(first_clause_min=None)
    app.player = NullPlayer(sample_rate=SR, bus=app.bus, lead_s=0.05)
    app.emotion = EmotionEngine(cfg.emotion, bus=app.bus)
    return app


def test_a_spoken_reply_moves_the_face_in_a_connected_browser():
    async def go():
        port = free_port()
        app = make_app(web=True, web_port=port)
        app.bus.bind_loop()
        await app._start_avatar()
        assert app.avatar_driver is not None
        frames = []
        try:
            async with aiohttp.ClientSession() as s:
                ws = await s.ws_connect(f"http://127.0.0.1:{port}/ws")

                async def collect():
                    async for msg in ws:
                        frames.append(msg.json())

                collector = asyncio.create_task(collect())
                await app.emotion.on_user_utterance("You're really smart.")
                app.state.transition(SystemState.THINKING)
                # The fake brain answers instantly, so without this the THINKING phase would last ~10 ms:
                # shorter than one frame at 60 fps, and the test would sometimes never sample it.
                await asyncio.sleep(0.15)
                await asyncio.wait_for(app._respond("hi", TurnTrace(1), asyncio.Event()), 10)
                app.state.transition(SystemState.IDLE)
                await asyncio.sleep(0.5)                               # let the mouth close again
                collector.cancel()
                await asyncio.gather(collector, return_exceptions=True)
        finally:
            await app.shutdown()
        return frames

    frames = asyncio.run(go())
    assert len(frames) > 30
    states = [f["state"] for f in frames]
    assert "THINKING" in states and "SPEAKING" in states and states[-1] == "IDLE"
    speaking = [f for f in frames if f["state"] == "SPEAKING"]
    assert max(f["mouth_open"] for f in speaking) > 0.8, "mouth never opened while she spoke"
    assert frames[-1]["mouth_open"] < 0.05, "mouth stayed open after she finished"
    assert {f["label"] for f in frames} >= {"flustered"} and max(f["blush"] for f in frames) > 0.2


def test_with_the_avatar_off_nothing_is_started():
    async def go():
        cfg = load_config()
        app = Amadeus(cfg, text_mode=True, mute=True)       # neither flag nor avatar.enabled
        assert app.avatar is False and app.avatar_driver is None
    asyncio.run(go())


def test_the_cli_flag_turns_it_on():
    app = Amadeus(load_config(), text_mode=True, mute=True, avatar=True)
    assert app.avatar is True


def test_a_port_conflict_leaves_the_voice_assistant_running_without_a_face():
    async def go():
        port = free_port()
        squatter = await asyncio.start_server(lambda r, w: None, "127.0.0.1", port)   # something else owns the port
        app = make_app(web=True, web_port=port)
        app.bus.bind_loop()
        try:
            await app._start_avatar()                        # must not raise
            return app.avatar_driver
        finally:
            squatter.close()
            await app.shutdown()

    assert asyncio.run(go()) is None


def test_an_avatar_that_blows_up_at_startup_is_contained(monkeypatch):
    import amadeus.expression.driver as driver_mod

    async def boom(self):
        raise RuntimeError("renderer exploded")

    monkeypatch.setattr(driver_mod.AvatarDriver, "start", boom)

    async def go():
        app = make_app(web=True, web_port=free_port())
        app.bus.bind_loop()
        try:
            await app._start_avatar()                        # swallowed and logged
        finally:
            await app.shutdown()

    asyncio.run(go())


def test_the_voice_path_does_not_depend_on_the_avatar():
    """The same reply, with and without an avatar attached, speaks the same clauses."""
    async def speak(with_avatar):
        app = make_app(web=True, web_port=free_port())
        app.bus.bind_loop()
        if with_avatar:
            await app._start_avatar()
        app.state.transition(SystemState.THINKING)
        try:
            return await asyncio.wait_for(app._respond("hi", TurnTrace(1), asyncio.Event()), 10)
        finally:
            await app.shutdown()

    assert asyncio.run(speak(True)) == asyncio.run(speak(False))
