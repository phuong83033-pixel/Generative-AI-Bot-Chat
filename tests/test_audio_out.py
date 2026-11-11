"""Tests for StreamingAudioPlayer and NullPlayer.

No sound device is touched. sounddevice.OutputStream is replaced by a fake whose
"device thread" calls the player's REAL callback at real-time pace (20 ms blocks), so
the queueing, completion signalling and cancellation under test are the production
code, and the samples the "speaker" receives can be inspected exactly.

Test clips contain no zero samples, so any zero between the first and last non-zero
sample of the captured stream is a gap the listener would hear.
"""

import asyncio
import threading
import time

import numpy as np
import pytest

from amadeus.synthesis import audio_out
from amadeus.synthesis.audio_out import NullPlayer, StreamingAudioPlayer

SR = 24000
BLOCK = 480  # 20 ms


def clip(seconds, start=0.1):
    """A non-zero ramp, so gaps (zeros) and reordering are both detectable."""
    return np.linspace(start, start + 0.5, int(SR * seconds), dtype=np.float32)


class FakeStream:
    """Stands in for sounddevice.OutputStream; a thread plays the part of the speaker."""

    instances = []

    def __init__(self, samplerate, channels, dtype, device, latency, extra_settings, callback):
        assert samplerate == SR and channels == 1 and dtype == "float32"
        self.callback = callback
        self.device = 0
        self.latency = 0.025
        self.captured = []
        self.stopped = self.closed = False
        self._running = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        FakeStream.instances.append(self)

    def start(self):
        self._running.set()
        self._thread.start()

    def _run(self):
        while self._running.is_set():
            out = np.zeros((BLOCK, 1), dtype=np.float32)
            self.callback(out, BLOCK, None, None)
            self.captured.append(out[:, 0].copy())
            time.sleep(BLOCK / SR)

    def stop(self):
        self._running.clear()
        self.stopped = True

    def close(self):
        self.closed = True

    def heard(self):
        return np.concatenate(self.captured) if self.captured else np.zeros(0, dtype=np.float32)


@pytest.fixture
def player(monkeypatch):
    FakeStream.instances.clear()
    monkeypatch.setattr(audio_out.sd, "OutputStream", FakeStream)
    monkeypatch.setattr(audio_out.sd, "query_hostapis", lambda: [])  # no WASAPI branch
    monkeypatch.setattr(audio_out.sd, "query_devices", lambda *a, **k: {"name": "fake"})
    p = StreamingAudioPlayer(sample_rate=SR)
    yield p
    p.close()


def stream_of(p):
    return FakeStream.instances[-1]


def nonzero_span(x):
    nz = np.flatnonzero(x)
    return x[nz[0]: nz[-1] + 1] if nz.size else x[:0]


def run(coro, timeout=10):
    return asyncio.run(asyncio.wait_for(coro, timeout))


# ------------------------------------------------------------- StreamingAudioPlayer

def test_every_sample_arrives_in_order(player):
    a, b = clip(0.20, 0.1), clip(0.20, 0.3)

    async def go():
        await player.play(a)
        await player.play(b)
        await player.drain()

    run(go())
    assert np.array_equal(nonzero_span(stream_of(player).heard()), np.concatenate([a, b]))


def test_clips_queued_early_play_back_to_back_with_no_gap(player):
    """play() returns a little BEFORE the clip ends, so the second clip is queued in
    time and follows it sample for sample: no zeros between the two."""
    # 0.3017 s = 7241 frames: NOT a whole number of 480-frame blocks, like real TTS
    # output. A block-aligned clip would end exactly on a block boundary and hide the
    # gap that appending late causes (the mutation check caught exactly that).
    a, b = clip(0.3017, 0.1), clip(0.3017, 0.3)

    async def go():
        await player.play(a)
        await player.play(b)  # called ~lead_s before `a` finishes
        await player.drain()

    run(go())
    span = nonzero_span(stream_of(player).heard())
    assert span.size == a.size + b.size, "zeros were inserted between the clips"
    assert np.all(span != 0)


def test_a_clip_that_arrives_late_leaves_silence_not_garbage(player):
    a, b = clip(0.10), clip(0.10)

    async def go():
        await player.play(a)
        await player.drain()
        await asyncio.sleep(0.15)
        await player.play(b)
        await player.drain()

    run(go())
    span = nonzero_span(stream_of(player).heard())
    assert np.array_equal(span[: a.size], a) and np.array_equal(span[-b.size:], b)
    between = span[a.size: span.size - b.size]
    assert between.size >= int(0.10 * SR), "expected ~150 ms of silence between the clips"
    assert not between.any(), "the gap must be exact silence, not leftover audio"


def test_play_returns_about_lead_before_the_clip_ends(player):
    a = clip(0.40)

    async def go():
        t0 = time.perf_counter()
        await player.play(a)
        returned = time.perf_counter() - t0
        await player.drain()
        return returned, time.perf_counter() - t0

    returned, total = run(go())
    assert total == pytest.approx(0.40, abs=0.10)
    assert returned == pytest.approx(0.40 - player.lead_s, abs=0.10)
    assert returned < total


def test_stop_flushes_the_queue_and_wakes_the_waiter(player):
    long_clip = clip(3.0)
    stop = asyncio.Event()

    async def go():
        asyncio.get_running_loop().call_later(0.15, stop.set)
        t0 = time.perf_counter()
        await player.play(long_clip, stop)
        return time.perf_counter() - t0

    elapsed = run(go())
    assert elapsed < 0.6, f"play() did not return promptly after stop ({elapsed:.2f}s)"
    time.sleep(0.1)
    tail = stream_of(player).heard()[-int(0.08 * SR):]
    assert not tail.any(), "audio kept playing after stop"
    assert not player.is_playing


def test_drain_on_an_idle_player_returns_immediately(player):
    async def go():
        t0 = time.perf_counter()
        await player.drain()
        return time.perf_counter() - t0

    assert run(go()) < 0.1


def test_empty_audio_is_ignored(player):
    run(player.play(np.array([], dtype=np.float32)))
    assert not FakeStream.instances            # not even a stream was opened


def test_close_stops_and_closes_the_stream(player):
    run(player.play(clip(0.05)))
    s = stream_of(player)
    player.close()
    assert s.stopped and s.closed


# ------------------------------------------------------------------------- NullPlayer

def test_null_player_lead_returns_early_and_drain_waits_for_the_end():
    async def go():
        p = NullPlayer(SR, lead_s=0.10)
        t0 = time.perf_counter()
        await p.play(np.zeros(int(SR * 0.30), dtype=np.float32))
        returned = time.perf_counter() - t0
        await p.drain()
        return returned, time.perf_counter() - t0

    returned, total = run(go())
    assert returned == pytest.approx(0.20, abs=0.06)
    assert total == pytest.approx(0.30, abs=0.06)


def test_null_player_queues_a_second_clip_behind_the_first():
    async def go():
        p = NullPlayer(SR, lead_s=0.05)
        a = np.zeros(int(SR * 0.30), dtype=np.float32)
        t0 = time.perf_counter()
        await p.play(a)
        await p.play(a)          # arrives ~0.05 s before the first ends
        await p.drain()
        return time.perf_counter() - t0

    assert run(go()) == pytest.approx(0.60, abs=0.08)  # back to back, no gap added


def test_null_player_zero_lead_matches_a_blocking_player():
    async def go():
        p = NullPlayer(SR)
        t0 = time.perf_counter()
        await p.play(np.zeros(SR // 4, dtype=np.float32))
        return time.perf_counter() - t0

    assert run(go()) == pytest.approx(0.25, abs=0.06)


def test_null_player_stop_cuts_playback_short():
    async def go():
        p = NullPlayer(SR)
        stop = asyncio.Event()
        asyncio.get_running_loop().call_later(0.1, stop.set)
        t0 = time.perf_counter()
        await p.play(np.zeros(SR * 2, dtype=np.float32), stop)
        return time.perf_counter() - t0

    assert run(go()) < 0.4


# ------------------------------------------------- level (lip-sync source, M14)

def test_level_is_zero_before_anything_plays(player):
    assert player.level == 0.0


def test_level_is_the_loudness_of_what_is_actually_playing(player):
    async def go():
        # play() returns lead_s BEFORE the clip ends, so sample while it is still playing.
        task = asyncio.create_task(player.play(np.full(int(SR * 0.5), 0.3, dtype=np.float32)))
        await asyncio.sleep(0.2)
        during = player.level
        await task
        await player.drain()
        await asyncio.sleep(0.1)                                           # a couple of silent blocks
        return during, player.level

    during, after = run(go())
    assert during == pytest.approx(0.3, abs=0.01)
    assert after == 0.0


def test_level_drops_to_zero_at_once_when_playback_is_cut():
    async def go():
        with pytest.MonkeyPatch.context() as mp:
            FakeStream.instances.clear()
            mp.setattr(audio_out.sd, "OutputStream", FakeStream)
            mp.setattr(audio_out.sd, "query_hostapis", lambda: [])
            mp.setattr(audio_out.sd, "query_devices", lambda *a, **k: {"name": "fake"})
            p = StreamingAudioPlayer(sample_rate=SR)
            try:
                task = asyncio.create_task(p.play(np.full(int(SR * 2), 0.3, dtype=np.float32)))
                await asyncio.sleep(0.15)
                loud = p.level
                p.stop_immediately()
                return loud, p.level
            finally:
                p.close()

    loud, cut = run(go())
    assert loud > 0.2 and cut == 0.0


def test_null_player_simulates_the_level_of_the_clip_that_would_be_playing():
    async def go():
        p = NullPlayer(sample_rate=SR)
        before = p.level
        task = asyncio.create_task(p.play(np.full(int(SR * 0.4), 0.3, dtype=np.float32)))
        await asyncio.sleep(0.1)
        during = p.level
        await task
        # asyncio's timer resolution on Windows is ~15 ms, so play() can return slightly BEFORE the
        # simulated end of the clip. Allow for that: the level must be 0 shortly after, not at that instant.
        await asyncio.sleep(0.06)
        return before, during, p.level

    before, during, after = run(go())
    assert before == 0.0 and during == pytest.approx(0.3, abs=0.01) and after == 0.0


def test_legacy_player_has_no_level():
    assert audio_out.AudioPlayer().level == 0.0
