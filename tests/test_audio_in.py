"""MicrophoneStream backlog handling. No audio device is opened."""

import asyncio

import numpy as np

from amadeus.perception.audio_in import MicrophoneStream


def chunk(v=0.1):
    return np.full(512, v, dtype=np.float32)


def test_flush_drops_everything_recorded_so_far_and_reports_the_count():
    async def go():
        mic = MicrophoneStream()
        for _ in range(7):
            mic.queue.put_nowait(chunk())
        dropped = mic.flush()
        return dropped, mic.queue.empty()

    dropped, empty = asyncio.run(go())
    assert dropped == 7 and empty


def test_flush_on_an_empty_queue_is_a_noop():
    async def go():
        return MicrophoneStream().flush()

    assert asyncio.run(go()) == 0


def test_audio_recorded_after_the_flush_is_kept():
    """The point of the flush: old audio goes, but what arrives afterwards is heard."""
    async def go():
        mic = MicrophoneStream()
        mic.queue.put_nowait(chunk(0.1))          # recorded while the agent was speaking
        mic.flush()
        mic.queue.put_nowait(chunk(0.9))          # the user's next sentence
        stop = asyncio.Event()
        async for c in mic.stream(stop):
            return float(c[0])

    assert asyncio.run(go()) == np.float32(0.9)
