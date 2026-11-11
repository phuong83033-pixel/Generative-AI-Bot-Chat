"""AvatarDriver: runs the ExpressionModel at a fixed frame rate and feeds the renderers.

It is an OUTPUT CONSUMER (README section 26). It listens to the bus and reads the speaker
level; it never writes to cognition, memory or the state machine, and nothing in the
pipeline waits for it. Remove it and the voice assistant behaves identically.

    bus "EmotionChanged" --\
    bus "StateChanged"  ----+--> ExpressionModel.step(dt) --> ExpressionFrame --> sinks
    player.level ----------/                                   (30 Hz)        (browser, VTube Studio)

A sink is anything with `async send(frame)` and `async close()`. A sink that raises is logged
(rate-limited) and skipped for that frame, never allowed to stall the loop or the others.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Protocol

from amadeus.bus import EventBus
from amadeus.expression.frame import ExpressionFrame, ExpressionModel
from amadeus.trace import get_tracer


class Sink(Protocol):
    async def send(self, frame: ExpressionFrame) -> None: ...
    async def close(self) -> None: ...


class AvatarDriver:
    def __init__(
        self,
        bus: EventBus,
        sinks: list[Sink],
        level_source: Callable[[], float] = lambda: 0.0,
        fps: int = 30,
        model: ExpressionModel | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._bus = bus
        self._sinks = list(sinks)
        self._level_source = level_source
        self._period = 1.0 / max(1, fps)
        self.model = model or ExpressionModel()
        self._clock = clock
        self._tracer = get_tracer("avatar")
        self._task: asyncio.Task | None = None
        self._last_warn: dict[int, float] = {}
        self.frames_sent = 0
        self.max_tick_ms = 0.0   # worst time one tick took: proves the avatar is not stealing the loop

    # ---------------------------------------------------------------- bus input

    def _on_emotion(self, snap) -> None:
        self.model.set_emotion(snap.pad, snap.baseline, snap.label, snap.intensity)

    def _on_state(self, data) -> None:
        self.model.set_state(data["new"].name)

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        self._bus.subscribe("EmotionChanged", self._on_emotion)
        self._bus.subscribe("StateChanged", self._on_state)
        self._task = asyncio.create_task(self._run(), name="avatar-driver")
        self._tracer.info(f"Avatar driver started at {1 / self._period:.0f} fps, {len(self._sinks)} sink(s)")

    async def stop(self) -> None:
        self._bus.unsubscribe("EmotionChanged", self._on_emotion)
        self._bus.unsubscribe("StateChanged", self._on_state)
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        for sink in self._sinks:
            try:
                await sink.close()
            except Exception as e:  # closing must never block shutdown
                self._tracer.warning(f"Error closing {type(sink).__name__}: {e}")

    # --------------------------------------------------------------------- loop

    async def tick(self, dt: float) -> ExpressionFrame:
        """One frame: sample the speaker, advance the face, push to every sink."""
        t0 = self._clock()
        self.model.set_level(self._level_source())
        frame = self.model.step(dt)
        for i, sink in enumerate(self._sinks):
            try:
                await sink.send(frame)
            except Exception as e:
                now = self._clock()
                if now - self._last_warn.get(i, -1e9) > 5.0:
                    self._last_warn[i] = now
                    self._tracer.warning(f"{type(sink).__name__} failed: {e}")
        self.frames_sent += 1
        self.max_tick_ms = max(self.max_tick_ms, (self._clock() - t0) * 1000)
        return frame

    async def _run(self) -> None:
        last = self._clock()
        next_due = last
        while True:
            now = self._clock()
            await self.tick(now - last)
            last = now
            # Fixed-rate pacing: schedule against the ideal timeline, not "sleep(period)",
            # which would let every slow tick push all later ones back (drift).
            next_due += self._period
            delay = next_due - self._clock()
            if delay < -self._period:      # fell badly behind: resync instead of bursting
                next_due = self._clock()
                delay = 0.0
            await asyncio.sleep(max(0.0, delay))


__all__ = ["AvatarDriver", "Sink"]
