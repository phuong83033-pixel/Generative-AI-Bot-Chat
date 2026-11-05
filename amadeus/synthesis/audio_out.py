import asyncio
import collections
import sys
import threading
import time

import numpy as np
import sounddevice as sd
from amadeus.trace import get_tracer
from amadeus.bus import EventBus


class AudioPlayer:
    """Plays audio through the default output device, one blocking sd.play() per clip.

    LEGACY. Kept as a fallback (audio.streaming_output: false). Measured on this
    machine each sd.play() adds ~200 ms of dead air before and after the clip (MME
    opens and drains a stream per call), so every clause boundary has a 200 ms hole
    however fast synthesis is. StreamingAudioPlayer replaces it.
    """

    lead_s = 0.0  # play() returns exactly when the clip ends

    def __init__(self, sample_rate: int = 24000,
                 device: int | None = None,
                 bus: EventBus | None = None):
        self._sample_rate = sample_rate
        self._device = device
        self._bus = bus
        self._tracer = get_tracer("audio_out")
        self._playing = False

    async def play(self, audio: np.ndarray,
                   stop: asyncio.Event | None = None) -> None:
        """Play audio array through speakers.

        Args:
            audio: float32 numpy array
            stop: Cancellation event to stop playback mid-stream
        """
        if audio.size == 0:
            return

        self._playing = True
        if self._bus:
            await self._bus.publish("TTSStarted", {})

        # Use sounddevice.play with a callback for cancellation
        # Run in executor to not block async loop
        loop = asyncio.get_event_loop()

        try:
            finished = asyncio.Event()

            def _play_blocking():
                try:
                    sd.play(audio, samplerate=self._sample_rate,
                            device=self._device, blocking=True)
                except sd.PortAudioError as e:
                    self._tracer.error(f"Audio playback error: {e}")
                finally:
                    loop.call_soon_threadsafe(finished.set)

            fut = loop.run_in_executor(None, _play_blocking)

            # Wait for either playback to finish or stop signal
            if stop:
                done = asyncio.ensure_future(finished.wait())
                stop_task = asyncio.ensure_future(stop.wait())
                await asyncio.wait(
                    [done, stop_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                if stop.is_set():
                    sd.stop()
                    self._tracer.info("Playback cancelled")
                stop_task.cancel()
                done.cancel()
            else:
                await finished.wait()

            await fut
        finally:
            self._playing = False
            if self._bus:
                await self._bus.publish("TTSDone", {})

    async def drain(self, stop: asyncio.Event | None = None) -> None:
        """play() already waits for the whole clip, so there is nothing left to wait for."""

    @property
    def is_playing(self) -> bool:
        return self._playing

    def stop_immediately(self) -> None:
        """Stop any current playback immediately."""
        sd.stop()
        self._playing = False

    def close(self) -> None:
        pass


class StreamingAudioPlayer:
    """Plays clips back to back through ONE persistent output stream.

    Why it exists
    -------------
    The legacy player opens and drains a stream for every clip, which costs ~200 ms
    of silence at each clause boundary (measured: min 197, median 201, max 214 ms
    over 8 plays). Here the stream stays open and its callback drains a queue, so a
    clip appended while the previous one is still playing follows it sample-for-sample.

    Why WASAPI
    ----------
    Windows' default host API here is MME, with 93 ms of stream latency. WASAPI shared
    mode measured 25 ms. It refuses 24 kHz outright ("Invalid sample rate") unless
    `auto_convert=True`, which lets Windows resample, so that is always set. If the
    WASAPI stream cannot be opened the player falls back to the default device.

    Early return
    ------------
    play() returns `lead_s` BEFORE its clip finishes, so the caller can queue the next
    clip in time. A caller that waited for the exact end would append after the
    callback had already started emitting silence. drain() waits for the true end.

    Thread model
    ------------
    The PortAudio callback runs on its own thread; everything it shares with the
    event loop (queue, counters, waiters) is guarded by one lock.
    """

    LEAD_S = 0.05

    def __init__(self, sample_rate: int = 24000,
                 device: int | str | None = None,
                 bus: EventBus | None = None,
                 lead_s: float = LEAD_S):
        self._sample_rate = sample_rate
        self._device = device
        self._bus = bus
        self._lead_s = lead_s
        self._tracer = get_tracer("audio_out")

        self._lock = threading.Lock()
        self._queue: collections.deque[np.ndarray] = collections.deque()
        self._offset = 0            # read position inside self._queue[0]
        self._queued_frames = 0     # frames ever appended
        self._played_frames = 0     # frames the callback has consumed
        self._waiters: list[tuple[int, asyncio.Future, asyncio.AbstractEventLoop]] = []
        self._stream = None

    @property
    def lead_s(self) -> float:
        return self._lead_s

    # ------------------------------------------------------------------ device

    def _resolve_output(self):
        """Choose (device, extra_settings): WASAPI shared on Windows, else as given."""
        if sys.platform != "win32" or isinstance(self._device, int):
            return self._device, None
        try:
            apis = sd.query_hostapis()
            wasapi = next(i for i, a in enumerate(apis) if "WASAPI" in a["name"])
        except (StopIteration, sd.PortAudioError):
            return self._device, None

        if self._device is None:
            dev = apis[wasapi]["default_output_device"]
            if dev is None or dev < 0:
                return self._device, None
        else:
            want = str(self._device).lower()
            dev = next((i for i, d in enumerate(sd.query_devices())
                        if d["hostapi"] == wasapi and d["max_output_channels"] > 0
                        and want in d["name"].lower()), None)
            if dev is None:
                return self._device, None
        return dev, sd.WasapiSettings(auto_convert=True)

    def _ensure_stream(self) -> None:
        if self._stream is not None:
            return
        device, extra = self._resolve_output()
        attempts = [(device, extra)]
        if extra is not None:
            attempts.append((self._device, None))  # fall back to the default host API

        last_error: Exception | None = None
        for dev, settings in attempts:
            try:
                stream = sd.OutputStream(
                    samplerate=self._sample_rate, channels=1, dtype="float32",
                    device=dev, latency="low", extra_settings=settings,
                    callback=self._callback,
                )
                stream.start()
            except Exception as e:  # PortAudioError, ValueError for bad device names
                last_error = e
                self._tracer.warning(f"Output stream failed on device {dev!r}: {e}")
                continue
            self._stream = stream
            self._tracer.info(
                f"Output stream open: {sd.query_devices(stream.device)['name']}, "
                f"latency {stream.latency * 1000:.0f} ms"
            )
            return
        raise RuntimeError(f"Could not open an audio output stream: {last_error}")

    # ---------------------------------------------------------------- callback

    def _callback(self, outdata, frames, time_info, status) -> None:
        """Runs on PortAudio's thread. Must be quick and must never raise."""
        if status:
            self._tracer.warning(f"Audio output status: {status}")
        done: list[tuple[asyncio.Future, asyncio.AbstractEventLoop]] = []
        with self._lock:
            n = 0
            while n < frames and self._queue:
                chunk = self._queue[0]
                take = min(frames - n, chunk.size - self._offset)
                outdata[n:n + take, 0] = chunk[self._offset:self._offset + take]
                n += take
                self._offset += take
                if self._offset >= chunk.size:
                    self._queue.popleft()
                    self._offset = 0
            if n < frames:
                outdata[n:, 0] = 0.0  # underrun between clips is silence, not noise
            self._played_frames += n

            if self._waiters:
                still = []
                for target, fut, loop in self._waiters:
                    if self._played_frames >= target:
                        done.append((fut, loop))
                    else:
                        still.append((target, fut, loop))
                self._waiters = still
        for fut, loop in done:
            self._release(fut, loop)

    @staticmethod
    def _release(fut: asyncio.Future, loop: asyncio.AbstractEventLoop) -> None:
        def _set() -> None:
            if not fut.done():
                fut.set_result(None)
        try:
            loop.call_soon_threadsafe(_set)
        except RuntimeError:
            pass  # loop already closed (shutdown)

    # --------------------------------------------------------------------- API

    async def _wait(self, fut: asyncio.Future, stop: asyncio.Event | None) -> None:
        if stop is None:
            await fut
            return
        stop_wait = asyncio.ensure_future(stop.wait())
        try:
            await asyncio.wait({fut, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop_wait.cancel()
        if stop.is_set() and not fut.done():
            self.stop_immediately()
            self._tracer.info("Playback cancelled")

    async def play(self, audio: np.ndarray, stop: asyncio.Event | None = None) -> None:
        """Queue a clip. Returns `lead_s` before it finishes playing (see class docs)."""
        if audio.size == 0:
            return
        loop = asyncio.get_running_loop()
        self._ensure_stream()
        clip = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        lead_frames = int(self._lead_s * self._sample_rate)
        fut = loop.create_future()

        with self._lock:
            self._queue.append(clip)
            self._queued_frames += clip.size
            target = self._queued_frames - lead_frames
            if self._played_frames >= target:
                fut.set_result(None)      # clip shorter than the lead: nothing to wait for
            else:
                self._waiters.append((target, fut, loop))

        if self._bus:
            await self._bus.publish("TTSStarted", {})
        await self._wait(fut, stop)

    async def drain(self, stop: asyncio.Event | None = None) -> None:
        """Wait until everything queued has actually been played."""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        with self._lock:
            if self._played_frames >= self._queued_frames:
                fut.set_result(None)
            else:
                self._waiters.append((self._queued_frames, fut, loop))
        await self._wait(fut, stop)
        if self._bus:
            await self._bus.publish("TTSDone", {})

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return bool(self._queue)

    def stop_immediately(self) -> None:
        """Drop everything queued. Safe from any thread; wakes any waiting play()/drain()."""
        with self._lock:
            self._queue.clear()
            self._offset = 0
            self._played_frames = self._queued_frames
            waiters, self._waiters = self._waiters, []
        for _, fut, loop in waiters:
            self._release(fut, loop)

    def close(self) -> None:
        self.stop_immediately()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:
                self._tracer.warning(f"Error closing output stream: {e}")


class NullPlayer:
    """Silent stand-in for the players above that takes exactly as long as playback would.

    Used by `--mute` and by scripts/bench_latency.py. It waits for the clip's real
    duration rather than returning instantly, because the pipeline's timing depends
    on playback taking time: synthesis of clause N+1 overlaps playback of clause N,
    and a player that returned immediately would hide every inter-clause gap.
    Honours `stop`, like the real players.

    It keeps a running "busy until" time, so a clip that arrives while the previous
    one is still playing starts exactly when it ends (gapless), and one that arrives
    late starts immediately (a gap). `lead_s` reproduces StreamingAudioPlayer's early
    return; leave it 0 to model the legacy blocking player. It does NOT model the
    legacy player's ~200 ms per-clip overhead.
    """

    def __init__(self, sample_rate: int = 24000, bus: EventBus | None = None,
                 lead_s: float = 0.0):
        self._sample_rate = sample_rate
        self._bus = bus
        self._lead_s = lead_s
        self._busy_until = 0.0  # time.monotonic() when the queued audio ends

    @property
    def lead_s(self) -> float:
        return self._lead_s

    async def _sleep_until(self, t: float, stop: asyncio.Event | None) -> None:
        timeout = max(0.0, t - time.monotonic())
        if stop is None:
            await asyncio.sleep(timeout)
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass  # reached the target time
        else:
            self.stop_immediately()

    async def play(self, audio: np.ndarray, stop: asyncio.Event | None = None) -> None:
        if audio.size == 0:
            return
        now = time.monotonic()
        start = max(now, self._busy_until)
        self._busy_until = start + audio.size / self._sample_rate
        await self._sleep_until(self._busy_until - self._lead_s, stop)

    async def drain(self, stop: asyncio.Event | None = None) -> None:
        await self._sleep_until(self._busy_until, stop)

    @property
    def is_playing(self) -> bool:
        return time.monotonic() < self._busy_until

    def stop_immediately(self) -> None:
        self._busy_until = 0.0

    def close(self) -> None:
        pass
