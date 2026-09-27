import asyncio
import numpy as np
import sounddevice as sd
from typing import AsyncIterator, Optional
from amadeus.trace import get_tracer
from amadeus.bus import EventBus

class MicrophoneStream:
    """Async microphone input stream using sounddevice."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 chunk_size: int = 512, device: Optional[int] = None,
                 bus: Optional[EventBus] = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = device
        self.bus = bus
        self.tracer = get_tracer("audio_in")
        
        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self.stream: Optional[sd.InputStream] = None
        self._loop = asyncio.get_event_loop()

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
        if status:
            self.tracer.warning(f"Audio callback status: {status}")
        data = indata.copy().flatten()
        self._loop.call_soon_threadsafe(self.queue.put_nowait, data)

    async def start(self) -> None:
        """Start recording."""
        if self.stream is not None:
            self.tracer.warning("Stream already started.")
            return

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            blocksize=self.chunk_size,
            device=self.device,
            callback=self._audio_callback
        )
        self.stream.start()
        self.tracer.info("Microphone stream started.")

    async def stop(self) -> None:
        """Stop recording."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.tracer.info("Microphone stream stopped.")

    async def read_chunk(self) -> np.ndarray:
        """Read a single audio chunk. Blocks until available."""
        return await self.queue.get()

    async def stream(self, stop: asyncio.Event) -> AsyncIterator[np.ndarray]:
        """Async generator yielding audio chunks until stop is set."""
        while not stop.is_set():
            try:
                chunk = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                yield chunk
            except asyncio.TimeoutError:
                continue

__all__ = ["MicrophoneStream"]
