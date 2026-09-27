import asyncio
import numpy as np
import sounddevice as sd
from amadeus.trace import get_tracer
from amadeus.bus import EventBus

class AudioPlayer:
    """Plays audio through the default output device."""
    
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
    
    @property
    def is_playing(self) -> bool:
        return self._playing
    
    def stop_immediately(self) -> None:
        """Stop any current playback immediately."""
        sd.stop()
        self._playing = False
