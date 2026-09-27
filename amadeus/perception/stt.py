import numpy as np
import asyncio
from typing import Optional
from faster_whisper import WhisperModel
from amadeus.trace import get_tracer
from amadeus.bus import EventBus
from amadeus.runtime.vram import vram_guard

class WhisperSTT:
    """Speech-to-Text using faster-whisper."""
    
    def __init__(self, model_size: str = "small",
                 compute_type: str = "int8_float16",
                 device: str = "cuda",
                 language: Optional[str] = None,
                 bus: Optional[EventBus] = None):
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.language = language
        self.bus = bus
        self.tracer = get_tracer("stt")
        
        self.tracer.info(f"Loading Whisper model ({model_size})...")
        with vram_guard("WhisperSTT Load"):
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
        self.tracer.info("Whisper model loaded.")
    
    async def transcribe(self, audio: np.ndarray, 
                          stop: Optional[asyncio.Event] = None) -> str:
        """Transcribe audio buffer to text."""
        
        def _run_transcribe():
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5
            )
            text = " ".join([segment.text for segment in segments])
            return text.strip()

        loop = asyncio.get_event_loop()
        
        with self.tracer.span("Whisper Transcription"):
            text = await loop.run_in_executor(None, _run_transcribe)
            
        if self.bus:
            self.bus.publish("TranscriptReady", {"text": text})
            
        return text

__all__ = ["WhisperSTT"]
