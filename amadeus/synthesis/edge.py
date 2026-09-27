import asyncio
import io
import numpy as np
import edge_tts
from pydub import AudioSegment
from amadeus.synthesis.base import TTSProvider
from amadeus.trace import get_tracer

class EdgeTTS(TTSProvider):
    """Text-to-Speech using Microsoft Edge TTS (free cloud API).
    
    Note: Requires `ffmpeg` to be installed on the system for MP3 to PCM decoding via pydub.
    """
    
    def __init__(self, voice: str = "en-US-AriaNeural",
                 rate: str = "+0%",
                 output_sample_rate: int = 24000):
        self._voice = voice
        self._rate = rate
        self._output_sample_rate = output_sample_rate
        self._tracer = get_tracer("edge_tts")
    
    async def synth(self, text: str,
                    stop: asyncio.Event | None = None) -> np.ndarray:
        """Synthesize speech using Edge TTS."""
        self._tracer.info(f"Synthesizing text: {text[:50]}...")
        communicate = edge_tts.Communicate(text, self._voice, rate=self._rate)
        audio_bytes = b""
        
        try:
            async for chunk in communicate.stream():
                if stop and stop.is_set():
                    self._tracer.info("Synthesis cancelled.")
                    return np.array([], dtype=np.float32)
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
        except Exception as e:
            self._tracer.error(f"Error during synthesis: {e}")
            return np.array([], dtype=np.float32)
            
        if not audio_bytes:
            return np.array([], dtype=np.float32)
        
        # Decode MP3 to PCM
        try:
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            
            # Resample if needed
            if audio_segment.frame_rate != self._output_sample_rate:
                audio_segment = audio_segment.set_frame_rate(self._output_sample_rate)
                
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0  # normalize int16 to float32
            
            if audio_segment.channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)  # downmix to mono
                
            return samples
        except Exception as e:
            self._tracer.error(f"Error decoding MP3 to PCM: {e}")
            return np.array([], dtype=np.float32)
    
    @property
    def sample_rate(self) -> int:
        return self._output_sample_rate
