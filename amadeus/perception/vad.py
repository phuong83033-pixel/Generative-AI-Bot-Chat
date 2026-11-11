import torch
import numpy as np
import asyncio
import collections
from typing import Optional, Deque
from amadeus.trace import get_tracer
from amadeus.bus import EventBus
from amadeus.perception.audio_in import MicrophoneStream

class SileroVAD:
    """Voice Activity Detection using Silero VAD model (CPU)."""
    
    def __init__(self, threshold: float = 0.5, 
                 min_speech_ms: int = 250,
                 silence_timeout_ms: int = 800,
                 pre_speech_pad_ms: int = 300,
                 sample_rate: int = 16000,
                 bus: Optional[EventBus] = None):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.silence_timeout_ms = silence_timeout_ms
        self.pre_speech_pad_ms = pre_speech_pad_ms
        self.sample_rate = sample_rate
        self.bus = bus
        self.tracer = get_tracer("vad")
        
        self.tracer.info("Loading Silero VAD model...")
        self.model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
        self.model.eval() # ensure evaluation mode
        self.tracer.info("Silero VAD model loaded.")

    def is_speech(self, chunk: np.ndarray) -> bool:
        """Check if audio chunk contains speech."""
        tensor = torch.from_numpy(chunk).float()
        
        # Add batch dimension if needed
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
            
        with torch.no_grad():
            prob = self.model(tensor, self.sample_rate).item()
            
        return prob > self.threshold

    async def listen_until_speech_ends(self, mic: MicrophoneStream, stop: asyncio.Event) -> Optional[np.ndarray]:
        """Listen to mic, accumulate speech, return when silence detected."""
        # Silero is a recurrent model: its hidden state carries over between calls.
        # Start each utterance clean so the previous one cannot bias this one.
        try:
            self.model.reset_states()
        except AttributeError:
            pass  # a model without recurrent state has nothing to reset

        chunk_ms = (mic.chunk_size / self.sample_rate) * 1000
        pad_chunks = int(self.pre_speech_pad_ms / chunk_ms)
        
        ring_buffer: Deque[np.ndarray] = collections.deque(maxlen=pad_chunks)
        speech_buffer = []
        
        in_speech = False
        silence_accumulated_ms = 0.0
        
        self.tracer.info("VAD listening...")
        
        async for chunk in mic.stream(stop):
            if stop.is_set():
                break
                
            has_speech = self.is_speech(chunk)
            
            if not in_speech:
                if has_speech:
                    in_speech = True
                    self.tracer.info("Speech started.")
                    if self.bus:
                        await self.bus.publish("SpeechStart", {})
                    speech_buffer.extend(list(ring_buffer))
                    speech_buffer.append(chunk)
                    silence_accumulated_ms = 0.0
                else:
                    ring_buffer.append(chunk)
            else:
                speech_buffer.append(chunk)
                if not has_speech:
                    silence_accumulated_ms += chunk_ms
                    if silence_accumulated_ms >= self.silence_timeout_ms:
                        self.tracer.info("Silence timeout reached. Speech ended.")
                        if self.bus:
                            await self.bus.publish("SpeechEnd", {})
                        return np.concatenate(speech_buffer)
                else:
                    silence_accumulated_ms = 0.0
                    
        return None

__all__ = ["SileroVAD"]
