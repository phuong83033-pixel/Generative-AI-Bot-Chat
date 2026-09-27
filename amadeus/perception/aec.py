import numpy as np
from typing import Optional
from amadeus.trace import get_tracer

class AECStub:
    """Acoustic Echo Cancellation stub - pass-through for development.
    
    Use headphones during development to avoid self-hearing.
    Real AEC (WebRTC AEC3) will be implemented in M8+.
    """
    
    def __init__(self):
        self._tracer = get_tracer("aec")
        self._tracer.info("AEC stub initialized - using pass-through. Use headphones!")
    
    def process(self, mic_chunk: np.ndarray, ref_chunk: Optional[np.ndarray] = None) -> np.ndarray:
        """Process audio chunk. Currently returns input unchanged."""
        return mic_chunk

__all__ = ["AECStub"]
