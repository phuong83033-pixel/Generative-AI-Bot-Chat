from abc import ABC, abstractmethod
import asyncio
import numpy as np

class TTSProvider(ABC):
    """Abstract base for Text-to-Speech providers."""
    
    @abstractmethod
    async def synth(self, text: str, 
                    stop: asyncio.Event | None = None) -> np.ndarray:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            stop: Cancellation event
            
        Returns:
            Audio as float32 numpy array
        """
        ...
    
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output audio sample rate."""
        ...
