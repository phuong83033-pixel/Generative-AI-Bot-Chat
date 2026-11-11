from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TtsStyle:
    """How the voice should be driven for this utterance (set by the emotion engine, M9).

    `speed` multiplies the provider's configured speed; 1.0 changes nothing. Only what a
    provider can actually honour belongs here: Kokoro takes a speed and nothing else, so
    pitch and energy are deliberately not fields yet. Add them when a provider can use them.
    """

    speed: float = 1.0


class TTSProvider(ABC):
    """Abstract base for Text-to-Speech providers."""

    @abstractmethod
    async def synth(self, text: str,
                    stop: asyncio.Event | None = None,
                    style: TtsStyle | None = None) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            stop: Cancellation event
            style: Optional emotional style. Providers may ignore what they cannot honour.

        Returns:
            Audio as float32 numpy array
        """
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output audio sample rate."""
        ...
