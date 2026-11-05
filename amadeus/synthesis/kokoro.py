"""Local text-to-speech with Kokoro-82M (ONNX, fp32, CPU).

WHY THIS REPLACES EDGE-TTS
--------------------------
Edge-TTS is a cloud service. Measured on this machine it accounts for 1909 ms of
the 2605 ms it takes to reach the first audible syllable - 73% of the turn - and
it needs network access. Kokoro runs entirely on the CPU through onnxruntime:

  * 0 MB of VRAM. The 698 MB of headroom on the 4 GB card is untouched.
  * No network. The agent works offline.
  * A CPU-only voice also leaves the GPU idle while the agent speaks, which is
    the window full-duplex listening will need later (README section 4).

The synth() contract is unchanged (text in, float32 array out), so nothing
upstream of the TTSProvider interface knows the voice was swapped.

WHY fp32 AND NOT THE int8 FILE
------------------------------
The original plan said "Kokoro-82M int8" and estimated a real-time factor of
~0.25. Measured on this i5-11400H, same 4.65 s sentence, 4 threads:

    int8 : 5242 ms  RTF 1.13   (slower than playback - and slower than Edge)
    fp32 : 1747 ms  RTF 0.38   (6 threads: 1668 ms, RTF 0.36)

onnxruntime's per-operator profile shows why: in the int8 graph `ConvInteger`
is 83% of runtime (4571 of 5504 ms over 87 calls). That kernel is a slow generic
path on CPU; the optimised MLAS float `Conv` it is replaced by in the fp32 graph
costs 972 ms. The int8 file also does not scale with threads (1 thread 6051 ms,
12 threads 6059 ms), which is the signature of a serial kernel. Quantising to
save compute made it ~3x slower here. fp32 costs 310 MB of RAM and nothing else.

Do not "optimise" back to int8 or fp16 without re-running the profile.

THREADS
-------
The session is built explicitly rather than through Kokoro(model, voices),
because that constructor calls resolve_providers() and would silently pick a
CUDA provider if onnxruntime-gpu were ever installed - spending VRAM this module
promises not to touch. Only CPUExecutionProvider is ever requested here.
6 threads (= physical cores) was the measured optimum; 12 hyperthreads was slower.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import numpy as np

from amadeus.synthesis.base import TTSProvider
from amadeus.trace import get_tracer

# Kokoro always renders at 24 kHz.
KOKORO_SAMPLE_RATE = 24000

# Characters that are spoken as noise or not at all by the phonemiser.
_MARKDOWN = re.compile(r"[*_`#>]+")
_SPACES = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    """Strip markdown the model sometimes emits, which would be read aloud."""
    return _SPACES.sub(" ", _MARKDOWN.sub("", text)).strip()


class KokoroTTS(TTSProvider):
    """Kokoro-82M via kokoro-onnx. CPU only."""

    def __init__(
        self,
        model_path: str,
        voices_path: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "en-us",
        threads: int = 6,
    ):
        self._voice = voice
        self._speed = speed
        self._lang = lang
        self._tracer = get_tracer("kokoro_tts")

        for label, path in (("model", model_path), ("voices", voices_path)):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"Kokoro {label} file not found: {path}. "
                    "See run.md section 3.7 for the download commands."
                )

        if "int8" in Path(model_path).name.lower():
            self._tracer.warning(
                "Kokoro int8 model selected. On this CPU it runs ~3x slower than fp32 "
                "(ConvInteger is 83% of runtime) and slower than real time. "
                "Use kokoro-v1.0.onnx - see the note at the top of kokoro.py."
            )

        # Imported here so --text runs with the Edge voice never pay for it.
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads

        with self._tracer.span("kokoro_load"):
            # CPU only, explicitly. Kokoro(model, voices) would resolve providers
            # itself and could pick CUDA, spending VRAM this module must not touch.
            session = ort.InferenceSession(
                model_path, sess_options=options, providers=["CPUExecutionProvider"]
            )
            self._kokoro = Kokoro.from_session(session, voices_path)

        available = self._kokoro.get_voices()
        if voice not in available:
            raise ValueError(
                f"Unknown Kokoro voice {voice!r}. Available: {', '.join(sorted(available))}"
            )
        # The first create() call pays a one-time ~200 ms (memory arena growth,
        # espeak init): measured 639 ms cold vs 444 ms warm for the same clause.
        # Pay it here at startup instead of on the user's first turn.
        with self._tracer.span("kokoro_warmup"):
            self._kokoro.create("Hello.", voice=voice, speed=speed, lang=lang)

        self._tracer.info(
            f"Kokoro ready: voice={voice} speed={speed} threads={threads} "
            f"({len(available)} voices)"
        )

    async def synth(self, text: str, stop: asyncio.Event | None = None) -> np.ndarray:
        """Synthesize speech. Returns an empty array when cancelled or on error."""
        spoken = clean_for_speech(text)
        empty = np.array([], dtype=np.float32)
        if not spoken:
            return empty
        if stop is not None and stop.is_set():
            return empty

        def _run() -> np.ndarray:
            samples, _rate = self._kokoro.create(
                spoken, voice=self._voice, speed=self._speed, lang=self._lang
            )
            return np.asarray(samples, dtype=np.float32)

        loop = asyncio.get_running_loop()
        try:
            with self._tracer.span("kokoro_synth"):
                audio = await loop.run_in_executor(None, _run)
        except Exception as e:
            self._tracer.error(f"Kokoro synthesis failed: {e}")
            return empty

        # create() is not interruptible, so honour a stop that arrived while it ran.
        if stop is not None and stop.is_set():
            return empty
        return audio

    @property
    def sample_rate(self) -> int:
        return KOKORO_SAMPLE_RATE


__all__ = ["KokoroTTS", "clean_for_speech", "KOKORO_SAMPLE_RATE"]
