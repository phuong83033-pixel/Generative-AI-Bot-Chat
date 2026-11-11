"""Audition Kokoro voices (and blends of voices) reading the same in-character lines.

Kokoro has 54 fixed voices and cannot clone one, but a voice is a style vector, so two
voices can be MIXED into a new one. That is the cheap way to get closer to a particular
character without training anything. This tool plays each candidate through the same
output path the app uses, so what you hear is what the app would say.

Run (headphones on; the audio plays through the Windows default output):
    venv/Scripts/python.exe scripts/audition_voices.py
    venv/Scripts/python.exe scripts/audition_voices.py --speed 1.08
    venv/Scripts/python.exe scripts/audition_voices.py --voices af_bella,bf_emma,af_bella:0.6+bf_emma:0.4
    venv/Scripts/python.exe scripts/audition_voices.py --save     # also writes WAVs to data/audition/

A voice spec is a name (af_bella) or a weighted mix (af_bella:0.6+bf_emma:0.4).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from amadeus.config import load_config  # noqa: E402

# Two lines in Kurisu's register: flustered-defensive, and confident-explaining. A voice that
# suits only one of them is a worse fit than one that carries both.
LINES = [
    "Hmph. Don't just say things like that out of nowhere. ...Fine. Thanks.",
    "Honestly, the math allows it. It's the paradoxes that give me a headache.",
]

# The current voice first, as the baseline to beat. Then the clearer/brisker female voices,
# then three blends. Nothing here was chosen by ear; that is what this tool is for.
DEFAULT_VOICES = [
    "af_heart",                      # current
    "af_bella", "af_nicole", "af_nova", "af_kore", "af_sarah", "af_aoede",
    "bf_emma", "bf_isabella",
    "af_bella:0.5+bf_emma:0.5",
    "af_nova:0.6+bf_emma:0.4",
    "af_heart:0.5+bf_emma:0.5",
]


def resolve(kokoro, spec: str):
    """Turn 'name' or 'a:0.6+b:0.4' into a voice name or a blended style array."""
    parts = []
    for piece in spec.split("+"):
        name, _, w = piece.partition(":")
        parts.append((name.strip(), float(w) if w else 1.0))
    available = set(kokoro.get_voices())
    for name, _ in parts:
        if name not in available:
            raise SystemExit(f"Unknown voice {name!r}. Available: {', '.join(sorted(available))}")
    if len(parts) == 1:
        return parts[0][0]
    total = sum(w for _, w in parts)
    return sum(kokoro.get_voice_style(n) * (w / total) for n, w in parts)


def save_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


async def main(voices: list[str], speed: float, save: bool, no_play: bool) -> None:
    from kokoro_onnx import Kokoro

    cfg = load_config()
    print("Loading Kokoro ...", flush=True)
    kokoro = Kokoro(str(PROJECT_ROOT / cfg.tts.model_path), str(PROJECT_ROOT / cfg.tts.voices_path))

    print(f"Synthesising {len(voices)} voices x {len(LINES)} lines at speed {speed} ...", flush=True)
    clips: list[tuple[str, list[np.ndarray]]] = []
    rate = 24000
    for spec in voices:
        voice = resolve(kokoro, spec)
        parts = []
        for line in LINES:
            samples, rate = kokoro.create(line, voice=voice, speed=speed, lang="en-us")
            parts.append(np.asarray(samples, dtype=np.float32))
        clips.append((spec, parts))
        if save:
            tag = spec.replace(":", "_").replace("+", "-")
            save_wav(PROJECT_ROOT / "data" / "audition" / f"{tag}.wav",
                     np.concatenate([p for part in parts for p in (part, np.zeros(int(rate * 0.4), np.float32))]), rate)

    if save:
        print(f"Saved WAVs to {PROJECT_ROOT / 'data' / 'audition'}")
    if no_play:
        print("(--no-play: not playing)")
        return

    from amadeus.synthesis.audio_out import StreamingAudioPlayer

    player = StreamingAudioPlayer(sample_rate=rate)
    print("\nListen and note a score from 1 to 5 for each. Starting in 2 seconds.\n", flush=True)
    await asyncio.sleep(2)
    try:
        for i, (spec, parts) in enumerate(clips, 1):
            print(f"[{i:>2}/{len(clips)}] {spec}", flush=True)
            for part in parts:
                await player.play(part)
                await player.drain()
                await asyncio.sleep(0.5)
            await asyncio.sleep(1.2)
    finally:
        player.close()
    print("\nDone. Tell me the numbers of the ones you liked best (and what is wrong with the rest).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES), help="comma-separated voice specs")
    ap.add_argument("--speed", type=float, default=1.0, help="speaking speed (1.0 normal)")
    ap.add_argument("--save", action="store_true", help="also write WAV files to data/audition/")
    ap.add_argument("--no-play", action="store_true", help="synthesise only (for checking the script)")
    a = ap.parse_args()
    asyncio.run(main([v.strip() for v in a.voices.split(",") if v.strip()], a.speed, a.save, a.no_play))
