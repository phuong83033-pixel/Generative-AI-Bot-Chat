"""Play audio files one after another through the same output path the app uses.

Used for blind listening tests: the file names carry no hint about what they are.

    venv/Scripts/python.exe scripts/play_wavs.py data/audition/blind/reference_TARGET.wav data/audition/blind/clip_1.wav
    venv/Scripts/python.exe scripts/play_wavs.py data/audition/blind/*.wav --pause 1.5
    venv/Scripts/python.exe scripts/play_wavs.py --blind        # reference, then clip_1 .. clip_N

Any format PyAV can decode works (wav, mp3, flac ...). Wear headphones.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import av  # noqa: E402
import numpy as np  # noqa: E402

BLIND_DIR = PROJECT_ROOT / "data" / "audition" / "blind"


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Decode to float32 mono at the file's native rate."""
    container = av.open(str(path))
    stream = container.streams.audio[0]
    rate = stream.rate
    chunks = []
    for frame in container.decode(stream):
        raw = frame.to_ndarray()
        a = raw.astype(np.float32)
        if raw.dtype.kind == "i":  # integer PCM -> [-1, 1]
            a /= float(np.iinfo(raw.dtype).max + 1)
        chunks.append(a)
    container.close()
    a = np.concatenate(chunks, axis=1) if chunks[0].ndim == 2 else np.concatenate(chunks)
    if a.ndim == 2:
        a = a.mean(axis=0) if a.shape[0] <= 8 else a.mean(axis=1)
    return a.astype(np.float32), rate


def resample(a: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return a
    import torch
    import torchaudio

    return torchaudio.functional.resample(torch.from_numpy(a), src, dst).numpy()


async def main(paths: list[Path], pause: float, rate: int) -> None:
    from amadeus.synthesis.audio_out import StreamingAudioPlayer

    player = StreamingAudioPlayer(sample_rate=rate)
    try:
        for i, p in enumerate(paths, 1):
            a, sr = load_mono(p)
            print(f"[{i}/{len(paths)}] {p.name}", flush=True)
            await player.play(resample(a, sr, rate))
            await player.drain()
            await asyncio.sleep(pause)
    finally:
        player.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--blind", action="store_true", help="play reference_TARGET.wav then clip_1..N from data/audition/blind")
    ap.add_argument("--pause", type=float, default=1.2, help="seconds of silence between files")
    ap.add_argument("--rate", type=int, default=24000, help="output sample rate")
    a = ap.parse_args()

    files = list(a.files)
    if a.blind:
        ref = BLIND_DIR / "reference_TARGET.wav"
        clips = sorted(BLIND_DIR.glob("clip_*.wav"), key=lambda p: int(p.stem.split("_")[1]))
        files = ([ref] if ref.is_file() else []) + clips
    if not files:
        ap.error("give files to play, or --blind")
    missing = [f for f in files if not f.is_file()]
    if missing:
        ap.error(f"not found: {', '.join(map(str, missing))}")
    asyncio.run(main(files, a.pause, a.rate))
