"""Show the avatar without the LLM, the GPU or a microphone.

Drives the REAL pieces end to end: the emotion engine (appraisal -> PAD), the state machine,
Kokoro speech, and the player's loudness for lip-sync. Only the conversation is scripted.

    venv/Scripts/python.exe scripts/avatar_demo.py                 # scripted tour, silent
    venv/Scripts/python.exe scripts/avatar_demo.py --sound         # same, with audio (headphones)
    venv/Scripts/python.exe scripts/avatar_demo.py --loop          # repeat until Ctrl+C
    venv/Scripts/python.exe scripts/avatar_demo.py --hold "Hey Christina." --speak
                                    # hold the mood that sentence causes (for looking at one pose)
    venv/Scripts/python.exe scripts/avatar_demo.py --hold "You're really smart." --hold "Hey Christina." --hold-secs 12
                                    # cycle through several held moods

Then open  http://127.0.0.1:8765/   (add ?debug to see the raw numbers).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from amadeus.bus import EventBus  # noqa: E402
from amadeus.config import load_config  # noqa: E402
from amadeus.emotion.engine import EmotionEngine  # noqa: E402
from amadeus.expression.driver import AvatarDriver  # noqa: E402
from amadeus.expression.server import WebAvatarServer  # noqa: E402
from amadeus.state import StateManager, SystemState  # noqa: E402
from amadeus.synthesis.audio_out import NullPlayer, StreamingAudioPlayer  # noqa: E402
from amadeus.synthesis.kokoro import KokoroTTS  # noqa: E402

# (what the user says, what she answers). The answers are Kurisu-style lines, not LLM output.
TOUR = [
    ("Hello there.", "Hello. You're here early."),
    ("You're really smart.", "W-what? Don't just say things like that out of nowhere. ...Fine, thanks."),
    ("What do you think about time travel?", "Honestly? The math allows it. The paradoxes give me a headache."),
    ("Hey Christina.", "It's Kurisu. Not Christina. Say it again and I'm ignoring you."),
    ("Sorry about that.", "Hmph. ...Fine. I'll let it go this time."),
    ("I had a bad day today.", "That sounds rough. Do you want to talk about it?"),
    ("Do you like Dr Pepper?", "Dr Pepper, obviously. Don't judge me."),
]


async def say(app, line: str) -> None:
    """User speaks -> she thinks -> she speaks, with the same state transitions as the real loop."""
    snap = await app["engine"].on_user_utterance(line[0])
    print(f"  you > {line[0]}\n  [mood: {snap}]", flush=True)
    state = app["state"]
    state.transition(SystemState.LISTENING)
    await asyncio.sleep(0.8)
    state.transition(SystemState.THINKING)
    await asyncio.sleep(1.0)
    audio = await app["tts"].synth(line[1], None, app["engine"].tts_style())
    state.transition(SystemState.SPEAKING)
    print(f"  Kurisu > {line[1]}", flush=True)
    await app["player"].play(audio)
    await app["player"].drain()
    state.transition(SystemState.IDLE)


async def main(a: argparse.Namespace) -> None:
    cfg = load_config()
    bus = EventBus()
    bus.bind_loop()
    state = StateManager(bus)
    engine = EmotionEngine(cfg.emotion, bus=bus)
    tts = KokoroTTS(
        model_path=str(PROJECT_ROOT / cfg.tts.model_path), voices_path=str(PROJECT_ROOT / cfg.tts.voices_path),
        voice=cfg.tts.kokoro_voice, speed=cfg.tts.speed, threads=cfg.tts.threads,
    )
    player = (StreamingAudioPlayer(sample_rate=tts.sample_rate, bus=bus) if a.sound
              else NullPlayer(sample_rate=tts.sample_rate, bus=bus, lead_s=StreamingAudioPlayer.LEAD_S))

    web = WebAvatarServer(port=a.port)
    await web.start()
    driver = AvatarDriver(bus, [web], level_source=lambda: player.level, fps=cfg.avatar.fps)
    await driver.start()
    ticker_stop = asyncio.Event()
    ticker = asyncio.create_task(engine.run_ticker(ticker_stop))

    print(f"\nAvatar at {web.url}   (add ?debug for raw values)  - Ctrl+C to stop", flush=True)
    for _ in range(a.wait * 10):          # give a browser time to connect before the show starts
        if web.client_count:
            break
        await asyncio.sleep(0.1)
    app = {"engine": engine, "state": state, "tts": tts, "player": player}

    try:
        if a.hold:
            while True:
                for text in a.hold:
                    engine.reset()                      # a clean pose, not a blend with the last one
                    snap = await engine.on_user_utterance(text)
                    print(f"Holding {a.hold_secs:.0f} s: {text!r} -> [mood: {snap}]", flush=True)
                    end = asyncio.get_running_loop().time() + a.hold_secs
                    while asyncio.get_running_loop().time() < end:
                        if a.speak:
                            state.transition(SystemState.THINKING)
                            state.transition(SystemState.SPEAKING)
                            audio = await tts.synth("Honestly, the math allows it. It's the paradoxes that give me a headache.",
                                                    None, engine.tts_style())
                            await player.play(audio)
                            await player.drain()
                            state.transition(SystemState.IDLE)
                            await asyncio.sleep(0.4)
                        else:
                            await asyncio.sleep(0.5)
                        await engine.on_user_utterance(text)   # keep it from fading while we look
        else:
            while True:
                for line in TOUR:
                    await say(app, line)
                    await asyncio.sleep(2.5)
                if not a.loop:
                    break
            print("\nTour finished; letting the mood fade for 20 s ...", flush=True)
            await asyncio.sleep(20)
    finally:
        ticker_stop.set()
        await asyncio.gather(ticker, return_exceptions=True)
        await driver.stop()
        player.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sound", action="store_true", help="play the speech aloud (headphones)")
    ap.add_argument("--loop", action="store_true", help="repeat the tour until Ctrl+C")
    ap.add_argument("--hold", metavar="TEXT", action="append", help="hold the mood this user sentence causes (repeatable: cycles)")
    ap.add_argument("--hold-secs", type=float, default=10.0, help="how long to hold each --hold mood")
    ap.add_argument("--speak", action="store_true", help="with --hold: keep talking so the mouth moves")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--wait", type=int, default=30, help="seconds to wait for a browser before starting")
    args = ap.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
