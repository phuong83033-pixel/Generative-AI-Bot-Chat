"""Amadeus entrypoint - the M7 voice MVP turn loop.

Wires the pipeline end to end:

    mic -> AEC -> VAD -> Whisper -> PromptBuilder -> Qwen
        -> ClauseChunker -> Edge-TTS -> speakers

The loop is deliberately turn-based for M7. Proactive initiative (README section
19) and full-duplex backchannelling (section 25) are M12/M13 and need state
machine changes that do not exist yet - see KNOWN LIMITS at the bottom of this
docstring.

Two contracts this file must honour, both recorded in fix.md section 6:

  1. bus.bind_loop() is the FIRST statement inside the running loop. EventBus
     deliberately does not capture a loop at construction, and publish_sync()
     raises until it is bound.
  2. MicrophoneStream.start() must be awaited before stream() is iterated - the
     loop is captured in start().

Run:
    venv/Scripts/python.exe -m amadeus.main           # voice
    venv/Scripts/python.exe -m amadeus.main --text    # type instead of speak
    venv/Scripts/python.exe -m amadeus.main --turns 3 # exit after N turns

WEAR HEADPHONES. perception/aec.py is still a pass-through stub, so on speakers
the agent hears itself, trips its own VAD and answers itself (README section 24).

KNOWN LIMITS at M7:
  * No barge-in. The stop Event is plumbed through every stage but nothing sets
    it mid-turn yet; VAD is not listening during SPEAKING.
  * TTS is not streaming. synthesis/base.py returns one complete array per
    clause, so first-audio latency includes full synthesis of clause 1.
  * Edge-TTS requires network. It is the M7 bring-up voice; Kokoro replaces it
    at M9 (README section 28, Phase 2).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amadeus.bus import EventBus  # noqa: E402
from amadeus.config import AmadeusConfig, load_config  # noqa: E402
from amadeus.runtime.vram import log_vram, require_free  # noqa: E402
from amadeus.state import StateManager, SystemState  # noqa: E402
from amadeus.trace import LatencyLog, TurnTrace, get_tracer, setup_logging  # noqa: E402

# Measured floor: Whisper small 370 MB + Qwen2.5-3B Q4_K_M 2210 MB = 2580 MB.
# See README section 31.3.
MIN_FREE_VRAM_MB = 2600


class Amadeus:
    """Owns every component and runs the turn loop."""

    def __init__(self, cfg: AmadeusConfig, *, text_mode: bool = False):
        self.cfg = cfg
        self.text_mode = text_mode
        self.tracer = get_tracer("main")
        self.bus = EventBus()
        self.state = StateManager(self.bus)
        self.latency = LatencyLog()
        self.turn_id = 0
        self._shutdown = asyncio.Event()

        # Populated in setup()
        self.mic = None
        self.aec = None
        self.vad = None
        self.stt = None
        self.brain = None
        self.prompt = None
        self.chunker = None
        self.tts = None
        self.player = None

    # ------------------------------------------------------------------ setup

    async def setup(self) -> None:
        """Bind the loop, check VRAM, then load every model once."""
        # CONTRACT 1: must be first, before anything can publish_sync.
        self.bus.bind_loop()

        if not self.text_mode:
            require_free(MIN_FREE_VRAM_MB, label="startup")
        log_vram("startup")

        # Imports are local so --text mode does not pay for CUDA or PortAudio.
        from amadeus.cognition.brain import QwenProvider
        from amadeus.cognition.prompt import PromptBuilder
        from amadeus.synthesis.audio_out import AudioPlayer
        from amadeus.synthesis.chunker import ClauseChunker
        from amadeus.synthesis.edge import EdgeTTS

        self.prompt = PromptBuilder(system_prompt=self.cfg.persona.system_prompt)
        self.chunker = ClauseChunker(bus=self.bus)

        self.tracer.info("Loading LLM ...")
        self.brain = QwenProvider(
            model_path=str(PROJECT_ROOT / self.cfg.llm.model_path),
            n_ctx=self.cfg.llm.n_ctx,
            n_gpu_layers=self.cfg.llm.n_gpu_layers,
            temperature=self.cfg.llm.temperature,
            max_tokens=self.cfg.llm.max_tokens,
            top_p=self.cfg.llm.top_p,
            repeat_penalty=self.cfg.llm.repeat_penalty,
            bus=self.bus,
        )

        self.tts = EdgeTTS(
            voice=self.cfg.tts.voice,
            rate=self.cfg.tts.rate,
            output_sample_rate=self.cfg.tts.output_sample_rate,
        )
        self.player = AudioPlayer(
            sample_rate=self.tts.sample_rate,
            device=self.cfg.audio.output_device,
            bus=self.bus,
        )

        if not self.text_mode:
            from amadeus.perception.aec import AECStub
            from amadeus.perception.audio_in import MicrophoneStream
            from amadeus.perception.stt import WhisperSTT
            from amadeus.perception.vad import SileroVAD

            self.tracer.info("Loading STT ...")
            self.stt = WhisperSTT(
                model_size=self.cfg.stt.model_size,
                compute_type=self.cfg.stt.compute_type,
                device=self.cfg.stt.device,
                language=self.cfg.stt.language,
                bus=self.bus,
            )
            self.aec = AECStub()
            self.vad = SileroVAD(
                threshold=self.cfg.vad.threshold,
                min_speech_ms=self.cfg.vad.min_speech_ms,
                silence_timeout_ms=self.cfg.vad.silence_timeout_ms,
                pre_speech_pad_ms=self.cfg.vad.pre_speech_pad_ms,
                sample_rate=self.cfg.audio.sample_rate,
                bus=self.bus,
            )
            self.mic = MicrophoneStream(
                sample_rate=self.cfg.audio.sample_rate,
                channels=self.cfg.audio.channels,
                chunk_size=self.cfg.audio.chunk_size,
                device=self.cfg.audio.input_device,
                bus=self.bus,
            )
            # CONTRACT 2: start() binds the loop used by the PortAudio callback.
            await self.mic.start()

        log_vram("all models resident")
        self.tracer.info("Setup complete.")

    # ------------------------------------------------------------- perception

    async def _listen(self, trace: TurnTrace, stop: asyncio.Event) -> str | None:
        """Capture one user utterance and return its transcript."""
        if self.text_mode:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, lambda: input("\nyou > ").strip())
            trace.stamp("speech_end")
            trace.stamp("stt_done")
            return text or None

        self.state.transition(SystemState.LISTENING)
        print("\n[listening ...]", flush=True)

        audio = await self.vad.listen_until_speech_ends(self.mic, stop)
        trace.stamp("speech_end")

        if audio is None or audio.size == 0:
            return None

        # AEC is a pass-through stub at M7; the call site exists so the real
        # implementation drops in without touching the turn loop.
        audio = self.aec.process(audio)

        text = await self.stt.transcribe(audio, stop)
        ms = trace.stamp("stt_done")
        self.tracer.info(f"Transcript in {ms:.0f}ms: {text!r}")
        return text or None

    # -------------------------------------------------------------- cognition

    async def _respond(self, user_text: str, trace: TurnTrace, stop: asyncio.Event) -> str:
        """Generate, synthesize and speak a reply. Returns the full text."""
        messages = self.prompt.build(user_text)
        trace.stamp("prompt_built")

        token_stream = self._tapped_tokens(self.brain.stream(messages, stop), trace)

        spoken: list[str] = []
        async for clause in self.chunker.process(token_stream, stop):
            if stop.is_set():
                break
            if not trace.has("first_clause"):
                trace.stamp("first_clause")

            audio = await self.tts.synth(clause, stop)
            if not trace.has("tts_done"):
                trace.stamp("tts_done")

            if audio.size == 0:
                self.tracer.warning(f"TTS returned no audio for: {clause!r}")
                spoken.append(clause)
                continue

            if self.state.current is not SystemState.SPEAKING:
                self.state.transition(SystemState.SPEAKING)

            if not trace.has("first_audio_out"):
                ms = trace.stamp("first_audio_out")
                print(f"[first audio at {ms:.0f} ms]", flush=True)

            print(f"{self.cfg.persona.name} > {clause}", flush=True)
            await self.player.play(audio, stop)
            spoken.append(clause)

        return " ".join(spoken).strip()

    async def _tapped_tokens(self, stream, trace: TurnTrace):
        """Pass tokens through, stamping the first one."""
        async for token in stream:
            if not trace.has("first_token"):
                trace.stamp("first_token")
            yield token

    # -------------------------------------------------------------- turn loop

    async def run_turn(self) -> bool:
        """Run one full turn. Returns False when the session should end."""
        self.turn_id += 1
        trace = TurnTrace(self.turn_id)
        stop = asyncio.Event()

        try:
            user_text = await self._listen(trace, stop)
        except (EOFError, KeyboardInterrupt):
            return False

        if not user_text:
            self.state.transition(SystemState.IDLE)
            return True
        if user_text.lower() in {"exit", "quit", "goodbye", "stop"}:
            print("[exit requested]")
            return False

        if self.text_mode:
            print(f"you > {user_text}")

        self.state.transition(SystemState.THINKING)

        try:
            reply = await self._respond(user_text, trace, stop)
        except Exception:
            self.tracer.error(f"Turn {self.turn_id} failed", exc_info=True)
            self.state.transition(SystemState.IDLE)
            return True

        if reply:
            self.prompt.add_user_message(user_text)
            self.prompt.add_assistant_message(reply)

        trace.stamp("turn_done")
        self.state.transition(SystemState.IDLE)

        trace.log()
        self.latency.add(trace)
        print(trace.table(), flush=True)
        return True

    async def run(self, max_turns: int | None = None) -> None:
        await self.setup()

        print(f"\n{'=' * 62}")
        print(f"  {self.cfg.persona.name} ready"
              f"{'  [text mode]' if self.text_mode else '  [voice - WEAR HEADPHONES]'}")
        print(f"  say or type 'exit' to quit")
        print(f"{'=' * 62}")

        try:
            while not self._shutdown.is_set():
                if not await self.run_turn():
                    break
                if max_turns is not None and self.turn_id >= max_turns:
                    self.tracer.info(f"Reached --turns {max_turns}, stopping.")
                    break
        except KeyboardInterrupt:
            print("\n[interrupted]")
        finally:
            await self.shutdown()

    # --------------------------------------------------------------- teardown

    async def shutdown(self) -> None:
        self.tracer.info("Shutting down ...")
        if self.player is not None:
            self.player.stop_immediately()
        if self.mic is not None:
            await self.mic.stop()

        print(f"\n{'-' * 62}")
        print(self.latency.report(
            {
                "first_syllable_p50_ms": self.cfg.latency_targets.first_syllable_p50_ms,
                "first_syllable_p95_ms": self.cfg.latency_targets.first_syllable_p95_ms,
            }
        ))
        log_vram("shutdown")


# ------------------------------------------------------------------------ cli


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="amadeus", description="Amadeus voice companion (M7 MVP)"
    )
    p.add_argument(
        "--text",
        action="store_true",
        help="type instead of speaking; skips mic, VAD and STT entirely",
    )
    p.add_argument(
        "--turns", type=int, default=None, metavar="N", help="exit after N turns"
    )
    p.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    setup_logging(args.log_level or cfg.log_level, log_dir=PROJECT_ROOT / "data" / "logs")

    app = Amadeus(cfg, text_mode=args.text)
    try:
        asyncio.run(app.run(max_turns=args.turns))
    except KeyboardInterrupt:
        return 130
    except RuntimeError as e:
        # require_free() raises this with actionable guidance.
        print(f"\nStartup failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
