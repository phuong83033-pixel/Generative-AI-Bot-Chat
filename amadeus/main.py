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
import time
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

# Marks the end of a reply on the synthesiser -> player queue.
_DONE = object()


class Amadeus:
    """Owns every component and runs the turn loop."""

    def __init__(self, cfg: AmadeusConfig, *, text_mode: bool = False, mute: bool = False,
                 avatar: bool = False):
        self.cfg = cfg
        self.text_mode = text_mode
        # M14: an optional output layer. Either the CLI flag or avatar.enabled turns it on.
        self.avatar = avatar or cfg.avatar.enabled
        # mute: play through NullPlayer, which waits out each clip's real duration
        # without making sound. Timing-faithful, so benchmarks stay valid.
        self.mute = mute
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
        self.emotion = None          # M9
        self._emotion_task = None
        self.avatar_driver = None    # M14

    # ------------------------------------------------------------------ setup

    def _build_tts(self):
        """Construct the voice named by settings.yaml `tts.provider`.

        Kokoro is local (CPU, 0 MB VRAM, offline). Edge is the cloud fallback and
        costs roughly 1900 ms per turn, so it is no longer the default.
        """
        t = self.cfg.tts
        provider = t.provider.lower()

        if provider == "kokoro":
            from amadeus.synthesis.kokoro import KokoroTTS

            self.tracer.info("Loading TTS (Kokoro, local CPU) ...")
            return KokoroTTS(
                model_path=str(PROJECT_ROOT / t.model_path),
                voices_path=str(PROJECT_ROOT / t.voices_path),
                voice=t.kokoro_voice,
                speed=t.speed,
                threads=t.threads,
            )

        if provider == "edge":
            from amadeus.synthesis.edge import EdgeTTS

            self.tracer.info("Loading TTS (Edge, cloud) ...")
            return EdgeTTS(
                voice=t.voice, rate=t.rate, output_sample_rate=t.output_sample_rate
            )

        raise ValueError(f"Unknown tts.provider {t.provider!r}; expected 'kokoro' or 'edge'.")

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

        self.prompt = PromptBuilder(system_prompt=self.cfg.persona.system_prompt)

        if self.cfg.emotion.enabled:
            from amadeus.emotion.engine import EmotionEngine

            self.emotion = EmotionEngine(self.cfg.emotion, bus=self.bus)
            # Lets the mood fade while nobody is talking (the avatar watches it).
            self._emotion_task = asyncio.create_task(self.emotion.run_ticker(self._shutdown))

        self.chunker = ClauseChunker(
            bus=self.bus,
            first_clause_min=self.cfg.chunker.first_clause_min_chars,
            first_clause_max=self.cfg.chunker.first_clause_max_chars,
        )

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

        self.tts = self._build_tts()
        from amadeus.synthesis.audio_out import NullPlayer, StreamingAudioPlayer

        streaming = self.cfg.audio.streaming_output
        if self.mute:
            # Model whichever real player is configured, so benchmarks stay faithful.
            self.player = NullPlayer(
                sample_rate=self.tts.sample_rate,
                bus=self.bus,
                lead_s=StreamingAudioPlayer.LEAD_S if streaming else 0.0,
            )
        elif streaming:
            self.player = StreamingAudioPlayer(
                sample_rate=self.tts.sample_rate,
                device=self.cfg.audio.output_device,
                bus=self.bus,
            )
        else:
            self.player = AudioPlayer(
                sample_rate=self.tts.sample_rate,
                device=self.cfg.audio.output_device,
                bus=self.bus,
            )

        if self.avatar:
            await self._start_avatar()

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

    async def _start_avatar(self) -> None:
        """Start the avatar renderers. Optional by design: if anything here fails the voice
        assistant carries on without a face (README section 26: output layer, not core)."""
        a = self.cfg.avatar
        sinks = []
        try:
            if a.web:
                from amadeus.expression.server import WebAvatarServer

                web = WebAvatarServer(port=a.web_port)
                try:
                    await web.start()
                    sinks.append(web)
                    print(f"[avatar preview: {web.url}]", flush=True)
                except RuntimeError as e:
                    self.tracer.warning(str(e))
            if a.vtube:
                from amadeus.expression.vtube import VTubeStudioSink

                vts = VTubeStudioSink(
                    url=a.vtube_url,
                    plugin_name=a.plugin_name,
                    plugin_developer=a.plugin_developer,
                    token_path=PROJECT_ROOT / a.vtube_token_path,
                    hotkeys=a.vtube_hotkeys,
                    invert=a.vtube_invert,
                )
                await vts.start()  # connects in the background; VTube Studio may not be running yet
                sinks.append(vts)
                print(f"[avatar: VTube Studio at {a.vtube_url} (click ALLOW in its popup on first run)]", flush=True)
            if not sinks:
                self.tracer.warning("Avatar requested but no renderer could start; continuing without it.")
                return

            from amadeus.expression.driver import AvatarDriver

            self.avatar_driver = AvatarDriver(
                self.bus, sinks, level_source=lambda: self.player.level, fps=a.fps
            )
            await self.avatar_driver.start()
        except Exception:
            self.tracer.error("Avatar failed to start; continuing without it", exc_info=True)

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
        # Discard whatever was recorded while the agent was thinking or speaking.
        dropped = self.mic.flush()
        if dropped:
            self.tracer.info(f"Discarded {dropped} stale mic chunks ({dropped * 32} ms) before listening")
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
        """Generate, synthesize and speak a reply. Returns the full text.

        Two stages run concurrently, joined by a one-slot queue:

            LLM -> chunker -> TTS ──(audio)──> player

        The synthesiser works on clause N+1 while the player is still speaking
        clause N. The original loop did synth -> play -> synth, so every clause
        after the first was preceded by silence for as long as it took to
        synthesise (Kokoro: roughly 0.4 s per second of speech). Pipelining hides
        that cost for everything except the first clause.

        The queue holds one clip, so synthesis runs at most one clip ahead of
        playback. Deeper would waste CPU on audio that an interruption throws away.
        """
        # M9: the mood colours both what she says and how fast she says it. Both are
        # None / speed 1.0 while neutral, i.e. exactly the pre-emotion behaviour.
        hint = self.emotion.prompt_hint() if self.emotion else None
        style = self.emotion.tts_style() if self.emotion else None
        messages = self.prompt.build(user_text, mood_hint=hint)
        trace.stamp("prompt_built")

        token_stream = self._tapped_tokens(self.brain.stream(messages, stop), trace)
        ready: asyncio.Queue = asyncio.Queue(maxsize=1)
        spoken: list[str] = []
        last_play_end: float | None = None

        async def synthesize() -> None:
            async for clause in self.chunker.process(token_stream, stop):
                if stop.is_set():
                    break
                if not trace.has("first_clause"):
                    trace.stamp("first_clause")

                audio = await self.tts.synth(clause, stop, style)
                if not trace.has("tts_done"):
                    trace.stamp("tts_done")

                await ready.put((clause, audio))
            # Sentinel on normal completion only. If this task fails or is
            # cancelled, _respond cancels the player directly; a put() here could
            # block forever on a full queue with nobody left to drain it.
            await ready.put(_DONE)

        async def play() -> None:
            nonlocal last_play_end
            while True:
                item = await ready.get()
                if item is _DONE:
                    # play() returns slightly before a clip ends; wait for the real end.
                    await self.player.drain(stop)
                    return
                clause, audio = item

                if stop.is_set():
                    continue  # interrupted: drain without speaking
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
                if last_play_end is not None:
                    # Silence the listener actually hears between two clauses.
                    trace.add_gap((time.perf_counter() - last_play_end) * 1000)
                await self.player.play(audio, stop)
                # play() returns lead_s before the clip's audible end (0 for the
                # blocking player), so that is when the next clip must be queued by.
                last_play_end = time.perf_counter() + self.player.lead_s
                spoken.append(clause)

        synth_task = asyncio.create_task(synthesize())
        play_task = asyncio.create_task(play())
        try:
            await asyncio.gather(synth_task, play_task)
        except BaseException:
            # One side failed or the turn was cancelled: stop the other, wait for
            # both to unwind, then let the original error propagate.
            for task in (synth_task, play_task):
                task.cancel()
            await asyncio.gather(synth_task, play_task, return_exceptions=True)
            raise

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

        if self.emotion is not None:
            snap = await self.emotion.on_user_utterance(user_text)
            style = self.emotion.tts_style()
            print(f"[mood: {snap} | voice x{style.speed:.2f}]", flush=True)

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
        self._shutdown.set()  # ends the emotion ticker
        if self._emotion_task is not None:
            await asyncio.gather(self._emotion_task, return_exceptions=True)
        if self.avatar_driver is not None:
            await self.avatar_driver.stop()
        if self.player is not None:
            self.player.stop_immediately()
            self.player.close()
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
    p.add_argument(
        "--mute",
        action="store_true",
        help="no sound: playback is simulated by waiting out each clip's duration",
    )
    p.add_argument(
        "--avatar",
        action="store_true",
        help="show the avatar (browser preview at http://127.0.0.1:8765/; see settings.yaml avatar:)",
    )
    p.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    setup_logging(args.log_level or cfg.log_level, log_dir=PROJECT_ROOT / "data" / "logs")

    app = Amadeus(cfg, text_mode=args.text, mute=args.mute, avatar=args.avatar)
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
