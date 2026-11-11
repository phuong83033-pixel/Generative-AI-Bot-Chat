"""EmotionEngine: owns the PAD state and exposes it to the rest of the system.

    user text --appraise--> ΔPAD --apply--> state --> label
                                              |--> prompt_hint()   (what the LLM is told)
                                              |--> tts_style()     (how the voice is driven)
                                              `--> "EmotionChanged" on the bus  (the avatar, M14)

Everything that CONSUMES emotion (prompt, voice, avatar) reads it from here or from the
bus event; nothing consumes it by reaching into PADState. That keeps the avatar an output
layer, as README section 26 requires.

The neutral guarantee: when the state is within NEUTRAL_RADIUS of the baseline,
prompt_hint() is None and tts_style() is exactly speed 1.0. The persona and the voice
behave byte-for-byte as they did before this milestone until she actually feels something.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

from amadeus.bus import EventBus
from amadeus.config import EmotionConfig
from amadeus.emotion.appraisal import Appraisal, appraise
from amadeus.emotion.labels import hint_for
from amadeus.emotion.pad import PAD, PADState
from amadeus.synthesis.base import TtsStyle
from amadeus.trace import get_tracer

# Below this per-axis change a tick is not worth publishing (the avatar would redraw
# an invisible difference ~once a second for nothing).
_PUBLISH_EPSILON = 0.01


@dataclass(frozen=True)
class EmotionSnapshot:
    """What the rest of the system sees. Immutable; safe to hand to any subscriber."""

    pad: PAD
    label: str
    intensity: float
    baseline: PAD
    reasons: tuple[str, ...] = ()

    def __str__(self) -> str:
        why = f" | why: {', '.join(self.reasons)}" if self.reasons else ""
        return f"{self.label} {self.intensity:.2f} | {self.pad}{why}"


class EmotionEngine:
    def __init__(self, cfg: EmotionConfig, bus: EventBus | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self._cfg = cfg
        self._bus = bus
        self._clock = clock
        self._tracer = get_tracer("emotion")
        self.state = self._fresh_state()
        self._last_published: PAD | None = None

    def _fresh_state(self) -> PADState:
        return PADState(
            baseline=PAD(*self._cfg.baseline),
            half_life_s=PAD(*self._cfg.half_life_s),
            clock=self._clock,
        )

    def reset(self) -> None:
        """Back to the resting mood immediately (tests, demos, a new session)."""
        self.state = self._fresh_state()

    # ----------------------------------------------------------------- reading

    def snapshot(self, reasons: tuple[str, ...] = ()) -> EmotionSnapshot:
        label = self.state.label()  # also decays up to now
        return EmotionSnapshot(
            pad=self.state.current,
            label=label.name,
            intensity=label.intensity,
            baseline=self.state.baseline,
            reasons=reasons,
        )

    def prompt_hint(self) -> str | None:
        """System-prompt line for the current mood, or None while neutral / too faint."""
        label = self.state.label()
        if label.intensity < self._cfg.hint_min_intensity:
            return None
        return hint_for(label)

    def tts_style(self) -> TtsStyle:
        """Voice style for the current mood. Speed only: Kokoro exposes no pitch or energy.

        Driven by the DEVIATION from baseline, so a neutral state is exactly 1.0.
        """
        self.state.tick()
        c, b, cfg = self.state.current, self.state.baseline, self._cfg
        speed = (1.0
                 + cfg.speed_per_arousal * (c.arousal - b.arousal)
                 + cfg.speed_per_pleasure * (c.pleasure - b.pleasure))
        return TtsStyle(speed=max(cfg.speed_min, min(cfg.speed_max, speed)))

    # ----------------------------------------------------------------- updating

    async def on_user_utterance(self, text: str) -> EmotionSnapshot:
        """Appraise what the user said, move the state, announce it."""
        appraisal: Appraisal = appraise(text, self._cfg.appraisal_strength)
        if appraisal.tags:
            self.state.apply(appraisal.delta)
            self._tracer.info(f"Appraisal {appraisal.tags} -> Δ {appraisal.delta}")
        snap = self.snapshot(appraisal.tags)
        self._tracer.info(f"Mood: {snap}")
        await self._publish(snap)
        return snap

    async def _publish(self, snap: EmotionSnapshot) -> None:
        self._last_published = snap.pad
        if self._bus is not None:
            await self._bus.publish("EmotionChanged", snap)

    async def run_ticker(self, stop: asyncio.Event) -> None:
        """Let the mood fade while nobody is talking, and tell the avatar about it.

        A turn-driven engine would only decay when the user next speaks, leaving a
        flustered face frozen on screen for as long as the room stays quiet.
        """
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._cfg.tick_interval_s)
                return
            except asyncio.TimeoutError:
                pass
            snap = self.snapshot()
            last = self._last_published
            if last is None or max(
                abs(a - b) for a, b in zip(snap.pad.as_tuple(), last.as_tuple())
            ) >= _PUBLISH_EPSILON:
                await self._publish(snap)


__all__ = ["EmotionEngine", "EmotionSnapshot"]
