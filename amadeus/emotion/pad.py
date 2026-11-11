"""PAD emotional state: Pleasure, Arousal, Dominance, each in [-1, 1].

Design rule from the plan (README section 15): emotion is a DELTA applied to a
continuously evolving state, never a label that is switched on and off.

    appraisal -> ΔPAD -> state.apply(Δ) -> label

Between events the state relaxes toward a personality BASELINE with a per-axis
half-life, so a compliment fades over a minute or two instead of vanishing on the next
turn or sticking forever. Time is read from an injectable clock, which is what makes the
decay testable without sleeping.

The three axes (Mehrabian):
    pleasure   unpleasant (-1) .. pleasant (+1)
    arousal    calm / sleepy (-1) .. excited / agitated (+1)
    dominance  submissive / flustered (-1) .. in control / confident (+1)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

from amadeus.emotion.labels import NEUTRAL_RADIUS, EmotionLabel, classify

# A new event must be at least this fraction as strong as the mood already in force to rename it.
LABEL_TAKEOVER = 0.6


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class PAD:
    """A point (or a delta) in PAD space."""

    pleasure: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    def __add__(self, other: "PAD") -> "PAD":
        return PAD(
            self.pleasure + other.pleasure,
            self.arousal + other.arousal,
            self.dominance + other.dominance,
        )

    def scaled(self, k: float) -> "PAD":
        return PAD(self.pleasure * k, self.arousal * k, self.dominance * k)

    def clamped(self, lo: float = -1.0, hi: float = 1.0) -> "PAD":
        return PAD(_clamp(self.pleasure, lo, hi), _clamp(self.arousal, lo, hi),
                   _clamp(self.dominance, lo, hi))

    def distance(self, other: "PAD") -> float:
        return math.sqrt(
            (self.pleasure - other.pleasure) ** 2
            + (self.arousal - other.arousal) ** 2
            + (self.dominance - other.dominance) ** 2
        )

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.pleasure, self.arousal, self.dominance)

    def __str__(self) -> str:
        return f"P{self.pleasure:+.2f} A{self.arousal:+.2f} D{self.dominance:+.2f}"


class PADState:
    """The current emotional state, decaying toward `baseline`.

    `half_life_s` is the time for the distance to the baseline to halve, per axis.
    0 means "snap back immediately"; math.inf means "never decays".
    """

    def __init__(
        self,
        baseline: PAD = PAD(),
        half_life_s: PAD = PAD(90.0, 45.0, 120.0),
        clock: Callable[[], float] = time.monotonic,
        initial: PAD | None = None,
    ):
        self.baseline = baseline
        self.half_life_s = half_life_s
        self._clock = clock
        self._current = (initial if initial is not None else baseline).clamped()
        self._last = clock()
        # Label in force since the last event; see labels.classify(sticky=...).
        self._sticky: str | None = None

    @property
    def current(self) -> PAD:
        return self._current

    @staticmethod
    def _relax(x: float, base: float, dt: float, half_life: float) -> float:
        if half_life <= 0:
            return base
        if math.isinf(half_life):
            return x
        return base + (x - base) * 0.5 ** (dt / half_life)

    def tick(self) -> bool:
        """Decay toward the baseline up to now. Returns True if the state moved."""
        now = self._clock()
        dt = now - self._last
        self._last = now
        if dt <= 0:
            return False
        c, b, h = self._current, self.baseline, self.half_life_s
        new = PAD(
            self._relax(c.pleasure, b.pleasure, dt, h.pleasure),
            self._relax(c.arousal, b.arousal, dt, h.arousal),
            self._relax(c.dominance, b.dominance, dt, h.dominance),
        )
        moved = new != c
        self._current = new
        return moved

    @staticmethod
    def _push(x: float, dx: float) -> float:
        """Add dx to x with diminishing returns near the bounds.

        A delta is shrunk by the room left in its direction (never enlarged), so two
        arousing events in a row cannot pin an axis at +/-1: arousal .45 + .4 -> .67, not 0.85
        clamped to 1. A baseline that is not centred (dominance .25) keeps full-size deltas
        toward the far side, because min(1, room) caps the factor at 1.
        """
        room = 1.0 - x if dx >= 0 else 1.0 + x
        return x + dx * min(1.0, max(0.0, room))

    def apply(self, delta: PAD) -> PAD:
        """Decay up to now, THEN add the delta (so an old mood is not revived).

        The label is chosen from the EVENT, not from the accumulated state: a blend of two
        moods lands near an unrelated prototype (a flustered state plus an annoyed one
        looked 'curious'), whereas "what just happened" has an unambiguous reading. A new
        event renames the mood only if it is at least LABEL_TAKEOVER as strong as the
        mood already in force; a weaker one adds to that mood without renaming it.
        """
        self.tick()
        before = self._current
        in_force = before.distance(self.baseline)
        if in_force < NEUTRAL_RADIUS:
            self._sticky = None  # nothing in force any more

        self._current = PAD(
            self._push(before.pleasure, delta.pleasure),
            self._push(before.arousal, delta.arousal),
            self._push(before.dominance, delta.dominance),
        ).clamped()

        event = classify(self.baseline + delta, self.baseline)
        strength = delta.distance(PAD())
        if event.name != "neutral" and (self._sticky is None or strength >= LABEL_TAKEOVER * in_force):
            self._sticky = event.name
        return self._current

    def label(self) -> EmotionLabel:
        self.tick()
        lab = classify(self._current, self.baseline, sticky=self._sticky)
        self._sticky = lab.name if lab.name != "neutral" else None
        return lab


__all__ = ["PAD", "PADState"]
