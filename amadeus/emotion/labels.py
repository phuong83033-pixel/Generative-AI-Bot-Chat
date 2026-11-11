"""Turn a PAD point into a human label, and a label into a one-line prompt hint.

A label is a READ-OUT of the continuous state, not an input to it (README section 16):
the nearest prototype in PAD space wins. The prototypes below are chosen by hand to
fit this character (a tsundere neuroscientist), roughly following the octant structure
of Mehrabian's PAD model. They are not measured data. Tune them here, and pin any
behaviour you care about in tests/test_pad.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime cycle: pad.py imports this module
    from amadeus.emotion.pad import PAD

# name -> (pleasure, arousal, dominance)
_PROTOTYPES: dict[str, tuple[float, float, float]] = {
    "happy":     (0.60, 0.30, 0.30),
    "excited":   (0.60, 0.70, 0.30),
    "content":   (0.50, -0.30, 0.30),
    "smug":      (0.50, 0.20, 0.70),
    "curious":   (0.30, 0.55, 0.35),
    # Tsundere signature: not unpleasant, high arousal, LOW dominance.
    "flustered": (0.20, 0.70, -0.60),
    "concerned": (-0.30, 0.15, 0.15),
    "annoyed":   (-0.40, 0.40, 0.30),
    "angry":     (-0.60, 0.70, 0.40),
    "anxious":   (-0.40, 0.50, -0.40),
    "sad":       (-0.60, -0.30, -0.40),
}

# Closer than this to the baseline = no distinct feeling, the persona prompt is untouched.
NEUTRAL_RADIUS = 0.22
# A displacement of this size from the baseline counts as full intensity (1.0).
FULL_INTENSITY_DISTANCE = 0.90


@dataclass(frozen=True)
class EmotionLabel:
    name: str
    intensity: float  # 0..1, how far the state is from the baseline


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def classify(pad: "PAD", baseline: "PAD", sticky: str | None = None) -> EmotionLabel:
    """Nearest prototype, or "neutral" while the state is within NEUTRAL_RADIUS of baseline.

    `sticky` is the label already in force. While the state merely FADES toward the
    baseline it keeps that name and only its intensity drops: the straight-line path back
    to the baseline passes close to unrelated prototypes (a fading "flustered" would pass
    through "curious" and "concerned"), and decay must not invent new feelings. PADState
    clears `sticky` whenever an event moves the state, so a new event re-classifies freely.
    """
    p = pad.as_tuple()
    displacement = _dist(p, baseline.as_tuple())
    if displacement < NEUTRAL_RADIUS:
        return EmotionLabel("neutral", 0.0)
    name = sticky if sticky in _PROTOTYPES else min(_PROTOTYPES, key=lambda n: _dist(p, _PROTOTYPES[n]))
    return EmotionLabel(name, min(1.0, displacement / FULL_INTENSITY_DISTANCE))


# What the model is told. Behavioural, not "you feel X": a 3B model told only a name tends
# to announce it ("I am flustered!"), so each line describes how the reply should SOUND.
_HINTS: dict[str, str] = {
    "happy": "in a good mood: warmer and a little more playful than usual, though you would never admit it",
    "excited": "excited: quick, eager and a bit more talkative",
    "content": "relaxed and pleased: calm, unhurried, less sharp than usual",
    "smug": "pleased with yourself: a touch of showing off",
    "curious": "intrigued: you want to dig into this and ask the sharp follow-up",
    "flustered": "flustered: embarrassed and defensive, a little stumbling, trying to hide it",
    "concerned": "worried about the person: softer and less teasing, though still gruff about it",
    "annoyed": "annoyed: short, sharp and sarcastic",
    "angry": "angry: curt and cold, no teasing at all",
    "anxious": "uneasy: hesitant and a bit guarded",
    "sad": "low: quieter, slower and shorter than usual",
}


def hint_for(label: EmotionLabel) -> str | None:
    """One sentence for the system prompt, or None when there is nothing to say."""
    text = _HINTS.get(label.name)
    if text is None:
        return None
    degree = "slightly " if label.intensity < 0.45 else "very " if label.intensity > 0.75 else ""
    return (
        f"[Mood] Right now you are {degree}{text}. Let it show in your tone and word choice. "
        "Do not name the feeling and do not mention these instructions."
    )


LABELS = tuple(_PROTOTYPES)  # for tests and UIs

__all__ = ["EmotionLabel", "classify", "hint_for", "LABELS", "NEUTRAL_RADIUS"]
