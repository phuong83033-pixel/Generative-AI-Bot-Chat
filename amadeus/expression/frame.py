"""ExpressionFrame and ExpressionModel: what the face should be doing right now.

This file knows nothing about renderers. It turns three inputs

    emotion  (PAD point + label, from "EmotionChanged")
    state    (IDLE / LISTENING / THINKING / SPEAKING, from "StateChanged")
    level    (loudness of the audio actually leaving the speakers, for lip-sync)

into one ExpressionFrame of normalised face parameters, 30 times a second. Sinks (the
browser preview, VTube Studio) then translate that frame into whatever their renderer
needs. Keeping the maths here means it can be tested without a window or a socket, and
a third renderer costs one small sink, not a second copy of the emotion mapping.

Conventions (renderer-independent, and the only place they are defined):
    mouth_open   0 closed .. 1 wide open
    mouth_smile  -1 frown .. 0 neutral .. +1 smile
    eye_open     0 closed .. 1 wide open (the resting value is ~0.85, not 1)
    brow_raise   -1 lowered .. +1 raised
    brow_inner   -1 furrowed (angry) .. +1 inner ends raised (worried)
    blush        0 .. 1
    head_yaw / head_pitch / head_roll   degrees; yaw + = her right, pitch + = up, roll + = tilt to her right
    gaze_x / gaze_y   -1 .. 1; x + = her right, y + = up

Emotion is read as the DEVIATION from her baseline, so a resting face is a neutral face
even though her resting dominance is +0.25.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class ExpressionFrame:
    mouth_open: float = 0.0
    mouth_smile: float = 0.0
    eye_open: float = 0.85
    brow_raise: float = 0.0
    brow_inner: float = 0.0
    blush: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    label: str = "neutral"
    state: str = "IDLE"

    def to_dict(self) -> dict:
        return asdict(self)


# Smoothing time constants, seconds. Shorter = snappier. Mouth is fast because lip-sync that
# lags the audio by more than ~80 ms is visibly wrong; mood is slow because a face that snaps
# between expressions looks like a slideshow, not a person.
TAU_MOOD = 0.35
TAU_GAZE = 0.12
TAU_MOUTH_ATTACK = 0.03
TAU_MOUTH_RELEASE = 0.07

# Speech loudness (RMS of one output block, full scale = 1.0) that opens the mouth fully.
# Kokoro speech is ~0.1 RMS on average with syllable peaks near 0.2.
MOUTH_FULL_LEVEL = 0.12

BLINK_CLOSE_S = 0.06
BLINK_OPEN_S = 0.10

_WORRIED = frozenset({"concerned", "sad", "anxious"})
_ANGRY = frozenset({"annoyed", "angry"})

# What each named mood LOOKS like at full strength. The face is a blend of this and the raw PAD
# reading, weighted by the label's intensity. Why both: the label follows the most recent EVENT
# (she was just called Christina: annoyed) while PAD is an accumulation that still holds the
# compliment from 14 s ago (blush, a small smile). Driven by PAD alone, the face looked
# embarrassed while the label said annoyed. PAD still supplies the continuous shading.
_PRESETS: dict[str, dict[str, float]] = {
    "flustered": dict(smile=0.30, brow_raise=0.50, brow_inner=0.15, eye=1.00, blush=1.00),
    "annoyed":   dict(smile=-0.50, brow_raise=-0.20, brow_inner=-0.70, eye=0.75, blush=0.00),
    "angry":     dict(smile=-0.70, brow_raise=-0.40, brow_inner=-1.00, eye=0.65, blush=0.00),
    "happy":     dict(smile=0.70, brow_raise=0.30, brow_inner=0.00, eye=0.90, blush=0.15),
    "excited":   dict(smile=0.80, brow_raise=0.60, brow_inner=0.00, eye=1.00, blush=0.10),
    "content":   dict(smile=0.50, brow_raise=0.00, brow_inner=0.00, eye=0.75, blush=0.00),
    "smug":      dict(smile=0.50, brow_raise=0.10, brow_inner=-0.10, eye=0.70, blush=0.00),
    "curious":   dict(smile=0.10, brow_raise=0.50, brow_inner=0.10, eye=0.95, blush=0.00),
    "concerned": dict(smile=-0.30, brow_raise=0.20, brow_inner=0.80, eye=0.85, blush=0.00),
    "anxious":   dict(smile=-0.30, brow_raise=0.40, brow_inner=0.90, eye=1.00, blush=0.00),
    "sad":       dict(smile=-0.60, brow_raise=-0.10, brow_inner=0.80, eye=0.60, blush=0.00),
}
_PRESET_MAX_WEIGHT = 0.75   # at full intensity the preset is 75% of the face, PAD the other 25%


@dataclass
class _Inputs:
    pleasure: float = 0.0   # all as DEVIATION from the baseline
    arousal: float = 0.0
    dominance: float = 0.0
    label: str = "neutral"
    intensity: float = 0.0
    state: str = "IDLE"
    level: float = 0.0


@dataclass
class _Smoothed:
    smile: float = 0.0
    brow_raise: float = 0.0
    brow_inner: float = 0.0
    eye: float = 0.85
    blush: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    mouth: float = 0.0


class ExpressionModel:
    """Stateful: smoothing, blinking and idle motion need memory. step() is cheap (~20 float ops)."""

    def __init__(self, rng: random.Random | None = None):
        self._rng = rng or random.Random()
        self._in = _Inputs()
        self._s = _Smoothed()
        self._t = 0.0
        self._next_blink = self._rng.uniform(1.5, 4.0)
        self._blink_start: float | None = None
        self._saccade_until = 0.0
        self._saccade = (0.0, 0.0)

    # ------------------------------------------------------------------- inputs

    def set_emotion(self, pad, baseline, label: str, intensity: float) -> None:
        i = self._in
        i.pleasure = pad.pleasure - baseline.pleasure
        i.arousal = pad.arousal - baseline.arousal
        i.dominance = pad.dominance - baseline.dominance
        i.label, i.intensity = label, intensity

    def set_state(self, state: str) -> None:
        self._in.state = state

    def set_level(self, level: float) -> None:
        self._in.level = max(0.0, level)

    # ------------------------------------------------------------------ targets

    def _targets(self) -> dict[str, float]:
        """The face the current emotion and state call for, before smoothing."""
        i = self._in
        dP, dA, dD = i.pleasure, i.arousal, i.dominance

        # Negative pleasure splits into anger (brows pulled down) and worry (inner ends raised).
        # Dominance alone cannot tell them apart for HER: her resting dominance is high, so a
        # caring "concerned" reads as dominant and came out furrowed. The label, which is chosen
        # from the event, settles it; dominance only decides labels outside these two sets.
        unpleasant = _clamp(-dP * 1.3, 0.0, 1.0)
        if i.label in _WORRIED:
            anger_w = 0.1
        elif i.label in _ANGRY:
            anger_w = 0.9
        else:
            anger_w = _clamp(0.5 + dD, 0.0, 1.0)
        anger, worry = unpleasant * anger_w, unpleasant * (1.0 - anger_w)

        smile = _clamp(1.4 * dP, -1.0, 1.0)
        brow_raise = _clamp(0.8 * dA + 0.4 * worry - 0.5 * anger, -1.0, 1.0)
        brow_inner = _clamp(1.6 * worry - anger, -1.0, 1.0)
        # Wide-eyed when aroused, narrowed by anger, drooping when low.
        sad = unpleasant * (1.0 - _clamp(dA, 0.0, 1.0))
        eye = _clamp(0.85 + 0.2 * dA - 0.15 * anger - 0.2 * sad, 0.45, 1.0)
        # Embarrassment is high arousal AND low dominance: the tsundere signature.
        blush = _clamp((dA - dD - 0.15) * 0.9, 0.0, 1.0)

        preset = _PRESETS.get(i.label)
        if preset is not None and i.intensity > 0.0:
            w = _clamp(i.intensity * 1.2, 0.0, 1.0) * _PRESET_MAX_WEIGHT
            smile = smile * (1 - w) + preset["smile"] * w
            brow_raise = brow_raise * (1 - w) + preset["brow_raise"] * w
            brow_inner = brow_inner * (1 - w) + preset["brow_inner"] * w
            eye = eye * (1 - w) + preset["eye"] * w
            blush = blush * (1 - w) + preset["blush"] * w

        pitch = 4.0 * dA - 10.0 * max(0.0, -dD)            # submissive = chin down
        roll = 6.0 * i.intensity if i.label == "curious" else (3.0 * worry)
        gaze_x, gaze_y = 0.0, 0.0
        if blush > 0.2:                                      # too embarrassed to look at you
            gaze_x, gaze_y = 0.45 * blush, -0.2 * blush

        # System state overrides what the face is DOING without changing how it FEELS.
        if i.state == "LISTENING":
            brow_raise += 0.15
            roll += 3.0
        elif i.state == "THINKING":
            gaze_x, gaze_y = -0.35, 0.35                     # looks up and away, as people do
            brow_raise += 0.10

        yaw = 8.0 * gaze_x                                   # the head follows the eyes a little
        return dict(smile=smile, brow_raise=_clamp(brow_raise, -1, 1), brow_inner=brow_inner, eye=eye,
                    blush=blush, pitch=pitch, roll=roll, yaw=yaw, gaze_x=gaze_x, gaze_y=gaze_y)

    def _mouth_target(self) -> float:
        if self._in.level <= 0.0:
            return 0.0
        return _clamp((self._in.level / MOUTH_FULL_LEVEL) ** 0.7, 0.0, 1.0)

    # --------------------------------------------------------------------- step

    @staticmethod
    def _approach(cur: float, target: float, dt: float, tau: float) -> float:
        return cur + (target - cur) * (1.0 - math.exp(-dt / tau))

    def _blink_factor(self) -> float:
        """1 = open, 0 = closed. Schedules the next blink (and sometimes a quick second one)."""
        if self._blink_start is None:
            if self._t >= self._next_blink:
                self._blink_start = self._t
            else:
                return 1.0
        e = self._t - self._blink_start
        if e < BLINK_CLOSE_S:
            return 1.0 - e / BLINK_CLOSE_S
        if e < BLINK_CLOSE_S + BLINK_OPEN_S:
            return (e - BLINK_CLOSE_S) / BLINK_OPEN_S
        self._blink_start = None
        gap = 0.25 if self._rng.random() < 0.15 else self._rng.uniform(2.5, 6.0) / (1.0 + 0.6 * max(0.0, self._in.arousal))
        self._next_blink = self._t + gap
        return 1.0

    def step(self, dt: float) -> ExpressionFrame:
        """Advance `dt` seconds and return the frame to draw."""
        dt = _clamp(dt, 0.0, 0.25)  # a stalled loop must not make the face jump
        self._t += dt
        s, tg = self._s, self._targets()
        a = self._approach

        s.smile = a(s.smile, tg["smile"], dt, TAU_MOOD)
        s.brow_raise = a(s.brow_raise, tg["brow_raise"], dt, TAU_MOOD)
        s.brow_inner = a(s.brow_inner, tg["brow_inner"], dt, TAU_MOOD)
        s.eye = a(s.eye, tg["eye"], dt, TAU_MOOD)
        s.blush = a(s.blush, tg["blush"], dt, TAU_MOOD * 1.5)
        s.pitch = a(s.pitch, tg["pitch"], dt, TAU_MOOD)
        s.roll = a(s.roll, tg["roll"], dt, TAU_MOOD)
        s.yaw = a(s.yaw, tg["yaw"], dt, TAU_MOOD)

        # Gaze: a small random saccade every second or two keeps a speaking face from staring.
        if self._t >= self._saccade_until:
            self._saccade = (self._rng.uniform(-0.12, 0.12), self._rng.uniform(-0.08, 0.08))
            self._saccade_until = self._t + self._rng.uniform(0.8, 2.2)
        s.gaze_x = a(s.gaze_x, tg["gaze_x"] + self._saccade[0], dt, TAU_GAZE)
        s.gaze_y = a(s.gaze_y, tg["gaze_y"] + self._saccade[1], dt, TAU_GAZE)

        mt = self._mouth_target()
        s.mouth = a(s.mouth, mt, dt, TAU_MOUTH_ATTACK if mt > s.mouth else TAU_MOUTH_RELEASE)

        # Idle life: slow sway and breathing, so a silent avatar is not a still photograph.
        sway = 2.0 * math.sin(2 * math.pi * self._t / 7.0)
        tilt = 1.0 * math.sin(2 * math.pi * self._t / 9.0 + 1.0)
        breathe = 0.6 * math.sin(2 * math.pi * self._t / 4.0)

        return ExpressionFrame(
            mouth_open=round(s.mouth, 4),
            mouth_smile=round(s.smile, 4),
            eye_open=round(s.eye * self._blink_factor(), 4),
            brow_raise=round(s.brow_raise, 4),
            brow_inner=round(s.brow_inner, 4),
            blush=round(s.blush, 4),
            head_yaw=round(s.yaw + sway, 3),
            head_pitch=round(s.pitch + breathe, 3),
            head_roll=round(s.roll + tilt, 3),
            gaze_x=round(_clamp(s.gaze_x, -1, 1), 4),
            gaze_y=round(_clamp(s.gaze_y, -1, 1), 4),
            label=self._in.label,
            state=self._in.state,
        )


__all__ = ["ExpressionFrame", "ExpressionModel", "MOUTH_FULL_LEVEL"]
