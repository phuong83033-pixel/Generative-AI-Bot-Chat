"""Appraisal: what did the user just say, and how does it move Kurisu?  text -> ΔPAD.

Why rules and not the LLM
-------------------------
Appraisal sits on the critical path (README section 16: utterance -> appraisal -> PAD ->
prompt). A second LLM call would add a full generation's worth of latency before the
reply could even start, on a GPU that is already full. Rules cost microseconds, are
deterministic, and every decision comes back as a TAG, so "why is she annoyed?" has an
answer in the log (README section 19 asks for reason logging for the same reason).

The cost is coverage: a rule only fires on wording it knows. Unmatched input changes
nothing (the state just keeps decaying), which fails safe: she never becomes emotional
about text that was not understood. Rules are data (_RULES), so extending them is a
one-line change plus a test.

Deltas are written for THIS character. A compliment raises arousal and LOWERS dominance
(she gets flustered), an insult raises dominance (she pushes back).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from amadeus.emotion.pad import PAD

# A negator inside the matched phrase ("you're NOT stupid") or within a few words before
# it ("I don't think you're stupid") flips its meaning.
_NEGATORS = re.compile(
    r"\b(?:not|never|no|hardly|don't|doesn't|didn't|isn't|aren't|weren't|wasn't|ain't)\b"
)
_NEGATION_WINDOW_WORDS = 4

_MAX_AXIS_DELTA = 0.8  # one utterance cannot swing an axis further than this


@dataclass(frozen=True)
class Rule:
    tag: str
    pattern: re.Pattern
    delta: PAD
    # Delta when the match is negated. None = a negated match does nothing.
    negated_delta: PAD | None = None
    # False for phrases that CONTAIN a negator as part of their meaning ("I don't like you").
    negatable: bool = True


def _r(tag: str, pattern: str, delta: tuple[float, float, float],
       negated: tuple[float, float, float] | None = None, negatable: bool = True) -> Rule:
    return Rule(tag, re.compile(pattern), PAD(*delta), PAD(*negated) if negated else None, negatable)


_POSITIVE_TRAIT = (
    r"smart|brilliant|amazing|genius|great|clever|incredible|awesome|impressive|"
    r"talented|wonderful|fantastic|intelligent|the best"
)
_NEGATIVE_TRAIT = (
    r"stupid|dumb|useless|worthless|annoying|pathetic|terrible|awful|boring|"
    r"an idiot|a fool|ridiculous|a joke"
)

# Order does not matter: every rule is tried and all that match are summed.
_RULES: list[Rule] = [
    # Praise: pleased but flustered; dominance drops. Negated praise stings a little.
    _r("praise", rf"\byou(?:'re| are| were| aren't| weren't)? (?:not |never )?(?:really |so |very |extremely |just )?(?:{_POSITIVE_TRAIT})\b"
                 r"|\bgood (?:job|work)\b|\bwell done\b|\bnice work\b|\bi admire you\b",
       (0.20, 0.40, -0.50), negated=(-0.30, 0.30, 0.10)),
    # Affection: stronger version of the same flustered response.
    _r("affection", r"\bi (?:really |truly )?(?:love|adore|like) you\b|\bi miss you\b|\bmarry me\b|\bdate me\b"
                    r"|\byou(?:'re| are) (?:not )?(?:so |really |very )?(?:cute|beautiful|pretty|adorable|lovely)\b",
       (0.30, 0.50, -0.60), negated=(-0.10, 0.10, 0.10)),
    # Insult: she pushes back. Negated insult ("you're not stupid") is neutral.
    _r("insult", rf"\byou(?:'re| are| were| aren't| weren't)? (?:not |never )?(?:really |so |very |such )?(?:{_NEGATIVE_TRAIT})\b"
                 r"|\byou suck\b|\bstupid (?:ai|bot|machine)\b",
       (-0.40, 0.45, 0.25)),
    # Open hostility: harsher than an insult, so it reaches "angry" rather than "annoyed".
    _r("hostility", r"\bshut up\b|\bi (?:don't |do not )?hate you\b", (-0.55, 0.50, 0.30)),
    # "I don't like you" already contains its own negator, so it must not be flipped by it.
    _r("dislike", r"\bi (?:really |just )?(?:don't|do not) like you\b|\bi can't stand you\b",
       (-0.35, 0.35, 0.25), negatable=False),
    # The nickname she hates.
    _r("nickname_christina", r"\bchristina\b|\bzombie\b|\bassistant\b", (-0.35, 0.50, 0.30)),
    # Gratitude: pleased, slightly bashful.
    _r("gratitude", r"\bthank(?:s| you)\b|\bi appreciate\b|\bmuch appreciated\b", (0.20, -0.05, -0.10)),
    # Apology: she is in control and softens.
    _r("apology", r"\b(?:i'm |i am |so |really )?sorry\b|\bmy bad\b|\bforgive me\b|\bi apologi[sz]e\b",
       (0.15, -0.15, 0.20)),
    # The user is struggling: concern. Negation ("I'm not sad") cancels it.
    _r("user_distress",
       r"\bi(?:'m| am| feel| felt)(?: not)?(?: so| really| very| kind of| a bit)? "
       r"(?:sad|depressed|lonely|scared|anxious|stressed|upset|worried|exhausted|miserable|terrible|awful|down)\b"
       r"|\bbad day\b|\bi failed\b|\bi can't do this\b|\bi give up\b",
       (-0.25, 0.10, 0.15)),
    # Disagreement: she wants evidence.
    _r("challenge", r"\byou(?:'re| are) wrong\b|\bthat'?s (?:not true|wrong|nonsense)\b|\bi disagree\b"
                    r"|\bprove it\b|\bthat doesn'?t make sense\b|\bno you (?:don't|can't|didn't)\b",
       (-0.15, 0.40, 0.30)),
    # Her actual interests.
    _r("science_topic",
       r"\b(?:brain|memory|memories|neuro\w*|time ?travel|time machine|physics|quantum|experiment|"
       r"theory|paradox|hypothesis|consciousness|science|scientist|research)\b",
       (0.12, 0.30, 0.10)),
    _r("favorite_drink", r"\bdr\.? ?pepper\b", (0.30, 0.20, 0.05)),
    _r("humor", r"\b(?:haha+|lol|lmao|just kidding|i'?m joking|jk)\b", (0.35, 0.20, 0.00)),
    _r("greeting", r"^(?:hello|hi|hey|good (?:morning|afternoon|evening))\b", (0.10, 0.10, 0.00)),
    _r("farewell", r"\b(?:goodbye|bye|see you|good night|talk to you later)\b", (0.05, -0.20, 0.00)),
]


@dataclass(frozen=True)
class Appraisal:
    delta: PAD
    tags: tuple[str, ...] = ()

    def __bool__(self) -> bool:  # truthy = "something was felt"
        return bool(self.tags)


def _normalise(text: str) -> str:
    return text.lower().replace("’", "'").replace("‘", "'")


def _is_negated(text: str, match: re.Match) -> bool:
    """A negator inside the matched phrase, or within a few words before it?"""
    before = text[: match.start()].split()[-_NEGATION_WINDOW_WORDS:]
    return bool(_NEGATORS.search(" ".join(before) + " " + match.group(0)))


def _emphasised(raw: str) -> bool:
    """Shouting reads as more arousal: '!' or two+ ALL-CAPS words."""
    caps = sum(1 for w in re.findall(r"[A-Za-z']+", raw) if len(w) >= 2 and w.isupper())
    return "!" in raw or caps >= 2


def appraise(text: str, strength: float = 1.0) -> Appraisal:
    """Map one user utterance to a PAD delta and the reasons for it."""
    norm = _normalise(text).strip()
    if not norm:
        return Appraisal(PAD(), ())

    total = PAD()
    tags: list[str] = []
    for rule in _RULES:
        match = rule.pattern.search(norm)
        if match is None:
            continue
        if rule.negatable and _is_negated(norm, match):
            if rule.negated_delta is None:
                tags.append(f"{rule.tag}(negated, ignored)")
                continue
            total = total + rule.negated_delta
            tags.append(f"{rule.tag}(negated)")
        else:
            total = total + rule.delta
            tags.append(rule.tag)

    if not any("ignored" not in t for t in tags):
        return Appraisal(PAD(), tuple(tags))

    if _emphasised(text):
        total = PAD(total.pleasure, total.arousal * 1.3, total.dominance)
        tags.append("emphasis")

    total = total.scaled(strength).clamped(-_MAX_AXIS_DELTA, _MAX_AXIS_DELTA)
    return Appraisal(total, tuple(tags))


__all__ = ["Appraisal", "appraise"]
