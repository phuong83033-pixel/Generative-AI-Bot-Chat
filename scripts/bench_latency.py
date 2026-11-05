"""Measure first-syllable latency and inter-clause silence over a fixed prompt set.

README section 31.4 deliverable. Runs the REAL pipeline - real Qwen, real TTS, the
real chunker and the real Amadeus._respond - with only the speaker replaced by
NullPlayer, which waits out each clip's true duration without making sound. That
keeps timing faithful (synthesis of clause N+1 still overlaps playback of clause N)
while making iteration silent and repeatable.

Two numbers are reported because optimising one alone is a trap:

  first syllable   end of user speech -> first sound.   Lower is better.
  max gap          longest silence between two clauses. Lower is better.

A short first clause cuts the first number but can leave a hole while the next
clause is still being synthesised. A configuration that wins on latency by making
speech stutter is not a win.

Run (close games / GPU apps first):
    venv/Scripts/python.exe scripts/bench_latency.py --label baseline
    venv/Scripts/python.exe scripts/bench_latency.py --label pipelined --repeats 3

Results are also written to data/logs/bench_latency_<label>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from amadeus.config import load_config  # noqa: E402
from amadeus.main import Amadeus  # noqa: E402
from amadeus.state import SystemState  # noqa: E402
from amadeus.trace import LatencyLog, TurnTrace, setup_logging  # noqa: E402

# Varied on purpose: greetings open with a short interjection ("Hello!"), factual
# questions open with a long clause, and short answers have only one clause. The
# first-clause strategy behaves differently on each.
PROMPTS = [
    "Hello! Who are you?",
    "Tell me one fun fact about the brain.",
    "What's your favourite colour?",
    "Why is the sky blue?",
    "Good morning, how are you today?",
    "Can you help me plan a study schedule?",
    "What do you think about coffee?",
    "Say something encouraging.",
]


def pct(values: list[float], p: float) -> float:
    return LatencyLog.percentile(sorted(values), p) if values else float("nan")


async def run(label: str, repeats: int, first_min: str | None, first_max: str | None,
              seed: int | None) -> int:
    cfg = load_config()
    app = Amadeus(cfg, text_mode=True, mute=True)
    await app.setup()

    # Sweep the first-clause threshold without editing settings.yaml between runs.
    if first_min is not None:
        app.chunker.first_clause_min = None if first_min.lower() == "none" else int(first_min)
    if first_max is not None:
        app.chunker.first_clause_max = None if first_max.lower() == "none" else int(first_max)
    app.brain.seed = seed  # identical replies across configurations => paired comparison
    print(f"first_clause_min = {app.chunker.first_clause_min}  (None = original rules), first_clause_max = {app.chunker.first_clause_max} | seed = {seed}\n")

    traces: list[TurnTrace] = []
    replies: list[tuple[str, str]] = []
    n = 0
    total = len(PROMPTS) * repeats
    for rep in range(repeats):
        for prompt in PROMPTS:
            n += 1
            app.prompt.clear_history()  # each prompt independent of the last
            app.turn_id += 1
            trace = TurnTrace(app.turn_id)
            stop = asyncio.Event()

            trace.stamp("speech_end")
            trace.stamp("stt_done")
            app.state.transition(SystemState.THINKING)
            reply = await app._respond(prompt, trace, stop)
            trace.stamp("turn_done")
            app.state.transition(SystemState.IDLE)

            traces.append(trace)
            replies.append((prompt, reply))
            fs = trace.first_syllable_ms
            print(
                f"[{n:>2}/{total}] {fs if fs is not None else float('nan'):>6.0f} ms  "
                f"gap {trace.max_gap_ms:>5.0f} ms  clauses {len(trace.gaps_ms) + 1}  | {prompt[:38]}",
                flush=True,
            )

    first = [t.first_syllable_ms for t in traces if t.first_syllable_ms is not None]
    gaps = [t.max_gap_ms for t in traces]
    multi = [t for t in traces if t.gaps_ms]

    t_p50 = cfg.latency_targets.first_syllable_p50_ms
    t_p95 = cfg.latency_targets.first_syllable_p95_ms
    print(f"\n=== {label}  (provider={cfg.tts.provider}, {len(first)} turns) ===")
    print(
        f"first syllable : p50 {pct(first, 50):.0f} ms | p95 {pct(first, 95):.0f} ms | "
        f"min {min(first):.0f} | max {max(first):.0f}"
    )
    print(
        f"  vs target    : p50 {'PASS' if pct(first, 50) <= t_p50 else 'OVER'} ({t_p50} ms) | "
        f"p95 {'PASS' if pct(first, 95) <= t_p95 else 'OVER'} ({t_p95} ms)"
    )
    print(
        f"max gap        : median {statistics.median(gaps):.0f} ms | worst {max(gaps):.0f} ms  "
        f"({len(multi)}/{len(traces)} turns had >1 clause)"
    )

    stage_names = ["prompt_built -> first_token", "first_token -> first_clause", "first_clause -> tts_done"]
    print("median stage cost:")
    for name in stage_names:
        vals = [t.breakdown().get(name) for t in traces if t.breakdown().get(name) is not None]
        if vals:
            print(f"  {name:<30}{statistics.median(vals):>7.0f} ms")

    out = PROJECT_ROOT / "data" / "logs" / f"bench_latency_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "label": label,
                "tts_provider": cfg.tts.provider,
                "first_clause_min": app.chunker.first_clause_min,
                "first_clause_max": app.chunker.first_clause_max,
                "seed": seed,
                "turns": len(first),
                "first_syllable_ms": {
                    "p50": round(pct(first, 50), 1),
                    "p95": round(pct(first, 95), 1),
                    "min": round(min(first), 1),
                    "max": round(max(first), 1),
                },
                "max_gap_ms": {"median": round(statistics.median(gaps), 1), "worst": round(max(gaps), 1)},
                "per_turn": [
                    {"prompt": p, "reply": r, **t.summary()} for (p, r), t in zip(replies, traces)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", default="run", help="tag for the output file")
    ap.add_argument("--repeats", type=int, default=2, help="passes over the prompt set")
    ap.add_argument("--seed", type=int, default=1234,
                    help="LLM sampling seed so every run sees the same replies (-1 = random)")
    ap.add_argument("--first-max", default=None, metavar="N|none",
                    help="override chunker.first_clause_max_chars; 'none' = off")
    ap.add_argument("--first-min", default=None, metavar="N|none",
                    help="override chunker.first_clause_min_chars; 'none' = original rules")
    args = ap.parse_args()
    setup_logging("WARNING", log_dir=PROJECT_ROOT / "data" / "logs")
    return asyncio.run(run(args.label, args.repeats, args.first_min, args.first_max,
                           None if args.seed < 0 else args.seed))


if __name__ == "__main__":
    sys.exit(main())
