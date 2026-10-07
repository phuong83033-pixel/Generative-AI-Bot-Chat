import logging
import time
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Generator, List

@dataclass
class SpanInfo:
    name: str
    start_time: float
    end_time: float = 0.0  
    duration_ms: float = 0.0

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage()
        }
        if hasattr(record, "kwargs_extra"):
            log_record.update(record.kwargs_extra) # type: ignore
        return json.dumps(log_record)

class Tracer:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
    def _log(self, level: int, msg: str, **kwargs) -> None:
        if kwargs:
            self.logger.log(level, msg, extra={"kwargs_extra": kwargs})
        else:
            self.logger.log(level, msg)

    def info(self, msg: str, **kwargs) -> None:
        self._log(logging.INFO, msg, **kwargs)
        
    def warning(self, msg: str, **kwargs) -> None:
        self._log(logging.WARNING, msg, **kwargs)
        
    def error(self, msg: str, **kwargs) -> None:
        self._log(logging.ERROR, msg, **kwargs)
        
    def debug(self, msg: str, **kwargs) -> None:
        self._log(logging.DEBUG, msg, **kwargs)

    @contextmanager
    def span(self, name: str) -> Generator[SpanInfo, None, None]:
        start = time.perf_counter()
        span_info = SpanInfo(name=name, start_time=start)
        try:
            yield span_info
        finally:
            end = time.perf_counter()
            span_info.end_time = end
            span_info.duration_ms = (end - start) * 1000
            self.info(f"Span '{name}' completed", duration_ms=span_info.duration_ms)

    def trace_event(self, name: str, data: Optional[Dict[str, Any]] = None) -> None:
        kwargs = data if data else {}
        self.info(f"Event: {name}", event=name, **kwargs)

def setup_logging(level: str = "INFO", log_dir: Optional[Path] = None) -> None:
    if log_dir is None:
        log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "amadeus.jsonl"
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler (JSON)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    # Console handler (Human readable)
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

def get_tracer(name: str) -> Tracer:
    return Tracer(name)

# ---------------------------------------------------------------------------
# Per-turn latency instrumentation
#
# Tracer.span() measures one block. A conversational turn is a chain of stages
# across several components, and the number that matters - time from the user
# finishing speaking to the first audible syllable - spans all of them. TurnTrace
# collects timestamps across that chain so main.py can report it, and so
# bench_latency.py can aggregate p50/p95 over many turns.
#
# Every stamp is monotonic (time.perf_counter), so these are durations, never
# wall-clock times.
# ---------------------------------------------------------------------------

# Canonical stage order for one turn. speech_end is t=0.
TURN_STAGES = (
    "speech_end",       # VAD reported end of user speech - the clock starts here
    "stt_done",         # transcript available
    "prompt_built",     # messages assembled
    "first_token",      # LLM emitted its first token
    "first_clause",     # chunker emitted a speakable clause
    "tts_done",         # synthesis returned audio for that clause
    "first_audio_out",  # playback began - THE headline metric
    "turn_done",        # playback drained
)


class TurnTrace:
    """Timestamps for the stages of a single conversational turn."""

    def __init__(self, turn_id: int):
        self.turn_id = turn_id
        self._stamps: Dict[str, float] = {}
        self._tracer = get_tracer("turn")

    def stamp(self, stage: str) -> float:
        """Record the current time against a stage name. Returns ms since speech_end."""
        now = time.perf_counter()
        self._stamps[stage] = now
        origin = self._stamps.get(TURN_STAGES[0])
        return (now - origin) * 1000.0 if origin is not None else 0.0

    def has(self, stage: str) -> bool:
        return stage in self._stamps

    def elapsed_ms(self, start: str, end: str) -> Optional[float]:
        """Milliseconds between two stamped stages, or None if either is missing."""
        a, b = self._stamps.get(start), self._stamps.get(end)
        return None if a is None or b is None else (b - a) * 1000.0

    @property
    def first_syllable_ms(self) -> Optional[float]:
        """End of user speech to first audible output. The headline metric."""
        return self.elapsed_ms("speech_end", "first_audio_out")

    def breakdown(self) -> Dict[str, float]:
        """Per-stage durations in ms, in canonical order, for stages reached."""
        out: Dict[str, float] = {}
        reached = [s for s in TURN_STAGES if s in self._stamps]
        for prev, cur in zip(reached, reached[1:]):
            d = self.elapsed_ms(prev, cur)
            if d is not None:
                out[f"{prev} -> {cur}"] = round(d, 1)
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "turn": self.turn_id,
            "first_syllable_ms": (
                round(self.first_syllable_ms, 1)
                if self.first_syllable_ms is not None
                else None
            ),
            "total_ms": (
                round(self.elapsed_ms("speech_end", "turn_done"), 1)
                if self.has("turn_done")
                else None
            ),
            "stages": self.breakdown(),
        }

    def log(self) -> None:
        s = self.summary()
        head = (
            f"{s['first_syllable_ms']:.0f}ms to first audio"
            if s["first_syllable_ms"] is not None
            else "incomplete"
        )
        self._tracer.info(f"Turn {self.turn_id}: {head}", **s)

    def table(self) -> str:
        """Human-readable stage breakdown for the console."""
        lines = [f"  turn {self.turn_id}"]
        for label, ms in self.breakdown().items():
            lines.append(f"    {label:<34}{ms:>8.0f} ms")
        if self.first_syllable_ms is not None:
            lines.append(f"    {'= first audible syllable':<34}{self.first_syllable_ms:>8.0f} ms")
        return "\n".join(lines)


class LatencyLog:
    """Aggregates TurnTrace results across a session."""

    def __init__(self) -> None:
        self.turns: List[TurnTrace] = []

    def add(self, trace: "TurnTrace") -> None:
        self.turns.append(trace)

    def _first_syllables(self) -> List[float]:
        return sorted(
            t.first_syllable_ms for t in self.turns if t.first_syllable_ms is not None
        )

    @staticmethod
    def percentile(values: List[float], pct: float) -> Optional[float]:
        if not values:
            return None
        k = (len(values) - 1) * (pct / 100.0)
        lo, hi = int(k), min(int(k) + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (k - lo)

    def report(self, targets: Optional[Dict[str, float]] = None) -> str:
        vals = self._first_syllables()
        if not vals:
            return "No completed turns measured."

        p50 = self.percentile(vals, 50)
        p95 = self.percentile(vals, 95)
        lines = [
            f"Turns measured : {len(vals)}",
            f"first syllable : p50 {p50:.0f} ms | p95 {p95:.0f} ms | "
            f"min {vals[0]:.0f} | max {vals[-1]:.0f}",
        ]
        if targets:
            for label, key, value in (
                ("p50", "first_syllable_p50_ms", p50),
                ("p95", "first_syllable_p95_ms", p95),
            ):
                target = targets.get(key)
                if target:
                    verdict = "PASS" if value <= target else "OVER"
                    lines.append(f"  {label} target {target:.0f} ms -> {verdict}")
        return "\n".join(lines)


__all__ = [
    "Tracer",
    "SpanInfo",
    "setup_logging",
    "get_tracer",
    "TurnTrace",
    "LatencyLog",
    "TURN_STAGES",
]
