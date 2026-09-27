import logging
import time
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Generator

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

__all__ = ["Tracer", "SpanInfo", "setup_logging", "get_tracer"]
