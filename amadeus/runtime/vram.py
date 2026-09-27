import torch
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional, Generator
from amadeus.trace import get_tracer

@dataclass
class VRAMInfo:
    total_mb: float
    used_mb: float  
    free_mb: float
    percent_used: float

def get_vram_info() -> Optional[VRAMInfo]:
    """Returns current VRAM info, or None if no CUDA GPU."""
    if not torch.cuda.is_available():
        return None
    
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    free = total - allocated
    
    total_mb = total / (1024 ** 2)
    used_mb = allocated / (1024 ** 2)
    free_mb = free / (1024 ** 2)
    percent_used = (used_mb / total_mb) * 100 if total_mb > 0 else 0.0
    
    return VRAMInfo(
        total_mb=total_mb,
        used_mb=used_mb,
        free_mb=free_mb,
        percent_used=percent_used
    )

def log_vram(label: str = "") -> Optional[VRAMInfo]:
    """Logs current VRAM with a context label and returns info."""
    tracer = get_tracer("vram")
    info = get_vram_info()
    if info:
        msg = f"VRAM [{label}]: {info.used_mb:.2f}MB / {info.total_mb:.2f}MB ({info.percent_used:.1f}%)"
        tracer.info(msg, label=label, used_mb=info.used_mb, total_mb=info.total_mb)
    else:
        tracer.warning(f"VRAM [{label}]: CUDA not available")
    return info

@contextmanager
def vram_guard(label: str) -> Generator[None, None, None]:
    """Context manager that logs VRAM before and after, plus delta."""
    tracer = get_tracer("vram")
    info_before = log_vram(f"{label} (Before)")
    try:
        yield
    finally:
        info_after = log_vram(f"{label} (After)")
        if info_before and info_after:
            delta = info_after.used_mb - info_before.used_mb
            tracer.info(f"VRAM Delta [{label}]: {delta:+.2f}MB", delta_mb=delta)

__all__ = ["VRAMInfo", "get_vram_info", "log_vram", "vram_guard"]
