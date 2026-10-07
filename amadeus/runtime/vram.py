"""VRAM accounting and the startup budget guard.

WHY THIS WAS REWRITTEN
----------------------
The first version measured with `torch.cuda.memory_allocated()`, which reports
only what **torch's own caching allocator** holds. The two largest consumers in
this project do not allocate through torch:

  * llama-cpp-python  -> ggml-cuda allocates via the CUDA driver directly
  * faster-whisper    -> CTranslate2 allocates via the CUDA driver directly

Measured with the model fully loaded, `memory_allocated()` returned **0.0 MB**
while the driver reported ~3.9 GB in use. The guard was blind to essentially the
entire budget it existed to protect, and would have reported a comfortable
margin right up to an out-of-memory crash.

`torch.cuda.mem_get_info()` queries the CUDA driver and therefore sees every
allocation in the process regardless of which library made it. That is the
number this module reports as `used_mb`.

A second lesson from that run: `mem_get_info` is also **process-wide across the
whole device**, so another application holding VRAM (a game, a browser with
hardware acceleration) counts against the same 4096 MiB. This is why
`require_free()` exists - the budget in README section 8 assumes effectively
exclusive access to the GPU, and that assumption needs checking at startup
rather than discovering it as a mysterious slowdown.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional

import torch

from amadeus.trace import get_tracer


@dataclass(frozen=True)
class VRAMInfo:
    """A VRAM snapshot.

    `used_mb` / `free_mb` are driver-level and device-wide: they include
    allocations made by llama.cpp, CTranslate2 and other processes.
    `torch_allocated_mb` is torch's allocator only, kept purely for diagnostics -
    never use it for budget decisions.
    """

    total_mb: float
    used_mb: float
    free_mb: float
    percent_used: float
    torch_allocated_mb: float


def get_vram_info(device: Optional[int] = None) -> Optional[VRAMInfo]:
    """Driver-level VRAM snapshot, or None when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None

    if device is None:
        device = torch.cuda.current_device()

    # Driver-level: sees ggml-cuda, CTranslate2 and other processes.
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = total_bytes - free_bytes

    total_mb = total_bytes / (1024**2)
    used_mb = used_bytes / (1024**2)

    return VRAMInfo(
        total_mb=total_mb,
        used_mb=used_mb,
        free_mb=free_bytes / (1024**2),
        percent_used=(used_mb / total_mb * 100) if total_mb > 0 else 0.0,
        torch_allocated_mb=torch.cuda.memory_allocated(device) / (1024**2),
    )


def log_vram(label: str = "") -> Optional[VRAMInfo]:
    """Log the current driver-level VRAM usage and return the snapshot."""
    tracer = get_tracer("vram")
    info = get_vram_info()

    if info is None:
        tracer.warning(f"VRAM [{label}]: CUDA not available")
        return None

    tracer.info(
        f"VRAM [{label}]: {info.used_mb:.0f}MB / {info.total_mb:.0f}MB "
        f"({info.percent_used:.1f}%), {info.free_mb:.0f}MB free",
        label=label,
        used_mb=round(info.used_mb, 1),
        free_mb=round(info.free_mb, 1),
        total_mb=round(info.total_mb, 1),
        torch_allocated_mb=round(info.torch_allocated_mb, 1),
    )
    return info


@contextmanager
def vram_guard(label: str) -> Generator[None, None, None]:
    """Log driver-level VRAM before and after a block, plus the delta.

    Wrap every model load in this. The delta is the real cost of that model,
    which is the number README section 8 should contain.
    """
    tracer = get_tracer("vram")
    before = log_vram(f"{label} (before)")
    try:
        yield
    finally:
        after = log_vram(f"{label} (after)")
        if before and after:
            delta = after.used_mb - before.used_mb
            tracer.info(
                f"VRAM delta [{label}]: {delta:+.0f}MB "
                f"(now {after.free_mb:.0f}MB free)",
                label=label,
                delta_mb=round(delta, 1),
                free_mb=round(after.free_mb, 1),
            )


def require_free(min_free_mb: float, *, label: str = "startup") -> VRAMInfo:
    """Fail fast when the GPU does not have room for what we are about to load.

    The budget assumes near-exclusive use of the device. Another application
    holding VRAM does not cause a clean failure: llama.cpp silently offloads
    layers to CPU and decode throughput collapses (measured at 5.3 tok/s against
    an expected 30-40 when a game held 2.8 GB). A slow pipeline is far harder to
    diagnose than a refusal at startup, so this refuses.
    """
    info = get_vram_info()
    if info is None:
        raise RuntimeError(
            f"[{label}] CUDA is not available, so VRAM cannot be reserved. "
            "Check the torch install is the CUDA build."
        )

    if info.free_mb < min_free_mb:
        raise RuntimeError(
            f"[{label}] Only {info.free_mb:.0f}MB VRAM free, need {min_free_mb:.0f}MB. "
            f"({info.used_mb:.0f}MB of {info.total_mb:.0f}MB already in use.) "
            "Close other GPU applications - games and hardware-accelerated "
            "browsers are the usual culprits. Run `nvidia-smi "
            "--query-compute-apps=pid,process_name,used_memory --format=csv` "
            "to see what holds the memory."
        )

    get_tracer("vram").info(
        f"VRAM check passed [{label}]: {info.free_mb:.0f}MB free, "
        f"{min_free_mb:.0f}MB required",
        label=label,
        free_mb=round(info.free_mb, 1),
        required_mb=min_free_mb,
    )
    return info


__all__ = ["VRAMInfo", "get_vram_info", "log_vram", "vram_guard", "require_free"]
