"""Make the CUDA runtime visible to llama-cpp-python on Windows.

WHY THIS EXISTS
---------------
The prebuilt llama-cpp-python CUDA wheels link ggml-cuda.dll against
cudart64_12.dll and cublas64_12.dll but do not bundle them. They normally come
from the CUDA Toolkit, which is NOT installed on this machine - only the driver
(nvcuda.dll). Importing llama_cpp therefore fails with a misleading error that
names llama.dll rather than the missing CUDA library:

    RuntimeError: Failed to load shared library '...\\llama_cpp\\lib\\llama.dll':
    Could not find module ... (or one of its dependencies).

The real dependency chain is:
    llama.dll -> ggml.dll -> ggml-cuda.dll -> cudart64_12.dll, cublas64_12.dll

The torch cu12 wheel already ships exactly those DLLs (cudart64_12.dll,
cublas64_12.dll and cublasLt64_12.dll, 608 MB together), so rather than
duplicating 608 MB into llama_cpp/lib we add torch's lib directory to the
process DLL search path.

MUST be called before `import llama_cpp`. See cognition/brain.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from amadeus.trace import get_tracer

# ggml-cuda.dll's actual CUDA imports, per its PE import table.
# cublasLt64_12.dll is not imported directly but is required by cublas64_12.dll.
REQUIRED_DLLS = ("cudart64_12.dll", "cublas64_12.dll")

_applied = False


def _candidate_dirs() -> list[Path]:
    """Directories that may hold the CUDA 12 runtime, in preference order."""
    dirs: list[Path] = []

    # 1. torch's bundled cu12 runtime - a hard dependency of this project, so
    #    this is the path that actually resolves on this machine.
    try:
        import torch

        dirs.append(Path(torch.__file__).resolve().parent / "lib")
    except ImportError:
        pass

    # 2. NVIDIA's standalone pip wheels (nvidia-cuda-runtime-cu12, nvidia-cublas-cu12),
        #  if someone installs them instead of relying on torch.
    try:
        import nvidia

        nvidia_base = Path(nvidia.__file__).resolve().parent
        dirs.extend(sorted(p for p in nvidia_base.glob("*/bin") if p.is_dir()))
    except ImportError:
        pass

    # 3. A real CUDA Toolkit install, if one ever appears.
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        dirs.append(Path(cuda_path) / "bin")

    return dirs


def ensure_cuda_dlls() -> list[Path]:
    """Add CUDA runtime directories to the DLL search path.

    Idempotent and a no-op off Windows. Returns the directories added.
    Logs a warning rather than raising when nothing is found: a CPU-only
    fallback is still useful, and llama_cpp will produce its own error.
    """
    global _applied
    tracer = get_tracer("cuda_dlls")

    if sys.platform != "win32":
        return []
    if _applied:
        return []

    added: list[Path] = []
    found: set[str] = set()

    for directory in _candidate_dirs():
        if not directory.is_dir():
            continue
        os.add_dll_directory(str(directory))
        added.append(directory)
        for dll in REQUIRED_DLLS:
            if (directory / dll).is_file():
                found.add(dll)

    _applied = True

    missing = [dll for dll in REQUIRED_DLLS if dll not in found]
    if missing:
        tracer.warning(
            "CUDA runtime DLLs not found on the search path: "
            f"{', '.join(missing)}. llama-cpp-python will fail to load its CUDA "
            "backend. Install the CUDA build of torch, or the CUDA Toolkit.",
            missing=missing,
            searched=[str(d) for d in added],
        )
    else:
        tracer.info(
            f"CUDA runtime resolved from {len(added)} director"
            f"{'y' if len(added) == 1 else 'ies'}",
            dirs=[str(d) for d in added],
        )

    return added


__all__ = ["ensure_cuda_dlls", "REQUIRED_DLLS"]
