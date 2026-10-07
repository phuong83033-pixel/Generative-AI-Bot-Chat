"""Verify the machine can actually run Amadeus.

Checks every dependency the pipeline needs, including the three that fail in
confusing ways rather than obvious ones:

  * llama-cpp-python CUDA offload - the prebuilt wheel imports fine but raises a
    misleading error about llama.dll when the CUDA runtime is missing.
  * ffmpeg on PATH - pydub shells out to it, and the failure surfaces as an
    empty audio array rather than an exception.
  * free VRAM - another process holding the GPU does not cause a clean failure;
    llama.cpp silently offloads to CPU and throughput collapses.

Run: venv/Scripts/python.exe scripts/check_environment.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GGUF_PATH = PROJECT_ROOT / "data" / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

# Rough floor for Qwen2.5-3B Q4_K_M at n_ctx 2048 plus the CUDA context.
MIN_FREE_VRAM_MB = 2600

_failures: list[str] = []


def result(name: str, passed: bool, details: str = "", *, warn: bool = False) -> None:
    if passed:
        tag = "\033[92m[PASS]\033[0m"
    elif warn:
        tag = "\033[93m[WARN]\033[0m"
    else:
        tag = "\033[91m[FAIL]\033[0m"
        _failures.append(name)
    print(f"{tag} {name}{f' - {details}' if details else ''}")


def check_python() -> None:
    v = sys.version_info
    # pydub needs stdlib audioop, removed in 3.13.
    ok = v.major == 3 and 10 <= v.minor <= 12
    result(
        "Python 3.10-3.12",
        ok,
        f"{v.major}.{v.minor}.{v.micro}"
        + ("" if ok else "  (3.13+ removes audioop, which pydub needs)"),
    )


def check_torch_cuda() -> None:
    try:
        import torch
    except ImportError:
        result("torch + CUDA", False, "torch not installed")
        return

    if not torch.cuda.is_available():
        result(
            "torch + CUDA",
            False,
            f"torch {torch.__version__} has no CUDA. Reinstall from "
            "https://download.pytorch.org/whl/cu121",
        )
        return

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cap = torch.cuda.get_device_capability(0)
    result(
        "torch + CUDA",
        True,
        f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
        f"{name} ({total:.1f} GB, sm_{cap[0]}{cap[1]})",
    )


def check_vram_free() -> None:
    """The budget assumes near-exclusive use of the GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            result("Free VRAM", False, "CUDA unavailable")
            return
        free_b, total_b = torch.cuda.mem_get_info(0)
    except Exception as e:
        result("Free VRAM", False, f"could not query: {e}")
        return

    free_mb = free_b / (1024**2)
    used_mb = (total_b - free_b) / (1024**2)
    detail = f"{free_mb:.0f} MB free, {used_mb:.0f} MB in use"

    if free_mb >= MIN_FREE_VRAM_MB:
        result("Free VRAM", True, detail)
        return

    hogs = gpu_processes()
    detail += f" (need >= {MIN_FREE_VRAM_MB} MB)"
    if hogs:
        detail += ". Holding VRAM: " + ", ".join(hogs)
    result("Free VRAM", False, detail)


def gpu_processes() -> list[str]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception:
        return []
    names = []
    for line in out.splitlines():
        if line.strip():
            names.append(Path(line.split(",")[0].strip()).name)
    return names


def check_llama_cuda() -> None:
    """Import through the CUDA DLL shim, exactly as brain.py does."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from amadeus.runtime.cuda_dlls import ensure_cuda_dlls

        ensure_cuda_dlls()
    except Exception as e:
        result("llama-cpp CUDA offload", False, f"CUDA DLL shim failed: {e}")
        return

    try:
        import llama_cpp
    except Exception as e:
        result("llama-cpp CUDA offload", False, f"import failed: {type(e).__name__}: {e}")
        return

    offload = llama_cpp.llama_supports_gpu_offload()
    result(
        "llama-cpp CUDA offload",
        offload,
        f"llama_cpp {llama_cpp.__version__}, GPU offload "
        + ("available" if offload else "NOT available - CPU-only wheel"),
    )


def check_packages() -> None:
    for pkg, label in [
        ("faster_whisper", "faster-whisper"),
        ("edge_tts", "edge-tts"),
        ("pydub", "pydub"),
        ("sounddevice", "sounddevice"),
        ("yaml", "pyyaml"),
        ("dotenv", "python-dotenv"),
    ]:
        try:
            __import__(pkg)
            result(f"package {label}", True)
        except ImportError:
            result(f"package {label}", False, "not installed")


def check_ffmpeg() -> None:
    exe = shutil.which("ffmpeg")
    if exe:
        try:
            ver = subprocess.run(
                [exe, "-version"], capture_output=True, text=True, timeout=15
            ).stdout.splitlines()[0]
        except Exception:
            ver = exe
        result("ffmpeg on PATH", True, ver[:72])
        return

    # winget installs to the user PATH, which running shells do not inherit.
    guess = (
        Path.home()
        / "AppData/Local/Microsoft/WinGet/Packages"
    )
    found = list(guess.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe")) if guess.is_dir() else []
    if found:
        result(
            "ffmpeg on PATH",
            False,
            f"installed at {found[0]} but not on this shell's PATH - "
            "restart the terminal",
            warn=True,
        )
    else:
        result("ffmpeg on PATH", False, "not found. `winget install Gyan.FFmpeg`")


def check_audio_devices() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        result("audio devices", False, "sounddevice not installed")
        return
    try:
        devices = sd.query_devices()
        din, dout = sd.default.device
        name_in = devices[din]["name"] if din is not None and din >= 0 else "None"
        name_out = devices[dout]["name"] if dout is not None and dout >= 0 else "None"
        result("audio devices", True, f"in: {name_in} | out: {name_out}")
    except Exception as e:
        result("audio devices", False, f"query failed: {e}")


def check_model_weights() -> None:
    if not GGUF_PATH.is_file():
        result(
            "Qwen2.5-3B GGUF",
            False,
            f"missing: {GGUF_PATH.relative_to(PROJECT_ROOT)} - see README section 4.3",
        )
        return

    size_mb = GGUF_PATH.stat().st_size / (1024**2)
    with GGUF_PATH.open("rb") as f:
        magic = f.read(4)

    if magic != b"GGUF":
        result("Qwen2.5-3B GGUF", False, f"bad magic {magic!r} - truncated download?")
        return
    result("Qwen2.5-3B GGUF", True, f"{size_mb:.0f} MB, valid GGUF header")


def check_directories() -> None:
    for rel in ["data/models", "data/logs", "config"]:
        path = PROJECT_ROOT / rel
        result(f"directory {rel}", path.is_dir(), "" if path.is_dir() else "missing")


def main() -> int:
    print("Amadeus environment check\n")
    check_python()
    check_torch_cuda()
    check_vram_free()
    check_llama_cuda()
    check_packages()
    check_ffmpeg()
    check_audio_devices()
    check_model_weights()
    check_directories()

    print()
    if _failures:
        print(f"\033[91m{len(_failures)} check(s) failed:\033[0m " + ", ".join(_failures))
        return 1
    print("\033[92mAll checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
