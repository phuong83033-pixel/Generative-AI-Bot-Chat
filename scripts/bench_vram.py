"""Measure real VRAM cost per model, and LLM throughput.

Phase 0 deliverable. Replaces the estimates in README section 8 with
measurements. Loads each resident model in pipeline order and reports the
driver-level delta, then runs a short generation to measure TTFT and decode rate.

Measured at the driver level via torch.cuda.mem_get_info - torch's
memory_allocated() reports 0 for llama.cpp and CTranslate2, which do not use
torch's allocator. See runtime/vram.py.

IMPORTANT: close other GPU applications first. A game or hardware-accelerated
browser holding VRAM does not fail cleanly - llama.cpp silently offloads layers
to CPU and the numbers become meaningless.

Run: venv/Scripts/python.exe scripts/bench_vram.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from amadeus.config import load_config  # noqa: E402
from amadeus.runtime.cuda_dlls import ensure_cuda_dlls  # noqa: E402

PROMPT = [
    {"role": "system", "content": "You are Amadeus, a concise local AI companion."},
    {"role": "user", "content": "In about 120 words, explain what a GGUF quantised model is."},
]


def used_mb() -> float:
    free, total = torch.cuda.mem_get_info(0)
    return (total - free) / (1024**2)


def free_mb() -> float:
    return torch.cuda.mem_get_info(0)[0] / (1024**2)


def total_mb() -> float:
    return torch.cuda.mem_get_info(0)[1] / (1024**2)


def gpu_hogs() -> list[str]:
    import subprocess

    import os

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except Exception:
        return []

    hogs = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        # Exclude ourselves - we are supposed to be holding VRAM.
        if parts and parts[0].isdigit() and int(parts[0]) == os.getpid():
            continue
        hogs.append(", ".join(parts[1:]) if len(parts) > 1 else line.strip())
    return hogs


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable - nothing to measure.")
        return 1

    cfg = load_config()
    rows: list[tuple[str, float, float]] = []

    torch.cuda.init()
    baseline = used_mb()
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"total VRAM     : {total_mb():.0f} MB")
    print(f"baseline in use: {baseline:.0f} MB (CUDA context + anything else)")

    hogs = gpu_hogs()
    if hogs:
        print("\n*** WARNING: other processes hold GPU memory. Numbers will be wrong: ***")
        for h in hogs:
            print(f"      {h}")
    print()

    # ---- faster-whisper -------------------------------------------------
    before = used_mb()
    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    whisper = WhisperModel(
        cfg.stt.model_size, device=cfg.stt.device, compute_type=cfg.stt.compute_type
    )
    whisper_load = time.perf_counter() - t0
    # CTranslate2 allocates lazily; force a real decode to see true cost.
    import numpy as np

    list(whisper.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)[0])
    whisper_mb = used_mb() - before
    rows.append((f"faster-whisper {cfg.stt.model_size} ({cfg.stt.compute_type})",
                 whisper_mb, whisper_load))
    print(f"faster-whisper {cfg.stt.model_size:<8} {whisper_mb:+8.0f} MB   "
          f"load {whisper_load:5.2f}s   free now {free_mb():.0f} MB")

    # ---- llama.cpp ------------------------------------------------------
    ensure_cuda_dlls()
    from llama_cpp import Llama

    before = used_mb()
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(PROJECT_ROOT / cfg.llm.model_path),
        n_ctx=cfg.llm.n_ctx,
        n_gpu_layers=cfg.llm.n_gpu_layers,
        verbose=False,
    )
    llm_load = time.perf_counter() - t0
    llm_mb = used_mb() - before
    rows.append((f"Qwen2.5-3B Q4_K_M (n_ctx {cfg.llm.n_ctx})", llm_mb, llm_load))
    print(f"Qwen2.5-3B Q4_K_M      {llm_mb:+8.0f} MB   load {llm_load:5.2f}s   "
          f"free now {free_mb():.0f} MB")

    # ---- throughput -----------------------------------------------------
    print("\ngeneration:")
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    text = []
    for chunk in llm.create_chat_completion(
        messages=PROMPT, stream=True, max_tokens=200,
        temperature=cfg.llm.temperature, top_p=cfg.llm.top_p,
    ):
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta:
            if ttft is None:
                ttft = time.perf_counter() - t0
            tokens += 1
            text.append(delta["content"])
    elapsed = time.perf_counter() - t0
    tps = tokens / elapsed if elapsed else 0.0

    print(f"  TTFT            : {ttft * 1000:.0f} ms" if ttft else "  TTFT: n/a")
    print(f"  decode          : {tokens} tok in {elapsed:.2f}s = {tps:.1f} tok/s")
    print(f"  reply           : {''.join(text).strip()[:100]}")

    peak = used_mb()
    print(f"\npeak VRAM        : {peak:.0f} MB / {total_mb():.0f} MB "
          f"({peak / total_mb() * 100:.1f}%)")
    print(f"headroom         : {free_mb():.0f} MB")

    target = cfg.latency_targets.peak_vram_gb * 1024
    verdict = "WITHIN" if peak <= target else "OVER"
    print(f"budget           : {verdict} target of {target:.0f} MB "
          f"(settings.yaml latency_targets.peak_vram_gb)")

    out = PROJECT_ROOT / "data" / "logs" / "bench_vram.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "total_mb": round(total_mb(), 1),
                "baseline_mb": round(baseline, 1),
                "models": [
                    {"name": n, "vram_mb": round(v, 1), "load_s": round(s, 2)}
                    for n, v, s in rows
                ],
                "ttft_ms": round(ttft * 1000, 1) if ttft else None,
                "decode_tok_s": round(tps, 2),
                "peak_mb": round(peak, 1),
                "headroom_mb": round(free_mb(), 1),
                "other_gpu_processes": hogs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")
    return 0 if peak <= target else 1


if __name__ == "__main__":
    sys.exit(main())
