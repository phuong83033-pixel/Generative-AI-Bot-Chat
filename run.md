# RUN GUIDE — Amadeus

Everything needed to get the project running, with copy-paste commands.

> **Current stage: M7 (voice MVP).** A turn-based voice loop:
> mic → VAD → Whisper → Qwen → clause chunker → Edge-TTS → speakers.
> **No avatar, no emotion engine, no memory, no proactive speech yet** — those are
> M9 through M14. See README §28.

> **WEAR HEADPHONES.** `perception/aec.py` is a pass-through stub, so on speakers
> the agent hears itself, trips its own voice detector and starts answering
> itself. This is not a bug you can configure around yet — see §7 below.

---

## 1. Quickstart

Already set up? Three commands.

Check the machine is ready:

```bash
.\venv\Scripts\python.exe scripts\check_environment.py
```

Try it without a microphone (type instead of speak):

```bash
.\venv\Scripts\python.exe -m amadeus.main --text
```

Run it for real, headphones on:

```bash
.\venv\Scripts\python.exe -m amadeus.main
```

Type or say `exit` to quit. `Ctrl+C` also works.

---

## 2. Requirements

| | Minimum | This machine |
|---|---|---|
| OS | Windows 10/11 | Windows 11 Pro |
| GPU | NVIDIA, 4 GB VRAM, CUDA capable | RTX 2050, 4096 MiB, sm_86 |
| RAM | 8 GB | 15.7 GB |
| Disk | ~6 GB free | 59.9 GB |
| Python | **3.10–3.12** (not 3.13) | 3.12.10 |
| Network | needed for setup and for Edge-TTS at runtime | |

**Python 3.13 will not work.** `pydub` needs the stdlib `audioop` module, which
3.13 removed.

**The GPU must be effectively free.** Close games and GPU-heavy applications
before running. This is not a soft preference — see §7.

No CUDA Toolkit, CMake or Visual Studio build tools are required. Everything
installs as a prebuilt binary.

---

## 3. First-time setup

Skip to §4 if `check_environment.py` already passes.

All commands run from the project root (`Generative_Agent_bot/`).

### 3.1 Create the virtual environment

```bash
python -m venv venv
```

### 3.2 Install PyTorch with CUDA — do this FIRST

Order matters. Install the CUDA build before anything else, or pip resolves a
CPU-only wheel and CUDA silently disappears.

```bash
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify before continuing:

```bash
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect `2.5.1+cu121 True`. If it says `False`, stop and fix this first — nothing
downstream will work.

Torch also supplies the CUDA runtime DLLs that `llama-cpp-python` needs, so this
step is doing double duty (see §7, "Failed to load shared library").

### 3.3 Install the project and its dependencies

```bash
.\venv\Scripts\python.exe -m pip install -e ".[stt,tts,dev]"
```

This leaves the CUDA torch untouched — an already-satisfied requirement is not
re-resolved.

### 3.4 Install llama-cpp-python with CUDA

Do **not** use plain `pip install llama-cpp-python`; it builds from source and
needs a toolchain this machine does not have. Download the prebuilt CUDA wheel
with `curl` (pip downloads this file at ~190 kB/s and tends to corrupt it):

```bash
curl -L -C - --retry 5 -o llama_cpp_cu121.whl https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu121/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl
```

That is a 448 MB download. Then install it from disk:

```bash
.\venv\Scripts\python.exe -m pip install .\llama_cpp_cu121.whl
```

Confirm the CUDA backend actually loaded:

```bash
.\venv\Scripts\python.exe -c "from amadeus.runtime.cuda_dlls import ensure_cuda_dlls; ensure_cuda_dlls(); import llama_cpp; print('GPU offload:', llama_cpp.llama_supports_gpu_offload())"
```

Expect `GPU offload: True`. You can delete `llama_cpp_cu121.whl` afterwards.

### 3.5 Install ffmpeg

`pydub` shells out to it to decode Edge-TTS audio.

```bash
winget install --id Gyan.FFmpeg --scope user
```

**Restart your terminal afterwards.** winget edits the user PATH and running
shells do not pick that up.

### 3.6 Download the model weights

2.1 GB, into `data/models/`:

```bash
curl -L -C - --create-dirs -o data/models/qwen2.5-3b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
```

The filename must match `config/settings.yaml` exactly. Both `curl` commands here
use `-C -`, so a dropped connection resumes rather than restarting.

---

## 4. Verify

```bash
.\venv\Scripts\python.exe scripts\check_environment.py
```

Expected:

```text
[PASS] Python 3.10-3.12 - 3.12.10
[PASS] torch + CUDA - torch 2.5.1+cu121, CUDA 12.1, NVIDIA GeForce RTX 2050 (4.0 GB, sm_86)
[PASS] Free VRAM - 3306 MB free, 790 MB in use
[PASS] llama-cpp CUDA offload - llama_cpp 0.3.4, GPU offload available
[PASS] package faster-whisper
[PASS] package edge-tts
[PASS] package pydub
[PASS] package sounddevice
[PASS] package pyyaml
[PASS] package python-dotenv
[PASS] ffmpeg on PATH - ffmpeg version 9.0.2
[PASS] audio devices - in: Microphone (...) | out: Headphones (...)
[PASS] Qwen2.5-3B GGUF - 2007 MB, valid GGUF header
[PASS] directory data/models
[PASS] directory data/logs
[PASS] directory config

All checks passed.
```

A `[WARN]` on ffmpeg means it is installed but your shell has a stale PATH —
restart the terminal. Exit code is non-zero if anything `[FAIL]`s, so this can
gate CI.

---

## 5. Running

### Text mode — no microphone

The fastest way to confirm the LLM and voice output work. Skips mic, VAD and
Whisper entirely, so it also starts faster and uses less VRAM.

```bash
.\venv\Scripts\python.exe -m amadeus.main --text
```

### Voice mode

Headphones on.

```bash
.\venv\Scripts\python.exe -m amadeus.main
```

Wait for `[listening ...]`, then speak. It detects when you stop talking (800 ms
of silence by default) and replies.

First launch downloads the Silero VAD model from torch.hub, taking ~10 s. Later
launches read it from cache.

### Options

| Flag | Effect |
|---|---|
| `--text` | type instead of speaking; no mic, VAD or STT |
| `--turns N` | exit after N turns — useful for benchmarking |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default `INFO` |

Quiet run, three turns:

```bash
.\venv\Scripts\python.exe -m amadeus.main --turns 3 --log-level WARNING
```

### Stopping

Say or type `exit`, `quit`, `stop` or `goodbye`. `Ctrl+C` also works. If a run
ever gets stuck holding the GPU:

```bash
taskkill /F /IM python.exe
```

---

## 6. What to expect

Every turn prints a latency breakdown, and the session prints p50/p95 on exit:

```text
  turn 1
    speech_end -> stt_done                 240 ms
    stt_done -> prompt_built                 0 ms
    prompt_built -> first_token            306 ms
    first_token -> first_clause            390 ms
    first_clause -> tts_done              1909 ms
    tts_done -> first_audio_out              0 ms
    = first audible syllable              2605 ms
```

**First audible syllable is currently ~2.6 s against a 1200 ms target, and that
is expected at M7.** Edge-TTS is a cloud service and accounts for roughly 1900 ms
of it — about 73%. Everything running locally sums to ~700 ms. Replacing
Edge-TTS with local Kokoro-82M synthesis at Phase 2 is the fix; there is little
point optimising the local pipeline before that.

Measured resource use with all three models loaded:

```text
Qwen2.5-3B Q4_K_M    2210 MB
faster-whisper small  370 MB
CUDA context          790 MB
------------------------------
peak                 3398 MB / 4096 MB   (698 MB free)
decode               41.6 tok/s
```

Logs are written as JSON lines to `data/logs/amadeus.jsonl`.

---

## 7. Troubleshooting

### The agent starts talking to itself

You are on speakers. `perception/aec.py` is a pass-through stub, so the
microphone hears the agent, the voice detector fires, Whisper transcribes it, and
it answers its own sentence. **Use headphones.** Real echo cancellation is M8
(README §24).

### `Only 1148 MB VRAM free, need 2600 MB`

Something else is using the GPU. Find it:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Close it and retry. Games and hardware-accelerated browsers are the usual
culprits.

### It runs but is painfully slow (~5 tok/s instead of ~40)

Same cause as above, but worse because it fails *silently*. If another process
holds VRAM, llama.cpp quietly offloads layers to CPU rather than erroring.
Measured on this machine with a game running: **1732 ms to first token and
5.3 tok/s, against 82 ms and 41.6 tok/s on a clean GPU** — a 21× difference with
no warning. `require_free()` now refuses at startup, but if you see slow decode,
check the GPU first.

### `RuntimeError: Failed to load shared library ...llama.dll`

Misleading error — `llama.dll` is fine. The real chain is
`llama.dll → ggml.dll → ggml-cuda.dll → cudart64_12.dll + cublas64_12.dll`, and
those CUDA runtime libraries are missing. They normally come from the CUDA
Toolkit; this project borrows them from torch instead.

Confirm torch has CUDA:

```bash
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints a CPU-only version, redo §3.2. If torch is fine, something imported
`llama_cpp` before `ensure_cuda_dlls()` ran — that import order is load-bearing
and a failed library load is not retried.

### `error: Multiple top-level packages discovered in a flat-layout`

`pyproject.toml` is missing its package declaration. It should contain:

```toml
[tool.setuptools.packages.find]
include = ["amadeus*"]
```

### `ERROR: Wheel 'llama-cpp-python' ... is invalid`

pip downloaded the 448 MB wheel too slowly and corrupted it. Use the `curl`
method in §3.4 instead.

### `Couldn't find ffmpeg or avconv` / TTS produces silence

ffmpeg is missing or your shell has a stale PATH. Restart the terminal, then:

```bash
ffmpeg -version
```

If that fails, redo §3.5.

### No audio, or the wrong device

Check what the project sees:

```bash
.\venv\Scripts\python.exe -c "import sounddevice as sd; print(sd.query_devices())"
```

Override via `.env` in the project root:

```text
AMADEUS_AUDIO_INPUT_DEVICE=Microphone (Realtek(R) Audio)
AMADEUS_AUDIO_OUTPUT_DEVICE=Headphones (Realtek(R) Audio)
```

### It never responds to my voice

The voice detector may not be triggering. Lower the threshold in
`config/settings.yaml`:

```yaml
vad:
  threshold: 0.5          # lower = more sensitive, try 0.3
  silence_timeout_ms: 800 # how long you pause before it replies
```

Run with `--log-level DEBUG` to see whether speech is detected at all.

### Edge-TTS fails or hangs

Edge-TTS is a cloud service and needs network access. There is no offline voice
until Kokoro lands at Phase 2. Confirm the rest of the pipeline independently
with `--text`.

---

## 8. Benchmarks

Measure VRAM per model plus decode throughput. Close other GPU applications
first; results are written to `data/logs/bench_vram.json`.

```bash
.\venv\Scripts\python.exe scripts\bench_vram.py
```

`scripts/bench_latency.py` does not exist yet — per-turn tracing is already in
`trace.py` and prints after every turn, but the aggregating script is still to be
written (README §31.4).

---

## 9. Configuration

| File | Purpose |
|---|---|
| `config/settings.yaml` | audio, VAD, STT, LLM, TTS parameters and latency targets |
| `config/persona.yaml` | character name, traits and system prompt |
| `.env` | optional device overrides and `AMADEUS_LOG_LEVEL` |

Useful knobs in `settings.yaml`:

```yaml
llm:
  n_ctx: 2048        # context window; 698 MB headroom allows raising this
  max_tokens: 256    # reply length cap
  temperature: 0.7
tts:
  voice: en-US-AriaNeural   # any Edge-TTS voice
  rate: "+0%"               # "+20%" speaks faster
```

List available Edge-TTS voices:

```bash
.\venv\Scripts\python.exe -m edge_tts --list-voices
```

---

## 10. Known limits at M7

* **No avatar.** Nothing renders. M14.
* **No emotion engine.** The PAD model is designed, not built. M9.
* **No memory.** The agent forgets everything when you exit; within a session it
  keeps a short rolling history. M10–M11.
* **No proactive speech.** It only answers; it never starts a conversation. The
  state machine now permits it, but `drive.py` does not exist. M12.
* **No barge-in.** You cannot interrupt it mid-sentence. M8.
* **No echo cancellation.** Headphones required. M8.
* **Cloud TTS.** Needs network, and costs ~1900 ms per turn. M9.
* **Voice path unproven.** The microphone opens and the detector listens, but no
  real utterance has yet been captured and transcribed end to end. Text mode is
  verified working.

Full roadmap in README §28–45. Defect history in `fix.md`.
