# FIX LOG — Pre-MVP Blocker Removal

> **Date:** 2026-10-05
> **Scope:** The four defects that made an end-to-end run impossible before `main.py` was written.
> **Status:** All four fixed and verified. The pipeline is still not runnable end-to-end — see §6.

---

## 1. Why this log exists

The repository had every M2–M6 component in place (`bus`, `trace`, `state`, `vram`, `scheduler`, `audio_in`, `vad`, `stt`, `aec`, `brain`, `prompt`, `chunker`, `edge`, `audio_out`) and an empty `amadeus/main.py`. On the surface it looked like one file away from the M7 voice MVP.

It was not. Four defects sat on the exact path `main.py` would have to take, and three of the four fail *silently or confusingly* rather than at the point of the mistake. Writing `main.py` first would have meant debugging the orchestration layer while the layers underneath it were quietly broken.

The four were fixed before `main.py` was written, deliberately.

| # | File | Defect | Failure mode |
|---|---|---|---|
| 1 | `perception/audio_in.py` | `self.stream` attribute shadowed the `stream()` method | Hard `TypeError` on first audio frame |
| 2 | `perception/vad.py` | `bus.publish(...)` coroutine never awaited | Silent — events simply never fire |
| 3 | `bus.py`, `perception/audio_in.py` | Event loop captured in `__init__`, before any loop runs | Silent — cross-thread events post into a dead loop |
| 4 | `pyproject.toml` | `torch` and `pydub` imported but never declared | `ModuleNotFoundError` on a clean install |

---

## 2. Blocker 1 — `audio_in.py`: attribute shadowed the method

### Problem with the previous protocol

`MicrophoneStream` declared both a coroutine method `stream()` and an instance attribute `self.stream` holding the `sounddevice.InputStream`:

```python
def __init__(self, ...):
    self.stream: Optional[sd.InputStream] = None   # <-- shadows the method

async def start(self) -> None:
    self.stream = sd.InputStream(...)              # <-- now holds an InputStream

async def stream(self, stop) -> AsyncIterator[np.ndarray]:   # <-- unreachable
    ...
```

Python resolves instance attributes before class attributes, so `mic.stream` was never the method. It was `None` before `start()` and an `InputStream` after it.

`SileroVAD.listen_until_speech_ends()` consumes audio via exactly that method:

```python
async for chunk in mic.stream(stop):
```

So the first time audio was requested, this raised:

```
TypeError: 'InputStream' object is not callable
```

This is a **guaranteed** crash on the first run, not an edge case. Confirmed by AST inspection before the fix: `COLLISION: ['stream']`.

### What was fixed

The private attribute was renamed; the public coroutine method keeps the name.

```python
self._stream: Optional[sd.InputStream] = None
```

`start()` and `stop()` were updated to use `self._stream`. The `stream()` async generator and every caller are untouched — `mic.stream(stop)` now resolves to the method, as the VAD always assumed.

---

## 3. Blocker 2 — `vad.py`: coroutines created and discarded

### Problem with the previous protocol

`EventBus.publish` is declared `async def`. `vad.py` called it from inside an async function without `await`, in two places:

```python
if self.bus:
    self.bus.publish("SpeechStart", {})   # creates a coroutine, drops it
...
if self.bus:
    self.bus.publish("SpeechEnd", {})     # same
```

Calling a coroutine function without awaiting it builds a coroutine object and discards it. The body never executes. The only trace is a `RuntimeWarning: coroutine 'EventBus.publish' was never awaited`, which is easy to lose in startup noise, especially with the JSON file handler in `trace.py` capturing everything.

Consequence: **`SpeechStart` and `SpeechEnd` never reached any subscriber.** Any component built on those events — barge-in, the state machine, latency stamping at the turn boundary — would have appeared correctly wired and done nothing. This is the worst of the four, because the symptom is absence rather than error.

### What was fixed

```python
if self.bus:
    await self.bus.publish("SpeechStart", {})
...
if self.bus:
    await self.bus.publish("SpeechEnd", {})
```

Both call sites are already inside `async def listen_until_speech_ends`, so awaiting is legal and changes no control flow.

A regex sweep over the whole `amadeus/` package confirms no other un-awaited `bus.publish(...)` remains.

---

## 4. Blocker 3 — `bus.py` / `audio_in.py`: loop captured before it existed

### Problem with the previous protocol

Both classes grabbed an event loop in their constructor:

```python
# bus.py
try:
    self._loop = asyncio.get_running_loop()
except RuntimeError:
    self._loop = asyncio.get_event_loop()   # <-- the trap

# audio_in.py
self._loop = asyncio.get_event_loop()
```

The `except RuntimeError` branch is reached precisely when there is no running loop — i.e. when the object is built in synchronous startup code, before `asyncio.run()`. In that situation `asyncio.get_event_loop()` on Python 3.12 emits a `DeprecationWarning` and returns a loop that **is never started**. In 3.14 it raises outright.

`publish_sync` then did:

```python
self._loop.call_soon_threadsafe(_publish_wrapper)
```

Posting to a loop that never runs. The callback is queued and never executed. No exception, no log line, no event.

This mattered more than it looks, because `StateManager.transition()` is synchronous and publishes through `publish_sync`. Every state transition in the system would have been invisible. And since `main.py` did not exist yet, this defect was aimed squarely at the file about to be written: the natural shape of `main()` is to construct the bus and the components, then call `asyncio.run(...)` — which is exactly the order that triggers the bug.

### What was fixed

**`bus.py`** no longer binds a loop at construction. It binds on first async use, or explicitly:

```python
def __init__(self):
    self._loop: Optional[asyncio.AbstractEventLoop] = None

def bind_loop(self, loop=None) -> asyncio.AbstractEventLoop:
    """Bind the bus to the running loop. Call from inside the event loop."""
    self._loop = loop or asyncio.get_running_loop()
    return self._loop

async def publish(self, event_name, data=None) -> None:
    if self._loop is None:
        self._loop = asyncio.get_running_loop()
    ...
```

`publish_sync` now **fails loudly** instead of dropping the event, and uses `run_coroutine_threadsafe`, the canonical cross-thread API, with a done-callback so a subscriber exception is logged rather than swallowed:

```python
def publish_sync(self, event_name, data=None) -> None:
    loop = self._loop
    if loop is None or loop.is_closed():
        raise RuntimeError(
            f"EventBus is not bound to a running loop, so '{event_name}' cannot be "
            "delivered. Call bind_loop() from inside the event loop during startup."
        )
    future = asyncio.run_coroutine_threadsafe(self.publish(event_name, data), loop)
    future.add_done_callback(_report)
```

The raise is a deliberate trade. A dropped event costs hours to diagnose; a `RuntimeError` at startup costs seconds.

**`audio_in.py`** binds inside `start()`, which is `async def` and therefore always has a running loop:

```python
async def start(self) -> None:
    if self._stream is not None:
        ...
        return
    self._loop = asyncio.get_running_loop()
    self._stream = sd.InputStream(...)
```

The PortAudio callback runs on a separate thread and now guards against an unbound loop rather than dereferencing `None`:

```python
def _audio_callback(self, indata, frames, time, status):
    if status:
        self.tracer.warning(f"Audio callback status: {status}")
    if self._loop is None:
        return
    data = indata.copy().flatten()
    self._loop.call_soon_threadsafe(self.queue.put_nowait, data)
```

---

## 5. Blocker 4 — `pyproject.toml`: undeclared imports

### Problem with the previous protocol

Three modules imported packages the project never declared:

| Import | Used by | Declared? |
|---|---|---|
| `torch` | `perception/vad.py` (Silero), `runtime/vram.py`, `scripts/check_environment.py` | no |
| `pydub` | `synthesis/edge.py` (mp3 → PCM) | no |

A clean `pip install -e .[all]` produced an environment that imports `amadeus.perception.vad` and dies with `ModuleNotFoundError: No module named 'torch'`. The current `venv` only works because torch was installed by hand.

### What was fixed

`torch` moved into core `dependencies` (the VAD is not optional — it is the turn detector), and `pydub` joined the `tts` extra alongside `edge-tts`.

Two hazards are now documented in the file itself rather than left to be rediscovered:

* **torch must come from the CUDA index first.** `pip install torch` from PyPI can resolve a CPU-only wheel, at which point `device: cuda` in `settings.yaml` fails and the VAD silently runs on CPU-only torch. The comment pins the correct install line. An already-satisfied torch is left untouched by `pip install -e .`, so the hand-installed `2.5.1+cu121` survives.
* **`pydub` needs stdlib `audioop`, removed in Python 3.13.** `requires-python = ">=3.10"` currently permits 3.13, where `synthesis/edge.py` would break. Noted in the file with the migration route (`soundfile` + `av`).

---

## 6. Behaviour changes callers must honour

Two of these fixes change the contract. Both land on `main.py`, which has not been written yet — this is the reason the fixes came first.

1. **`bus.bind_loop()` must be called early, from inside the event loop,** before anything can use `publish_sync`. `StateManager.transition()` publishes synchronously, so the first state transition will now raise if the bus is unbound. The correct shape:

   ```python
   async def run() -> None:
       bus.bind_loop()          # first statement, before any component starts
       ...

   def main() -> None:
       asyncio.run(run())
   ```

2. **`MicrophoneStream.start()` must be awaited before `stream()` is iterated.** The loop is bound in `start()`; iterating first yields nothing and the PortAudio callback drops frames via the new `None` guard. The VAD already calls them in this order.

---

## 7. Verification

Run against `venv/Scripts/python.exe` (Python 3.12.10):

```
=== BLOCKER 1: method/attribute collision ===
  collisions: none
=== BLOCKER 2: awaited publishes in vad ===
  SpeechStart awaited: True
  SpeechEnd   awaited: True
=== BLOCKER 3: bus loop binding ===
  loop after construction: None
  raises when unbound: EventBus is not bound to a running loop, so 'Boom' cannot ...
  cross-thread delivery: [{'n': 1}]

ALL FOUR BLOCKERS VERIFIED
```

Blocker 3's check is the meaningful one: the bus is constructed outside any loop (the exact condition that used to produce a dead loop), `publish_sync` now raises instead of dropping, and after `bind_loop()` an event published **from a worker thread** is delivered to a subscriber on the loop thread.

Blocker 4 verified by parsing `pyproject.toml`: `torch` present in core deps, `pydub` present in the `tts` extra.

Also green: `python -m compileall amadeus scripts` (exit 0), and `amadeus.bus`, `amadeus.trace`, `amadeus.state`, `amadeus.runtime.vram` all import cleanly.

**What this does not prove.** These are unit-level verifications of four specific defects. No audio has been captured, no model has been loaded, and no turn has run end to end. That is still blocked by §8.

---

## 8. Still outstanding

Deliberately out of scope for this pass, in priority order.

### Blocking the first run
* **`amadeus/main.py` is empty.** No orchestration exists. This is M7.
* **Dependencies missing from `venv`:** `faster-whisper`, `llama-cpp-python`, `edge-tts`, `pydub`, `sounddevice`, `pyyaml`, `python-dotenv`. Only `torch 2.5.1+cu121` and `numpy 2.5.2` are installed.
* **No model weights.** `settings.yaml` points at `data/models/qwen2.5-3b-instruct-q4_k_m.gguf`, which does not exist.
* **`ffmpeg`** must be on `PATH` for `pydub` to decode Edge-TTS output.

### Divergences from the spec, cheap to fix now
* **`state.py` has no `IDLE → THINKING` edge.** `_valid_transitions[IDLE] = {SystemState.LISTENING}`. README §6 and §19 make that edge the thing separating this system from a voice assistant, so proactive speech is currently structurally impossible. One line now; a refactor once the turn loop is built against the present table.
* **`state.py` transition validation has a hole.** `if new_state != SystemState.IDLE and new_state not in ...` means *any* transition to `IDLE` bypasses the table entirely, so the table understates what is legal and cannot be trusted as documentation.
* **`prompt.py` breaks the stable-prefix rule it documents.** `_trim_history()` slices from the front, so once history exceeds `max_history_turns` every turn shifts the tokens immediately after the system prompt and invalidates the KV cache — the exact prefill regression README §50.1 predicts, arriving around turn 21.

### Tuning and hygiene
* **`stt.py` uses `beam_size=5`,** roughly 2–3× slower than greedy, against a ~200 ms STT budget. `beam_size=1` is the right default on the realtime path.
* **`synthesis/base.py` is not streaming.** `TTSProvider.synth()` returns one complete `np.ndarray`; README §14 specifies `AsyncIterator[WavChunk]`. Acceptable at M7 — clause chunking already provides most of the win — but first-syllable latency includes full synthesis of clause 1.
* **`audio_out.py` uses the global `sd.play()`/`sd.stop()`,** which cannot overlap clauses and will gap audibly between them. Matters at M8.
* **`perception/aec.py` is a pass-through stub.** Headphones required. Fine until speakers or proactive speech (README §24, §6.6).
* **Repo layout:** `Generative_Agent_bot/` is its own git repo nested inside `AI_AGENT`, recorded as a gitlink with no `.gitmodules`, so a fresh clone of the outer repo gets an empty directory. `agent.py` is 0 bytes and `testing.py` is scratch; both are superseded by `scripts/check_environment.py`.

---

# PART 2 — Environment Build Fixes

> **Date:** 2026-10-07
> **Scope:** Defects found and fixed while standing up the runtime environment (README §30) and taking the first real measurements (README §31).
> **Status:** All fixed and verified. Environment complete; `scripts/check_environment.py` passes every check.

Part 1 above removed the four blockers that prevented any run. Part 2 covers what
the install itself exposed. Four of the five below were only discoverable by
actually running things — which is the argument for doing M0 and M1 before
writing orchestration, not after.

| # | File | Defect | Failure mode |
|---|---|---|---|
| 5 | `pyproject.toml` | Flat-layout package ambiguity | `pip install -e .` refused outright |
| 6 | (new) `runtime/cuda_dlls.py` | CUDA runtime not on DLL search path | `llama_cpp` import failed, blaming the wrong DLL |
| 7 | `runtime/vram.py` | Measured torch allocator, not the driver | **Silent** — reported 0 MB with 3.9 GB in use |
| 8 | `scripts/check_environment.py` | Wrong model path; no GPU/ffmpeg/VRAM checks | False PASS on an unusable environment |
| 9 | `scripts/bench_vram.py` | Counted its own process as a GPU hog | Spurious warning on every clean run |

---

## 9. Blocker 5 — `pyproject.toml`: flat-layout ambiguity

### Problem

`pip install -e ".[stt,tts,dev]"` failed before installing anything:

```text
error: Multiple top-level packages discovered in a flat-layout: ...
setuptools will not proceed with this build.
```

The project root holds `amadeus/`, `config/`, `scripts/`, `data/` and `venv/`.
Setuptools auto-discovery cannot tell which are importable packages and refuses
to guess. Only `amadeus` is code.

### What was fixed

```toml
[tool.setuptools.packages.find]
include = ["amadeus*"]
```

The glob is deliberate — `amadeus.expression`, `amadeus.cognition.emotion` and
`amadeus.cognition.memory` do not exist yet and will be picked up automatically
when they do.

---

## 10. Blocker 6 — the CUDA runtime was not on the DLL search path

### Problem

This is the defect that took longest to diagnose, because the error names the
wrong library:

```text
RuntimeError: Failed to load shared library '...\llama_cpp\lib\llama.dll':
Could not find module ... (or one of its dependencies).
```

Nothing is wrong with `llama.dll`. Reading the PE import tables of each library
in the chain gave the real answer:

```text
llama.dll      -> MSVCP140, VCRUNTIME140, ggml-base.dll, ggml.dll
ggml.dll       -> ggml-cpu.dll, ggml-cuda.dll, ggml-base.dll
ggml-cuda.dll  -> cudart64_12.dll   <-- MISSING
                  cublas64_12.dll   <-- MISSING
                  nvcuda.dll        <-- present (driver)
```

The prebuilt CUDA wheel links against the CUDA runtime but does not bundle it.
It normally comes from the CUDA Toolkit, which is not installed on this machine —
only the driver. The MSVC runtime was already present, so that half was fine.

Worth recording the diagnostic method, because the error message actively
misleads: dump the import table of each DLL in the chain rather than trusting the
name in the exception.

### What was fixed

`torch 2.5.1+cu121` already ships the exact libraries needed, but they total
608 MB (`cublasLt64_12.dll` alone is 513.9 MB), so duplicating them into
`llama_cpp/lib/` was rejected.

New module `amadeus/runtime/cuda_dlls.py` instead adds torch's lib directory to
the process DLL search path:

```python
def ensure_cuda_dlls() -> list[Path]:
    """Add CUDA runtime directories to the DLL search path."""
    for directory in _candidate_dirs():   # torch/lib, nvidia/*/bin, CUDA_PATH/bin
        if directory.is_dir():
            os.add_dll_directory(str(directory))
```

It is idempotent, a no-op off Windows, and warns rather than raising when nothing
is found — `llama_cpp` will produce its own error and a CPU fallback is still
useful. `cognition/brain.py` calls it before the import:

```python
from amadeus.runtime.cuda_dlls import ensure_cuda_dlls

ensure_cuda_dlls()

from llama_cpp import Llama  # noqa: E402  - import order is load-bearing
```

**That ordering is a contract, not a style preference.** Importing `llama_cpp`
anywhere earlier in the process poisons it; the failed load is not retried.

Verified:

```text
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 2050, compute capability 8.6, VMM: yes
GPU offload support: True
```

---

## 11. Blocker 7 — `vram.py` measured the wrong thing

### Problem

The worst defect of the whole set, because it fails silently in the direction of
false confidence. `get_vram_info()` reported usage via:

```python
allocated = torch.cuda.memory_allocated(device)
```

`memory_allocated()` reports only what **torch's own caching allocator** holds.
Neither of the two big consumers in this project uses it:

* `llama-cpp-python` → ggml-cuda → CUDA driver directly
* `faster-whisper` → CTranslate2 → CUDA driver directly

Measured side by side during the first benchmark:

| Stage | `nvidia-smi` | driver (`mem_get_info`) | `memory_allocated()` |
|---|---:|---:|---:|
| baseline | 2100 MB | 790 MB | 0.0 MB |
| + Qwen2.5-3B loaded | 3913 MB | 3973 MB | **0.0 MB** |

The guard existed to protect a 3.6 GB budget and could not see 3.9 GB of usage.
It would have reported a comfortable margin right up to an out-of-memory crash,
and `vram_guard()` would have logged a `+0.00MB` delta around every model load.

### What was fixed

Rewritten to query the driver:

```python
free_bytes, total_bytes = torch.cuda.mem_get_info(device)
used_bytes = total_bytes - free_bytes
```

`mem_get_info` sees every allocation in the process regardless of which library
made it. `torch_allocated_mb` is kept on `VRAMInfo` as a diagnostic field, with a
docstring stating it must never drive budget decisions.

A second consequence surfaced from the same run: `mem_get_info` is **device-wide**,
not process-wide, so other applications count against the same 4096 MiB. New
`require_free()` turns that into a startup check:

```python
def require_free(min_free_mb: float, *, label: str = "startup") -> VRAMInfo:
    """Fail fast when the GPU does not have room for what we are about to load."""
```

It raises with the `nvidia-smi` command needed to identify the offender. The
rationale is in §12.

---

## 12. Why `require_free()` exists — the Monster Hunter incident

Not a code defect, but the operational finding behind two code changes, and worth
recording because the symptom is so misleading.

The first benchmark on a GPU that *looked* idle produced TTFT 1732 ms and
5.3 tok/s — roughly 6× worse than predicted. The cause:

```text
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
15864, MonsterHunterWorld.exe, 2815 MiB
```

A game held 2815 MB of 4096 MB. llama.cpp was launched with `n_gpu_layers=-1`,
found ~1.1 GB available, and **silently offloaded most layers to CPU**. No error,
no warning, no log line. Re-measured on a clean GPU:

| | Game running | Clean GPU | Ratio |
|---|---:|---:|---:|
| Time to first token | 1732 ms | 82 ms | **21×** |
| Decode | 5.3 tok/s | 41.6 tok/s | **7.8×** |

Those first numbers were discarded rather than recorded. The lesson is that the
VRAM budget in README §8 assumes effectively exclusive use of the GPU, and that
assumption degrades rather than failing — which makes it exactly the kind of thing
that must be asserted at startup. Hence `require_free()` and the free-VRAM check
in `check_environment.py`.

---

## 13. Blocker 8 — `check_environment.py` passed on a broken environment

### Problem

The original script reported `[PASS]` on an environment that could not have run
the pipeline. It checked that `faster_whisper`, `llama_cpp` and `edge_tts` were
*importable* and nothing more. Specifically it missed:

* **Wrong model directory.** It checked for `models/`, while `settings.yaml`
  points at `data/models/`. The one `[FAIL]` it did emit was for a path the
  project never uses.
* **No GGUF check** — neither presence, size, nor header.
* **No CUDA offload check.** `import llama_cpp` succeeding says nothing about
  whether the GPU backend loaded; a CPU-only wheel imports identically.
* **No ffmpeg check**, although `pydub` shells out to it. Its absence shows up as
  an empty audio array, not an exception.
* **No free-VRAM check** — see §12.
* **No pydub / pyyaml / python-dotenv checks**, all of which are imported.

### What was fixed

Rewritten with nine check groups: Python version (now bounded at ≤ 3.12, since
3.13 removes the `audioop` module `pydub` needs), torch + CUDA with compute
capability, free VRAM with a GPU-process listing on failure, llama-cpp CUDA
offload *through the DLL shim exactly as brain.py does it*, all six packages,
ffmpeg with winget-PATH detection, audio devices, GGUF presence plus magic-byte
validation, and directories.

It now returns a non-zero exit code on failure so it can gate CI, and
distinguishes `[WARN]` from `[FAIL]` — ffmpeg installed but absent from a stale
shell PATH is a warning, not a blocker.

---

## 14. Blocker 9 — `bench_vram.py` flagged itself

### Problem

The benchmark warns when other processes hold GPU memory, since that invalidates
its results (§12). But `gpu_hogs()` parsed `nvidia-smi --query-compute-apps`
without filtering, so the benchmark's own process appeared in its own warning:

```text
*** WARNING: other processes hold GPU memory. Numbers will be wrong: ***
      C:\...\python.exe, [N/A]
```

A warning that fires on every correct run trains you to ignore it — which defeats
the purpose, given §12 is exactly the failure it exists to catch.

### What was fixed

`gpu_hogs()` now queries `pid` as well and filters `os.getpid()`. The warning is
silent on a clean run.

Separately, the generation sample was lengthened from 64 to 200 max tokens with a
prompt that actually produces a long reply. The first measurement derived
22.8 tok/s from a 9-token sample; the same configuration measured 41.6 tok/s over
83 tokens. A sample that short was dominated by startup variance and understated
throughput by 45%.

---

## 15. Verification

```text
Amadeus environment check

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
[WARN] ffmpeg on PATH - installed at ...ffmpeg.exe but not on this shell's PATH - restart the terminal
[PASS] audio devices - in: Microphone (Realtek(R) Audio) | out: Headphones (Realtek(R) Audio)
[PASS] Qwen2.5-3B GGUF - 2007 MB, valid GGUF header
[PASS] directory data/models
[PASS] directory data/logs
[PASS] directory config

All checks passed.
```

And the measurement that proves the stack works end to end on the GPU — full
record in README §31:

```text
faster-whisper small        +370 MB   load 1.81s
Qwen2.5-3B Q4_K_M          +2210 MB   load 1.25s
TTFT                        82 ms
decode                      83 tok in 1.99s = 41.6 tok/s
peak VRAM                   3398 MB / 4096 MB (83.0%)
headroom                    698 MB
budget                      WITHIN target of 3686 MB
```

**What this does not prove.** No microphone has been opened, no speech
transcribed, no audio played, and no turn run end to end. `amadeus/main.py` is
still empty. What is proven is that both GPU models load together inside the
budget and the LLM generates at usable speed.

---

## 16. Outstanding after Part 2

Unchanged from §8 of Part 1 except where noted.

### Blocking the first run
* **`amadeus/main.py` is still empty.** Now the only thing between here and M7.
* ~~Dependencies missing~~ — **done**, see README §30.
* ~~No model weights~~ — **done**, 2,104,932,768 bytes verified.
* ~~ffmpeg~~ — **done**, ffmpeg 9.0.2; restart the terminal for PATH.

### Divergences from the spec, still cheap to fix
* **`state.py` has no `IDLE → THINKING` edge.** Proactive speech remains
  structurally impossible. README §6 and §19.
* **`state.py` transition validation hole** — any transition to `IDLE` bypasses
  the table.
* **`prompt.py` front-trims history,** invalidating the KV cache prefix from
  roughly turn 21. README §50.1. Now measurable: at 790 MB for a CUDA context and
  41.6 tok/s decode, prefill is the remaining unknown on the hot path.

### Tuning and hygiene
* **`stt.py` uses `beam_size=5`** — roughly 2–3× slower than greedy on the
  realtime path.
* **`synthesis/base.py` is not streaming** — returns one array, not an
  `AsyncIterator[WavChunk]`.
* **`audio_out.py` uses global `sd.play()`/`sd.stop()`** — cannot overlap clauses.
* **`perception/aec.py` is a pass-through stub.** Headphones required.
* **`settings.yaml` has `n_ctx: 2048`** against the 4096 assumed in README §8.
  Headroom now measured at 698 MB, so raising it is affordable — but measure the
  KV delta first.
* **`synthesis/edge.py` could drop `pydub` + ffmpeg** in favour of `av`, already
  installed as a faster-whisper dependency. Removes an external binary and the
  Python 3.13 `audioop` problem.
* **Repo layout:** `Generative_Agent_bot/` is still a nested git repo recorded as
  a gitlink with no `.gitmodules`. `agent.py` is still 0 bytes. `testing.py` was
  deleted by something outside this session and remains unstaged.

---

# PART 3 — First Run (M7)

> **Date:** 2026-10-07
> **Scope:** Writing `amadeus/main.py` and everything that broke when the pipeline ran for the first time.
> **Status:** Text path runs end to end. Voice path initializes and listens; real speech untested.

Parts 1 and 2 were found by reading code and by installing. Part 3 is different:
every defect below was found by **running the thing**. Two of the three were
invisible to static inspection, and one of them was a bug in a fix from Part 1
that I had wrongly reported as verified.

| # | File | Defect | Found by |
|---|---|---|---|
| 10 | `state.py` | No `IDLE -> THINKING` edge; validation hole | First run, crashed immediately |
| 11 | `perception/stt.py` | Un-awaited `bus.publish` — **missed by the Part 1 sweep** | Re-scan before wiring events |
| 12 | `bus.py` | Cancelled publishes logged as errors | Teardown noise on first run |

Plus two additions: `TurnTrace` / `LatencyLog` in `trace.py`, and `main.py` itself.

---

## 17. Correction to Part 1 — the sweep was wrong

Part 1 §3 claimed:

> A regex sweep over the whole `amadeus/` package confirms no other un-awaited
> `bus.publish(...)` remains.

**That claim was false.** The pattern used was:

```python
r"(?<!await )(?<!\.)\bbus\.publish\("
```

The second lookbehind, `(?<!\.)`, asserts the character before `bus` is not a
dot. But every real call site is written `self.bus.publish(...)` — where the
preceding character *is* a dot. The pattern therefore excluded precisely the
calls it was meant to find, and matched nothing because there was nothing to
match. A sweep that cannot match any real call site reported "clean" and I
believed it.

The correct check is trivial and was run before wiring events in `main.py`:

```bash
grep -rn "bus\.publish(" amadeus/ | grep -v await | grep -v publish_sync | grep -v "def "
```

Which found one survivor, documented in §19.

Lesson worth keeping: a negative-result sweep is only evidence if the pattern is
first shown to match a known positive.

---

## 18. Blocker 10 — `state.py` rejected the first real transition

### Problem

The very first run of `main.py` died on the first turn:

```text
ValueError: Invalid state transition from IDLE to THINKING
```

The transition table had no `IDLE -> THINKING` edge:

```python
SystemState.IDLE: {SystemState.LISTENING},
```

Two separate things need that edge:

* **`--text` mode** has no LISTENING phase at all. There is no microphone and no
  VAD, so a typed turn goes straight from IDLE to THINKING.
* **Proactive initiative** (README §19) is defined by exactly this edge. It is
  the only transition in the system not caused by the user, and §6 calls it the
  thing that separates this project from a voice assistant.

This was flagged in Part 1 §8 as "cheap to fix now, a refactor later". It
surfaced on the first run, which is the cheapest possible moment.

A second defect in the same method: validation was bypassed for any transition
into IDLE.

```python
if new_state != SystemState.IDLE and new_state not in self._valid_transitions[self._current]:
    raise ValueError(...)
```

So `SPEAKING -> IDLE` was legal, and so was any other `X -> IDLE`, whether or not
the table said so. The table therefore understated what was actually permitted
and could not be trusted as documentation of the state machine.

### What was fixed

Table now mirrors README §6 and is authoritative:

```python
self._valid_transitions = {
    SystemState.IDLE:      {SystemState.LISTENING, SystemState.THINKING},
    SystemState.LISTENING: {SystemState.THINKING, SystemState.IDLE},
    SystemState.THINKING:  {SystemState.SPEAKING, SystemState.LISTENING, SystemState.IDLE},
    SystemState.SPEAKING:  {SystemState.IDLE, SystemState.LISTENING},
}
```

`THINKING -> IDLE` was added so an empty transcript or a failed turn can unwind
without faking a SPEAKING state. The IDLE bypass is gone — every edge now lives
in the table. Self-transitions return early as a no-op so callers can transition
defensively. The error message now lists the legal destinations:

```text
Invalid state transition LISTENING -> SPEAKING. Legal from LISTENING: IDLE, THINKING
```

Verified: `IDLE -> THINKING` succeeds, `THINKING -> IDLE` succeeds,
`LISTENING -> SPEAKING` is rejected.

**Note on scope.** This adds the *edge*, not the feature. `drive.py` does not
exist, so nothing ever takes `IDLE -> THINKING` except `--text` mode. The state
machine no longer forbids proactive speech; it still does not produce it.

---

## 19. Blocker 11 — `stt.py` dropped its transcript event

### Problem

The survivor from §17:

```python
# amadeus/perception/stt.py:52
if self.bus:
    self.bus.publish("TranscriptReady", {"text": text})
```

Inside `async def transcribe`, un-awaited. Identical in kind to Blocker 2, and it
had been sitting there through all of Part 1 and Part 2 because the sweep that
was supposed to catch it could not match it.

Consequence: `TranscriptReady` never fired. Nothing subscribes to it yet, which
is the only reason this had not caused a visible failure — but `main.py` was
about to.

### What was fixed

```python
if self.bus:
    await self.bus.publish("TranscriptReady", {"text": text})
```

Re-scanned with the corrected grep: no un-awaited publishes remain anywhere in
`amadeus/`.

### Also changed here

`beam_size` lowered from 5 to 1:

```python
segments, info = self.model.transcribe(
    audio,
    language=self.language,
    # Greedy. beam_size=5 is roughly 2-3x slower for marginal
    # accuracy gain and this sits on the realtime path.
    beam_size=1,
)
```

Flagged in Part 1 §8. Fixed before the first latency measurement so the numbers
would not be misleading from the start.

---

## 20. Blocker 12 — `bus.py` logged normal teardown as an error

### Problem

A defect introduced by my own Part 1 fix. `publish_sync` schedules through
`run_coroutine_threadsafe` and attaches a done-callback to surface subscriber
exceptions. But when the event loop shuts down before a queued publish runs, the
future is **cancelled**, and `CancelledError` is an `Exception` subclass, so:

```text
ERROR:root:Error publishing event StateChanged:
concurrent.futures._base.CancelledError
```

Seven of these on a clean three-transition test. Normal teardown rendered as a
wall of error tracebacks, which is the fastest way to teach someone to ignore
their own error log.

### What was fixed

```python
def _report(f: Any) -> None:
    try:
        f.result()
    except concurrent.futures.CancelledError:
        # The loop shut down before this publish ran. Normal during
        # teardown - the last few StateChanged events are simply dropped.
        pass
    except Exception as e:
        logging.exception(f"Error publishing event {event_name}: {e}")
```

Genuine subscriber exceptions are still reported.

---

## 21. Addition — per-turn latency instrumentation

`Tracer.span()` measures one block. The number that actually matters — end of
user speech to first audible syllable — spans five components, so there was
nothing capable of measuring the system's headline metric.

Added to `trace.py`:

* **`TURN_STAGES`** — the canonical ordered stage list, with `speech_end` as t=0.
* **`TurnTrace`** — stamps monotonic timestamps per stage, exposes
  `first_syllable_ms`, a per-stage `breakdown()`, and a console `table()`.
* **`LatencyLog`** — aggregates turns, computes p50/p95, and compares against the
  targets in `settings.yaml`.

This is what `scripts/bench_latency.py` will consume (README §31.4). It lives in
`trace.py` rather than `main.py` per README §47.

---

## 22. `amadeus/main.py`

Written. 359 lines. An `Amadeus` class owning every component, with `setup()`,
`run_turn()`, `run()` and `shutdown()`.

Both contracts from Part 1 §6 are honoured explicitly and commented as such:

```python
async def setup(self) -> None:
    # CONTRACT 1: must be first, before anything can publish_sync.
    self.bus.bind_loop()
    ...
    # CONTRACT 2: start() binds the loop used by the PortAudio callback.
    await self.mic.start()
```

`require_free(2600)` gates startup, using the floor measured in README §31.3
(Whisper 370 MB + Qwen 2210 MB). Perception imports are local to `setup()` so
`--text` mode pays for neither CUDA nor PortAudio.

CLI: `--text`, `--turns N`, `--log-level`.

---

## 23. First measurements

### Text mode — a complete turn

```text
you > Hello! Who are you?
[first audio at 2605 ms]
Amadeus > Hello! I'm Amadeus, a friendly AI here to chat and explore.
Amadeus > How can I assist you today?

  turn 1
    speech_end -> stt_done                   0 ms
    stt_done -> prompt_built                 0 ms
    prompt_built -> first_token            306 ms
    first_token -> first_clause            390 ms
    first_clause -> tts_done              1909 ms
    tts_done -> first_audio_out              0 ms
    = first audible syllable              2605 ms

  p50 target 1200 ms -> OVER
```

**Over target, with an unambiguous cause.** Edge-TTS is 1909 ms of the 2605 ms —
73% of total latency in a cloud round trip. Everything local sums to ~700 ms:

| Component | Time | Local? |
|---|---:|---|
| LLM prefill + first token | 306 ms | yes |
| chunker accumulating a clause | 390 ms | yes |
| **Edge-TTS synthesis** | **1909 ms** | **no — network** |

This is the cost README §28 predicts for the M7 bring-up voice. Kokoro-82M at
Phase 2 replaces it with local CPU synthesis. No optimisation of the local
pipeline is worth attempting before that swap — it would be tuning 27% of the
problem.

The `first_token -> first_clause` gap of 390 ms is the chunker waiting for
`MIN_CLAUSE_LENGTH` plus punctuation. Worth revisiting once TTS is local, since
at that point it becomes a meaningful share.

### Voice mode — initializes and listens

```text
VRAM check passed [startup]: 3306MB free, 2600MB required
Qwen model load   +2210MB  -> 1096MB free
WhisperSTT Load    +288MB  ->  808MB free
Silero VAD model loaded.
Microphone stream started.
State transition: IDLE -> LISTENING
VAD listening...
```

All three models resident at **3288 MB / 4096 MB, 808 MB free**. Within the
3686 MB budget.

Note Whisper measured **+288 MB** here versus **+370 MB** in `bench_vram.py`. The
benchmark forces a dummy decode so CTranslate2 allocates its working buffers;
`main.py` measures straight after construction, before any transcription. 370 MB
is the number to budget against.

**Not verified: actual speech.** The microphone opens and the VAD listens, but no
utterance has been captured, transcribed, or answered. Voice mode is unproven
past initialization.

---

## 24. Outstanding after Part 3

### Blocking a complete M7
* **Voice path untested with real speech.** Needs a human at the microphone.
* **No barge-in.** The `stop` Event is plumbed through every stage but nothing
  sets it mid-turn, and the VAD does not listen during SPEAKING. M8.

### Now the dominant latency problem
* **Edge-TTS costs 1909 ms.** Replacing it with Kokoro-82M is the single highest
  value change available and the gate on hitting the 1200 ms target.
* **`synthesis/base.py` is still not streaming** — one array per clause, so
  first-audio includes full synthesis of clause 1. Compounds the above.
* **`audio_out.py` uses global `sd.play()`/`sd.stop()`** — cannot overlap
  clauses; audible gaps between them.

### Spec divergences remaining
* ~~`state.py` missing `IDLE -> THINKING`~~ — **fixed**, §18.
* ~~`state.py` validation hole~~ — **fixed**, §18.
* **`prompt.py` front-trims history,** invalidating the KV prefix from roughly
  turn 21. Untouched and now the last of the three. README §50.1.
* **`perception/aec.py` is a pass-through stub.** Headphones mandatory.

### Hygiene
* **`settings.yaml` has `n_ctx: 2048`** against 4096 in README §8. Headroom is
  808 MB, so raising it is affordable — measure the KV delta first.
* **`synthesis/edge.py` could drop `pydub` + ffmpeg** for `av`, already installed.
* **Nothing is committed.** 13 modified, 3 untracked.
* **Repo layout:** nested git repo with no `.gitmodules`; `agent.py` still 0
  bytes; `testing.py` deleted outside this session and still unstaged.
* **No avatar.** Nothing in the repo renders anything. README §26 / M14.

---

# PART 4 — Local voice, pipelined playback and the first real latency numbers (M6 / M8)

> **Date:** 2026-11-04
> **Scope:** Replace the cloud voice with a local one, then bring first-sound latency down to target without making speech stutter.
> **Status:** Both latency targets met in text mode. **Not yet verified by ear, and not yet measured with real speech through Whisper** (see §31).

Part 3 ended with the pipeline working but slow: 2605 ms to first sound against a
1200 ms target, 73% of it one cloud call. This part is what it took to fix that, and
it is mostly a record of assumptions that measurement overturned.

| # | Where | Finding | Outcome |
|---|---|---|---|
| 13 | `synthesis/kokoro.py` | The planned int8 voice model is ~3x **slower** than fp32 on this CPU | switched to fp32 |
| 14 | `main.py` `_respond` | Sequential synth→play→synth put a silence before **every** clause | pipelined |
| 15 | `synthesis/chunker.py` | First clause = whole first sentence; no early break point | `first_clause_min/max` + backtracking |
| 16 | `synthesis/audio_out.py` | Real player adds **~200 ms dead air per clip** (invisible to the benchmark) | persistent WASAPI stream |
| 17 | (my own records) | Three claims I had written down were wrong | corrected, §32 |

New: `synthesis/kokoro.py`, `scripts/bench_latency.py`, `tests/` (59 tests), `NullPlayer`,
`StreamingAudioPlayer`, `TurnTrace` gap tracking, `--mute`, seeded benchmarking.

---

## 25. Kokoro int8 was the wrong model (13)

The plan said "Kokoro-82M int8 on CPU" and estimated a real-time factor of ~0.25.
First measurement, same 4.65 s sentence:

| Model | Time | RTF | vs Edge-TTS (1909 ms for this sentence) |
|---|---:|---:|---|
| int8, 4 threads (best of a sweep) | 5242 ms | 1.13 | **2.7x slower** than the cloud |
| fp32, 6 threads | 1668 ms | 0.36 | 13% faster |

Quantising to save compute made it slower. Evidence rather than guesswork: onnxruntime's
per-operator profile of the int8 graph:

```text
ConvInteger   4571 ms   83.1%   87 calls
Sin            220 ms    4.0%
STFT           158 ms    2.9%
```

`ConvInteger` is a slow generic CPU kernel. The same convolutions in the fp32 graph run as
the optimised float `Conv` (972 ms, 53%). The signature of a serial kernel was already
visible in the thread sweep: int8 took 6051 ms on 1 thread and 6059 ms on 12; fp32 scaled
properly (3319 ms on 1 thread, 1668 ms on 6).

Changes: the model file and `config.py` default are fp32; `KokoroTTS` builds its own
`InferenceSession` restricted to `CPUExecutionProvider` (the stock constructor resolves
providers itself and would pick CUDA if `onnxruntime-gpu` were ever installed, silently
spending the VRAM this voice exists to avoid); thread count is a setting (`tts.threads: 6`,
the physical core count, measured faster than 12); a warm-up synth at load removes a
one-time ~195 ms penalty from the user's first turn; `kokoro_voice` is separate from
`voice` because Edge and Kokoro voice names are incompatible and sharing one field made
switching provider fail on an unknown voice. `check_environment.py` now verifies Kokoro
file sizes exactly (an earlier int8 download had stopped at 241,954 bytes without any
error) and warns if the int8 model is selected.

VRAM change from loading Kokoro: **+0 MB**, measured.

---

## 26. Sequential playback put silence before every clause (14)

`_respond` ran synth → play → synth, so for every clause after the first the listener heard
silence for as long as synthesis took. Baseline, 16 turns (8 prompts x 2), real LLM and
Kokoro, simulated playback:

```text
first syllable   p50 1840 ms | p95 2704 ms | max 2989
longest silence  median 1111 ms | worst 3110 ms | turns with a >300 ms hole: 14 of 14
```

Every multi-clause reply had an audible hole. The fix is a producer/consumer pair with a
one-slot queue: the synthesiser works on clause N+1 while clause N plays.

```text
original rules, pipelined:
longest silence  median 0 ms | worst 795 ms | turns with a >300 ms hole: 2 of 14
first syllable   p50 1896 ms   (unchanged: pipelining cannot help the first clause)
```

The queue holds one clip so synthesis runs at most one clip ahead; deeper would waste CPU
on audio an interruption would throw away. The sentinel is only put on normal completion,
because a `put()` in a `finally` could block forever on a full queue with no consumer. On
any failure `_respond` cancels both tasks and re-raises. Tests cover ordering, overlap, a
real hole being reported, failure with no hang, and stop mid-reply (`test_respond_pipeline.py`).

---

## 27. The first clause decides first-sound latency (15)

After pipelining, time to first sound is almost exactly the synthesis time of the first
clause, and replies differ hugely in how early they offer a break. Measured cost model for
this CPU (three clause lengths, interleaved runs):

```text
synthesis ~= 440 ms fixed + ~19 ms per character      (ONNX inference; nothing outside it matters)
speech    ~= 15 characters per second
```

Two constraints follow. For first sound under 1200 ms the opener must be **at most ~35
characters**. For no pause before clause 2, clause 1 must **play at least as long as clause
2 takes to synthesise**, which needs **~19+ characters** for a typical next clause. The
workable window is roughly 19 to 35 characters. `Hello!` (6 characters, 0.8 s of audio)
cannot cover even the fixed cost of the next clause, so a bare interjection always leaves a
hole.

Sweep (16 turns, identical seeded replies, pipelined):

| Config | p50 | p95 | silence p90 / worst | turns >300 ms | ≤1200 ms | ≤1800 ms |
|---|---:|---:|---:|---:|---:|---:|
| baseline (sequential) | 1840 | 2704 | 1768 / 3110 | 14/14 | 5/16 | 8/16 |
| original rules | 1896 | 2659 | 505 / 795 | 2/14 | 4/16 | 6/16 |
| `min=12` | 1720 | 2533 | 253 / 397 | 1/14 | 6/16 | 8/16 |
| `min=0` | **724** | 1966 | 1234 / 1380 | 7/16 | 10/16 | 12/16 |
| `min=18` / `24` / `32` | 1763 / 1963 / 1965 | 2503 / 2810 / 2761 | ≤425 worst | 1/14 each | — | — |
| cap 16/30 | 1106 | 1303 | 651 / 1227 | 2/16 | 12/16 | 16/16 |
| **cap 16/30 + backtracking** | **1043** | **1246** | 562 / 1010 | 2/16 | **12/16** | **16/16** |
| cap 16/34 + backtracking | 1167 | 1565 | 682 / 1238 | 2/16 | 8/16 | 16/16 |

Why a threshold alone is not enough: many replies open with a 60 to 110 character sentence
that contains no punctuation, so no `min` finds a break. `min=0` is fast but leaves a hole in
7 of 16 replies (`Sure!` then a 77 character sentence). The fix is an **opt-in cap**
(`first_clause_max`): if the first clause reaches the cap with no punctuation, end it at the
next word boundary. A token that starts with a space marks a finished word, so a clause is
never cut inside a word (tested with sub-word pieces).

A plain cap makes some poor cuts (`...a beloved stimulant that | energizes`,
`...as active and | curious`). **Backtracking** fixes the common case: when the cap lands
right after a function word (`that`, `and`, `the`, `of`...), the cut moves back before it and
the word is carried into the next clause (`...a beloved stimulant | that energizes`). The
opener gets shorter, not longer. It is bounded by a floor so the opener cannot collapse, and
a property test checks that **no text is ever lost or duplicated** across 12 min/cap
combinations. It cannot fix `personal | preferences` (adjective-noun); no word list can.

Shipped default: `chunker: first_clause_min_chars: 16, first_clause_max_chars: 30`, in
`settings.yaml`. Set `first_clause_max_chars: null` to remove all mid-sentence seams; the
cost is p50 back to ~1700 ms.

---

## 28. The real player added ~200 ms of silence per clip, invisible to every benchmark (16)

`NullPlayer` models when a clip starts and ends. The real `AudioPlayer` also calls
`sd.play(blocking=True)` once per clip, and Windows' default host API here is **MME**.
Measured with silent clips through the real device:

```text
legacy sd.play    8 plays of a 0.5 s clip: overhead min 197 | median 201 | max 214 ms per clip
```

So the gap numbers in §26 and §27 understated what a listener would have heard by roughly
200 ms per clause boundary, and the cap deliberately creates more boundaries. Stream latency
by host API, same machine:

```text
MME      93.3 ms
WASAPI   24.8 ms   (fails outright at 24 kHz: "Invalid sample rate [-9997]" unless auto_convert=True)
```

`StreamingAudioPlayer` keeps one output stream open; its callback drains a queue, so a clip
queued while the previous one plays follows it sample for sample. WASAPI shared mode with
`auto_convert` is preferred on Windows, falling back to the default device. `play()` returns
`lead_s` (50 ms) **before** its clip ends so the next clip can be queued in time; `drain()`
waits for the real end. `NullPlayer` models both, so benchmarks stay faithful. The legacy
player remains behind `audio.streaming_output: false` because this change cannot be
verified by ear from here.

Real-device check, 6 silent 0.5 s clips:

```text
legacy sd.play        wall 4.210 s   dead air 202 ms per clip
streaming (WASAPI)    wall 2.998 s   dead air   0 ms per clip   (stream latency 25 ms)
```

Measured but not yet fixed: the default Windows output is currently **Speakers**, not
headphones. Windows changes the default when devices are plugged in. With the AEC stub this
means the agent can hear itself (README section 24).

---

## 29. Tests, and two that could not fail (and now can)

`tests/` did not exist. It now has 59 passing tests and 2 strict xfails: chunker behaviour
(including a decimal `3.5` and a clock time `10:30`, both split mid-number today and recorded
as strict xfails), the pipelined `_respond` against fake components, the streaming player
against a fake device thread that calls the real callback, and `NullPlayer`.

I checked that the tests could actually fail, by running the defect through them:

- The old **sequential** loop through the overlap test: wall 1.68 s and gaps of 156 ms against
  thresholds of 1.55 s and 60 ms, so the test fails it. The pipelined loop gets 1.39 s and 0 ms.
- The **gapless-player** test passed even with `lead_s=0`, so it proved nothing about the
  lead. Both test clips were exactly 15 blocks long, so the first always ended on a block
  boundary. With block-unaligned clips, `lead_s=0` inserts 440 samples (18 ms) of silence and
  the test now fails it; `lead_s=0.05` inserts 0.
- One assertion (`span[~(span != 0)] == 0`) selected the zeros and checked they were zeros. It
  is replaced with a check on the samples between the two clips.

---

## 30. Final numbers

Shipped defaults, `scripts/bench_latency.py --label 6-final-defaults`, 16 turns, simulated
playback that models the streaming player:

```text
first syllable : p50 1062 ms | p95 1275 ms | min 843 | max 1301     PASS (1200 / 1800)
silence        : median 0 ms | 4 turns with a 167-190 ms pause | 2 turns ~1 s (see below)
stage cost     : first token 33 ms, chunker 181 ms, first-clause synthesis 847 ms (medians)
```

One real turn through the real streaming player on the real device (typed input):

```text
first audible syllable 1095 ms (2605 ms at the start of this session)
silence between clauses [0, 0] ms
```

---

## 31. Caveats that decide how far to trust §30

* **Typed input.** Speech-to-text time is **not included**. Real voice-mode first syllable is
  these numbers **plus Whisper's time**, which has never been measured on real speech.
  README's estimate (~100 ms for a 3 s utterance) would put p50 near 1160 ms, which is
  borderline against 1200. The targets are met for the pipeline from transcript onward, not
  yet end to end from the microphone.
* **Nobody has listened to it.** The mid-sentence seams of the cap and the new output path are
  judged by ear, and I cannot. `first_clause_max_chars: null` and `audio.streaming_output:
  false` revert them.
* **Eight prompts, two repetitions of the same seeded replies.** The repetitions measure
  timing noise only. Differences of a few percent between configurations are not evidence.
* **One machine**, one CPU, one output device, WASAPI at 25 ms. Stream latency is not included
  in "first syllable" (the stamp is taken as audio is handed to the player).
* **Remaining silence.** A reply whose later clause is a long unpunctuated run-on (the "sky is
  blue" reply: 150 characters) still pauses ~1 s before it, because the cap only governs the
  first clause. A cap on later clauses relative to the previous one's length would address it.

---

## 32. Corrections to what I told you earlier

* **Kokoro's real-time factor is 0.36 (fp32), not ~0.25 (int8).** README sections 4.1 and 8
  carried the wrong figure and the wrong model.
* **"152 ms of phonemizing on every clause" was wrong.** It is a one-time ~141 ms start-up (espeak
  loading its data); afterwards 0.2 ms per call. `KokoroTTS` already warms it at load. I had
  also logged it as a task (M8.6). The actual floor is ~440 ms inside the neural network.
* **"ORT_DISABLE_ALL is ~10% faster" was noise** from one reading each. Eight interleaved
  rounds: +0.2% overall, wins split 4/8, 4/8, 3/8. Not adopted.
* **Gap numbers before §28 understated real silence by ~200 ms per boundary.**

---

## 33. Outstanding after Part 4

* **Listen to it.** Mid-sentence seams, the new output path, and whether the pauses sound
  natural. Needs you, on headphones.
* **Voice path end to end:** real speech through the microphone and Whisper (M3, M4, M7).
* **Barge-in** (M8.7), which now has the right primitive (`stop_immediately()` flushes the
  stream) but no trigger, and real AEC (M3.3).
* **Long run-on clauses** still pause (§31).
* **Chunker splits decimals and clock times** (xfail tests); **phonemizer prints
  `words count mismatch on 100.0% of the lines`** for some clauses, not yet investigated.
* **Uncommitted:** everything in this part. README sections 4.1 and 8 still carry the old
  Kokoro figures; the Word reference and the architecture page still show Edge-TTS.

---

# PART 5 — The Kurisu voice investigation, emotion (M9) and avatar (M14)

Written 2026-11-10. Three pieces of work, and in each one the first number or the first green test run was
wrong in a way only a second look exposed. That is the real content of this part.

## 34. Can the statistics of Kokoro's voice be moved to match the reference clip?

**Question.** You picked `audio (1).wav` (English, 24 kHz, 6.7 s) as the voice you want and asked whether Kokoro's
statistics can be fixed to sound like it, and said to move on if the cost was too high.

**Measured gap.** Reference: median pitch 296 Hz, range 8.3 semitones, spectral centroid 562 Hz. Kokoro's 15
female voices: 143 to 208 Hz. The nearest is +6 semitones away, `af_heart` (current) +7.4. No blend of voices can
reach it, because a blend lies between its parts.

**A way to measure "same voice".** I installed two speaker-identification models (WeSpeaker ResNet34, 26 MB each, run
through onnxruntime; one trained on VoxCeleb, one on CN-Celeb) and calibrated them on your three clips. Cosine
similarity of the same character across languages: 0.66 (VoxCeleb model) and 0.82 (CN-Celeb model). Raw Kokoro
voices to the target: 0.10 to 0.43 on all 15 with the VoxCeleb model; the CN-Celeb model I only ran on the handful
of voices in the second table below (0.32 to 0.65).

**What I tried, and what each gave (scored by the VoxCeleb model, the first one I had):**

| Approach | Similarity | Cost at run time |
|---|---:|---|
| `af_heart` raw (current) | 0.31 to 0.34 | none |
| Praat pitch shift to 296 Hz (+ formant ratio 1.0) on `af_nicole` | 0.35 to 0.49 | 7 ms per second of speech |
| Raising the formants as well | worse in every case | |
| **Search of Kokoro's 256-number style vector** (16 axes, 193 evaluations, 9 min), judged on lines it was not tuned on | **0.55 to 0.56**, Whisper word error 0% | none (a saved array) |

**The correction.** I reported "0.31 to 0.55" as a result. It was not. Scored by the *second* model, which the search
never saw, the searched voice was **worse than the current one** (0.44 against 0.48), and Praat shifting hurt
(0.29 against 0.32). The search had learned to please the first model, not to sound like her. Held-out *sentences*
do not protect against that; a held-out *judge* does, and it showed the gain was mostly artefact.

| Voice (4 held-out lines) | VoxCeleb model | CN-Celeb model |
|---|---:|---:|
| `af_heart` (current) | 0.31 | 0.48 |
| `af_nicole` + Praat 296 Hz | 0.35 | 0.29 |
| searched style vector | 0.56 | 0.44 |
| **`bf_isabella` raw** | 0.40 | **0.65** |

**What survives.** Both models rank `bf_isabella` best among raw Kokoro voices, and both say Praat pitch shifting does
not reliably help. Everything else here is one model's opinion.

**What cannot be measured.** How it sounds. I cannot hear, and two speaker models disagreeing shows the proxies are not
trustworthy enough to decide. So I built a **blind listening pack**: `data/audition/blind/reference_TARGET.wav` plus
`clip_1..4.wav` (the four candidates in a shuffled order, same sentence, same loudness, −20.5 dBFS). The key is kept
outside the project and unread. Play it with `scripts/play_wavs.py --blind` and tell me which clips sound closest.

**Verdict on cost.** The cheap routes are exhausted: pitch shifting does not help, and a style-vector search overfits
whatever judge it is given. The only zero-cost change with support from both judges is switching to `bf_isabella`,
which I have **not** applied because you have not heard it. Real fidelity needs the teacher (the 1.7B Qwen3-TTS Kurisu
model, on a rented or free GPU) and either a distilled small model or a remote voice. That is a decision for after
you have listened.

## 35. M9 — what the first run got wrong

Tests were green first time (the logic did what I wrote). Replaying real conversations through it found:

| Found by | Defect | Fix |
|---|---|---|
| running the phrases | negation broken: "You're **not** stupid", "I'm not sad" matched nothing, because the negator is *inside* the phrase | patterns allow a negator; detection looks inside the match as well as before it |
| running the phrases | "I can't believe you're so smart" counted as negated praise | `can't` / `won't` dropped from the negator list |
| a decay table | a fading *flustered* passed through *curious* and *concerned* | the label's name is frozen between events; only intensity fades |
| the live 6-turn run | `Hey Christina` came out **curious 1.00**: arousal pinned at +1.0 after two events | diminishing returns near the bounds (shrink a delta by the room left, never enlarge it); the label now follows the event, and renames the mood only if the event is at least 60% as strong |
| the phrases again | "Shut up!" came out *annoyed* once deltas shrank | separate `hostility` rule, harsher than an insult |

**Does the mood change what she says?** Controlled test: same prompt and seeds, with vs without the hint, 7 moods x 3
held-out user lines x 3 seeds. **0 of 63** replies leaked the mood or the instructions. Short social replies took on
some colour; a factual question opened "Sure!" in every mood, angry included. No null-control was run, so no effect size
is claimed. The hint is harmless and slightly useful on small talk. The 3B model's habit of copying the persona's example
lines word for word hides it further. M15 (persona LoRA) is the real fix.

**Mutation checks.** 14 deliberate breakages of the M9 logic, each caught by a named test. (My first attempt skipped
one because the match string did not fit the file's whitespace; I reran it rather than count it.)

## 36. M14 — what looking at it found

Unit and integration tests were green. Rendering the face, then recording the real application driving it, found three
more defects:

1. **Layering.** A dark band between hair and shoulders.
2. **"Concerned" had angry brows.** Brow direction came from dominance, and her resting dominance is high, so concern read
   as anger. It now follows the label (worried vs angry), with a regression test.
3. **The face disagreed with the label.** From a live recording: 14 s after a compliment, "Hey Christina" gave the label
   *annoyed 0.78* but a face with blush 0.57, a smile and brows barely furrowed. The face followed the accumulated PAD;
   the label followed the event. Each label now has an expression preset blended in by intensity. A test replays that
   exact sequence through the real engine and model; with the presets disabled it fails.

**Mutation checks.** 13 breakages of the avatar code: 12 caught, **1 missed**, and the miss was a real hole in a test
(every range tested had its neutral value at the midpoint, so "centre on the default" and "centre on the midpoint" were
indistinguishable). Fixed with a range whose default is not the midpoint, then re-checked. Later, the preset blend got
its own check (caught).

**Measured.** 30.1 fps to a browser (p99 inter-frame gap 48 ms); tick cost p50 0.25 ms, max 0.68 ms of a 33 ms budget;
real app, three turns, first sound p50 1102 / p95 1668 ms (targets 1200 / 1800: still met with emotion and avatar on).

**VTube Studio.** Written from the official README. It discovers the model's parameters and scales each value into the
parameter's own reported range, because the README does not list the default parameter names and I would not guess.
Fourteen tests against a mock server that speaks the documented protocol (pairing, token reuse and revocation, denial,
missing parameters, scaling, hotkeys, not-running-yet, restart and reconnect). **Never run against a real VTube Studio.**

## 37. A tooling accident worth recording

I patched `run.md` with a shell heredoc. The shell turned `\\` into `\`, so Python read `.\venv` as a **vertical tab**
plus `env`, and `scripts\avatar_demo` lost its `\a` to a bell character. Ten invisible control characters would have
broken every copy-pasted command in the new sections. Found by reading the result back, repaired with a script, and
the whole project scanned for control characters (none left). Documents are now edited with the file-edit tool, not
heredocs.

## 38. Corrections to what I told you earlier

* **"Style search lifts similarity from 0.31 to 0.55"**: retracted, see §34.
* **"`run.md`: voice path unproven"** was stale: you tested M3, M4 and M7 live and accepted them. Updated.
* **The plan implied the mood hint would visibly steer a 3B model.** Measured: barely, and not at all on factual answers.

## 39. Outstanding after Part 5

* **Listen to the blind pack** (`scripts/play_wavs.py --blind`) and say which clips sound closest to the reference. Then
  decide: switch to `bf_isabella`, or invest in the teacher/student route, or leave the voice.
* **M14 on a real Live2D model** in a real VTube Studio: the one piece never exercised.
* **Persona fidelity** (M15) is still the cap on how in-character she sounds.
* **Whisper in-app time** (~580 ms vs 190-310 ms standalone) is still unexplained.
* **Tracker (`.xlsx`), the Word reference and the architecture page are stale**: no M9, M14, voice work, or Kokoro.
* **Uncommitted and unpushed** (you have not asked): everything from M8 onward on this branch, i.e. `.gitignore`,
  `main.py`, `config.py`, `settings.yaml`, `emotion.yaml`, `persona.yaml`, `amadeus/emotion/`, `amadeus/expression/`,
  `scripts/audition_voices.py`, `scripts/play_wavs.py`, `scripts/avatar_demo.py`, and 205 passing tests (2 expected xfail).
