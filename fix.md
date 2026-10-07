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
