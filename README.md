# Amadeus AI System

## Real-Time Interactive Conversational Agent

> **Project status:** Architecture / Development Blueprint  
> **Target:** Local real-time AI companion running on a single laptop  
> **Primary constraint:** RTX 2050 4 GB VRAM, i5-11400H, 16 GB RAM

---

# 1. Project Overview

Amadeus is a real-time embodied conversational AI system designed to behave less like a traditional chatbot and more like a continuously present digital companion.

The target behavior is:

- Continuous listening instead of strict turn-taking
- Natural interruption / barge-in
- Streaming speech generation
- Continuous emotional state
- Memory and long-term user modeling
- Proactive conversation
- Context-aware initiative
- Expressive voice
- Live2D / VTube Studio avatar synchronization
- Local-first execution under strict hardware constraints

The project is inspired by the idea of an emotionally expressive digital human with a stable personality rather than a simple question-answering assistant.

The system must be designed around **real-time interaction first**, not around maximizing model size.

---

# 2. Hardware Constraints

The primary development target is:

| Component | Target |
|---|---|
| GPU | NVIDIA RTX 2050 |
| VRAM | 4 GB |
| CPU | Intel i5-11400H |
| RAM | 16 GB |
| Rendering | VTube Studio / Live2D on Intel iGPU |
| AI GPU usage | Reserved primarily for LLM inference |

The RTX 2050 VRAM is the main architectural constraint.

The system must avoid designs that require loading large models simultaneously.

## Important rule

> **RTX renders nothing. It is reserved for AI computation.**

VTube Studio should preferably use the Intel iGPU so that GPU VRAM remains available for the LLM and other GPU workloads.

---

# 3. Core Design Philosophy

The project should NOT be developed by implementing the entire repository at once.

The recommended strategy is:

```text
Build
  ↓
Run
  ↓
Measure
  ↓
Test
  ↓
Understand
  ↓
Commit
  ↓
Next module
```

The system should be developed through **vertical slices**.

Each milestone must produce something runnable.

Do not start with:

- Memory
- LoRA
- Avatar
- Complex emotion
- Autonomous behavior

before the basic audio conversation pipeline works.

---

# 4. Recommended Development Strategy

The entire project should be divided into these major layers:

```text
┌──────────────────────────────────────────┐
│              APPLICATION                 │
│        main.py / orchestration            │
├──────────────────────────────────────────┤
│               RUNTIME                    │
│ scheduler / state / barge-in / VRAM      │
├──────────────────────────────────────────┤
│                BRAIN                     │
│ memory / emotion / appraisal / drive     │
│ prompt / LLM                             │
├──────────────────────────────────────────┤
│              PERCEPTION                  │
│ microphone / AEC / VAD / STT             │
├──────────────────────────────────────────┤
│              SYNTHESIS                   │
│ LLM streaming / chunker / TTS / audio    │
├──────────────────────────────────────────┤
│              EXPRESSION                  │
│ Live2D / VTube Studio                    │
└──────────────────────────────────────────┘
```

The recommended dependency direction is:

```text
main
  ↓
runtime / scheduler
  ↓
reactive / initiative / idle lanes
  ↓
event bus
  ↓
perception
  ↓
cognition
  ↓
synthesis
  ↓
expression
```

---

# 5. System Architecture

Amadeus contains three concurrent lanes.

## 5.1 Reactive Lane

This is the primary conversational path.

```text
Microphone
    ↓
AEC
    ↓
VAD
    ↓
STT
    ↓
Appraisal
    ↓
Memory Retrieval
    ↓
PAD Emotion
    ↓
Prompt Builder
    ↓
LLM
    ↓
Clause Chunker
    ↓
TTS
    ↓
Audio Output
    ↓
Avatar
```

The reactive lane must prioritize latency.

---

## 5.2 Initiative Lane

The initiative lane allows Amadeus to decide when to speak without waiting for a user question.

It should run approximately as a low-frequency control loop.

Conceptually:

```text
Silence
   +
Arousal
   +
Memory Hooks
   +
Reflection
   +
Schedule
   +
Cooldown
   ↓
Drive Evaluation
   ↓
Should Speak?
   ↓
Initiative Intent
   ↓
Same Brain / TTS Pipeline
```

The initiative system must never become random chatter.

Every proactive initiation should have a reason.

---

## 5.3 Idle Lane

The idle lane performs non-real-time cognitive work.

Examples:

- Importance scoring
- Embedding generation
- Memory storage
- Memory consolidation
- Reflection
- User portrait updates
- Memory decay
- Generating future conversation hooks

The idle lane must avoid blocking real-time interaction.

---

# 6. Full-Duplex State Machine

The main runtime states are:

```text
IDLE
  ↓
LISTENING
  ↓
THINKING
  ↓
SPEAKING
  ↓
IDLE
```

During `SPEAKING`, the microphone remains active.

Possible transitions:

```text
SPEAKING
   │
   ├── AGREE / BACKCHANNEL
   │       ↓
   │   small PAD adjustment
   │
   └── INTERRUPT
           ↓
      cancel LLM/audio
           ↓
       LISTENING
```

Important:

> The microphone must never simply be muted while Amadeus speaks.

The system needs continuous awareness for natural interruption and full-duplex interaction.

---

# 7. Latency Targets

The architecture is designed around the following target measurements:

| Metric | Target | Measured (2026-11-04) |
|---|---:|---:|
| First audible syllable p50 | ≤ 1200 ms | **1062 ms** PASS (typed input, excludes speech-to-text, see 38.4) |
| Optimized first syllable | ~900 ms | — |
| First audible syllable p95 | ≤ 1800 ms | **1275 ms** PASS (same caveat) |
| Optimized p95 | ~1300 ms | 1275 ms |
| Barge-in reaction | ≤ 150 ms | not yet |
| Backchannel | ≤ 300 ms | — |
| Sustained LLM decoding | ≥ 25 tok/s | **41.6 tok/s** PASS |
| LLM time to first token | (not set) | **82 ms** |
| Peak VRAM | ≤ 3.6 GB | **3398 MB (3.32 GB)** PASS |

These measurements should be treated as engineering requirements rather than optional optimizations.

Decode throughput has useful margin: 41.6 tok/s against conversational speech of
roughly 4 tok/s means the LLM is about 10x faster than the mouth it feeds, so the
clause chunker will never starve.

**First-syllable numbers are for typed input.** They start the clock at the end of
the transcript, so they exclude Whisper, which has not yet been timed on real speech.
Voice mode will be slower by Whisper's time, which would put p50 near the 1200 ms
limit rather than comfortably under it. They also assume the first-clause cap in
section 38 is enabled (it puts a seam mid-sentence; nobody has listened to it yet).
Without the cap p50 is about 1700 ms. See sections 31 and 38 for method and record.

---

# 8. VRAM Strategy

The RTX 2050 has approximately 4 GB VRAM, therefore model selection must be conservative.

The planned Tier A stack is approximately:

| Component | Configuration | Estimated | **Measured** |
|---|---|---:|---:|
| CUDA context | one process | ~250 MB | **790 MB** |
| LLM | Qwen2.5-3B-Instruct GGUF Q4_K_M, n_ctx 2048 | ~2.35 GB | **2210 MB** |
| STT | faster-whisper small int8_float16 | ~600 MB | **370 MB** |
| VAD | Silero VAD CPU | CPU | CPU |
| AEC | WebRTC AEC3 CPU | CPU | CPU |
| Backchannel | CPU | CPU | CPU |
| TTS | Kokoro-82M **fp32** CPU (int8 is 3x slower, see 38.1) | CPU | CPU, **+0 MB VRAM**, ~310 MB RAM |
| Embeddings | multilingual-e5-small CPU | CPU | CPU |
| Avatar | VTube Studio / iGPU | iGPU | iGPU |
| **Peak** | both GPU models resident | ~3.45 GB | **3398 MB** |
| **Headroom** | of 4096 MB | ~650 MB | **698 MB** |

Measured on a clean GPU, 2026-10-07, via `scripts/bench_vram.py`. The total came
in almost exactly on estimate (3398 MB measured vs 3450 MB predicted) but the
distribution was wrong in both directions: the CUDA context costs 3x the estimate
(790 MB, not 250 MB) while Whisper costs only 60% of it (370 MB, not 600 MB).

Two consequences worth keeping in mind:

* The 790 MB CUDA context is the single largest fixed cost after the LLM, which
  strengthens the case for keeping everything in **one** process. The original
  plan assumed an Ollama server plus a Python client; at 790 MB per context that
  second process would cost more than Whisper does. In-process
  `llama-cpp-python` avoids it.
* Headroom is 698 MB, so exactly one Tier B upgrade fits (see section 11), and
  only if nothing else touches the GPU.

## The GPU must be effectively exclusive

This is not a soft preference. A competing process does not produce a clean
failure - llama.cpp silently offloads layers to CPU and throughput collapses.
Measured on this machine, with a game holding 2815 MB of the 4096 MB:

| | Game running | Clean GPU |
|---|---:|---:|
| Time to first token | 1732 ms | **82 ms** |
| Decode | 5.3 tok/s | **41.6 tok/s** |

A 21x difference in TTFT with no error, no warning and no log line. This is why
`runtime/vram.py` exposes `require_free()` and why startup should call it: the
budget below assumes the GPU is yours alone, and that assumption has to be
checked rather than hoped for.

This leaves limited headroom.

## Critical rule

Never load models dynamically during a conversation turn.

Models should be initialized once during startup.

---

# 9. GPU Scheduling Rules

The system must respect:

```text
REAL-TIME GPU WORK
        ↓
LLM / required AI inference

IDLE GPU WORK
        ↓
Only when it cannot interfere with real-time interaction
```

Do not perform heavy reflection or background generation on the GPU while the user is actively interacting.

---

# 10. Model Strategy

## Tier A

The primary local stack should use:

```text
LLM:
Qwen2.5-3B-Instruct
GGUF Q4_K_M

STT:
faster-whisper small
int8_float16

VAD:
Silero VAD
CPU

AEC:
WebRTC AEC3
CPU

TTS:
Kokoro-82M
fp32 (NOT int8: ~3x slower on this CPU)
CPU

Embeddings:
multilingual-e5-small
CPU
```

## Tier B

More aggressive configurations may be tested later.

Examples include:

- Whisper large-v3 turbo
- RVC
- Larger context windows
- Other heavier models

However, only one aggressive component should be introduced at a time if VRAM permits.

Do not assume every large model can coexist on the RTX 2050.

---

# 11. Why the 3B LLM Is Acceptable

A 3B model will not provide the same raw reasoning quality as much larger models.

The architecture therefore compensates through:

- Strong persona constraints
- Short outputs
- Stable prompt structure
- Memory retrieval
- Emotional state
- Appraisal
- Reflection
- Initiative logic
- Narrow task specialization
- Optional Persona LoRA later

The goal is not to make the model universally intelligent.

The goal is to make the complete system feel coherent and consistent.

---

# 12. Prompt Architecture

Prompt stability is important for both quality and latency.

The prompt should be organized approximately as:

```text
STABLE PREFIX
├── System instructions
├── Persona
└── User portrait

VOLATILE TAIL
├── Relevant memories
├── Current PAD state
└── Recent dialogue
```

The stable section should remain stable for as long as possible.

Do not insert changing information before the stable prefix.

The volatile tail should remain short, with a target of approximately:

```text
≤ 300 tokens
```

This helps preserve KV-cache efficiency.

The user portrait should preferably be rewritten at session end rather than continuously during every turn.

---

# 13. Event Bus

An event-driven architecture should connect the major subsystems.

Suggested events:

```text
TurnStart
TurnEnd

LLMToken
LLMClause

TTSStarted
TTSChunk
TTSEnd

BargeIn

MoodChanged

MemoryCreated

InitiativeTriggered
```

The event bus prevents every subsystem from becoming tightly coupled to every other subsystem.

---

# 14. Core Interfaces

Important interfaces should be kept stable.

## LLM

```python
LLMProvider.stream(messages, stop)
```

The LLM should support streaming output.

---

## TTS

```python
StreamingTTS.synth(clause, style, stop)
```

TTS must support cancellation.

---

## PAD

```python
PADState.tick()
PADState.apply()
PADState.label()
```

Emotion should evolve continuously.

---

## Drive

```python
DriveSignal.should_speak()
```

This determines whether proactive behavior is justified.

---

## Backchannel

```python
Backchannel.classify()
```

This determines whether the user's speech should trigger a small conversational response.

---

# 15. Emotion Architecture

The emotional system uses PAD:

```text
P = Pleasure
A = Arousal
D = Dominance
```

The important design principle is:

> Emotion is represented as a delta rather than an absolute scripted state.

For example:

```text
User says something positive
        ↓
Appraisal
        ↓
ΔPAD
        ↓
Current PAD + ΔPAD
        ↓
New emotional state
```

This allows emotion to evolve over time instead of switching abruptly between labels.

---

# 16. Emotion Pipeline

Recommended sequence:

```text
User utterance
      ↓
Appraisal
      ↓
Emotion delta
      ↓
PAD state update
      ↓
Emotion label
      ↓
Prompt
      ↓
TTS style
      ↓
Avatar expression
```

The same emotional state should influence both language and expression.

---

# 17. Memory Architecture

Memory should not be implemented as one large database immediately.

Build it in versions.

## V0 — Recent Conversation

Store only the current conversation context.

```text
Recent dialogue
      ↓
Prompt
```

---

## V1 — Important Memory

Add persistent structured storage.

SQLite can be used for the first version.

Examples:

```text
User preferences
Important events
Relationships
Projects
Facts
Conversation memories
```

---

## V2 — Semantic Memory

Add embeddings and vector retrieval.

Potential architecture:

```text
Memory
   ↓
Embedding
   ↓
Vector Database
   ↓
Semantic Retrieval
   ↓
Prompt
```

Qdrant can be introduced here.

---

## V3 — Reflection

Reflection should transform raw memories into higher-level insights.

```text
Raw Memories
      ↓
Importance
      ↓
Reflection
      ↓
Insight
      ↓
Memory Hook
      ↓
Initiative
```

This is where the system starts becoming more than a conversation logger.

---

# 18. Memory Decay

Long-term memory should not grow forever without control.

Memory should eventually support:

- Importance
- Recency
- Retrieval frequency
- Decay
- Consolidation
- Archival

The memory system must remain useful rather than becoming an enormous collection of irrelevant information.

---

# 19. Initiative System

The initiative system should consider multiple signals.

Conceptually:

```text
Silence
+
Arousal
+
Memory Hooks
+
Recent Interaction
+
Schedule
+
Cooldown
+
Previous User Response
```

Then:

```text
Drive Score
      ↓
Should Speak?
```

The system should log why it decided to initiate.

Example:

```text
Initiation:
reason = remembered_topic
silence = 8.4 sec
arousal = 0.63
cooldown = passed
negative_response = false
```

This makes initiative tunable and debuggable.

---

# 20. Initiative Safety Rules

The system should support:

- Cooldown
- Quiet hours
- Maximum consecutive initiations
- Negative response suppression
- Reason logging

The agent should never speak merely because the drive loop runs.

> **The agent never speaks without a reason.**

---

# 21. Audio Architecture

Audio is one of the most important subsystems.

The target architecture is:

```text
Microphone
   ↓
AEC
   ↓
VAD
   ↓
STT
   ↓
LLM
   ↓
Clause Chunker
   ↓
TTS
   ↓
Audio Output
```

For low latency, later stages should begin before earlier stages have completely finished.

For example:

```text
LLM generates:

"I think"
       ↓
chunker

"I think..."
       ↓
TTS starts

while LLM continues:

"...you should probably..."
```

This creates overlapping pipeline execution.

---

# 22. Clause Chunker

The clause chunker converts streaming LLM tokens into speech-sized units.

Instead of waiting for:

```text
FULL RESPONSE
```

the system should generate:

```text
CLAUSE 1 → TTS
CLAUSE 2 → TTS
CLAUSE 3 → TTS
```

This is one of the major latency improvements.

---

# 23. Barge-In

Barge-in means the user can interrupt Amadeus while it is speaking.

Target:

```text
User starts speaking
        ↓
VAD
        ↓
Barge-in detector
        ↓
Cancel TTS
        ↓
Cancel LLM generation
        ↓
LISTENING
```

All long asynchronous operations should accept:

```python
stop: Event
```

This includes:

- LLM generation
- TTS
- Long retrieval
- Other cancellable operations

---

# 24. Acoustic Echo Cancellation

AEC is mandatory for full-duplex operation.

Without AEC:

```text
Amadeus voice
     ↓
Speaker
     ↓
Microphone
     ↓
STT
     ↓
Amadeus thinks it heard the user
```

This creates self-hearing and feedback problems.

Preferred strategy:

```text
WebRTC AEC3
+
Headphones during development
+
Residual gating
```

Do not solve the problem by simply muting the microphone.

---

# 25. Backchannel System

Backchannels are small responses such as:

```text
"mm-hm"
"yeah"
"right"
```

They should be lightweight and fast.

Target:

```text
≤ 300 ms
```

Backchannels should not require a full LLM generation cycle.

They can use CPU-friendly logic or pre-generated audio.

---

# 26. Avatar Integration

Avatar integration should happen only after the conversational pipeline is stable.

Target:

```text
PAD
 ↓
Emotion Label
 ↓
Voice Style
 ↓
Avatar Expression
```

VTube Studio / Live2D should be treated as an output layer.

It should not become part of the core cognition architecture.

---

# 27. Persona LoRA

Persona LoRA is a late-stage feature.

Do NOT start the project with LoRA training.

First prove:

```text
Base LLM
+
Prompt
+
Memory
+
Emotion
+
Initiative
```

Then determine what problems remain.

Only after the system is stable should a Persona LoRA be considered.

---

# 28. Recommended Development Roadmap

## Phase 0 — Measurement Harness

Goal:

Create the ability to measure the system before optimizing it.

Implement:

```text
trace.py
VRAM measurement
latency measurement
benchmark scripts
startup checks
```

Definition of Done:

- Application can start
- VRAM can be measured
- Latency can be recorded
- Logs are structured

---

# Phase 1 — Core Pipeline / Audio Graph

Goal:

Build the first real conversational loop.

Implement:

```text
Microphone
→ STT
→ LLM
→ TTS
→ Speaker
```

Do not add:

- Memory
- Initiative
- Avatar
- LoRA

yet.

Definition of Done:

> User speaks → Amadeus understands → generates → speaks.

---

# Phase 2 — PAD Emotion

Add:

```text
Appraisal
→ PAD
→ Emotion label
```

Connect PAD to:

- Prompt
- TTS style

Definition of Done:

Amadeus can maintain a continuously changing emotional state.

---

# Phase 3 — Memory

Implement in order:

```text
Recent context
→ SQLite
→ Embeddings
→ Qdrant
→ Retrieval
```

Then add:

```text
Reflection
→ Memory hooks
```

Definition of Done:

Amadeus can remember relevant information across interactions.

---

# Phase 4 — Initiative / Full Duplex

Implement:

```text
Drive loop
+
Cooldown
+
Memory hooks
+
Silence detection
+
Barge-in
+
Backchannel
```

Definition of Done:

Amadeus can decide when to speak, listen while speaking, and stop when interrupted.

---

# Phase 5 — Avatar

Connect:

```text
Emotion
→ Avatar expression
```

and:

```text
Speech
→ Lip sync / expression
```

Definition of Done:

The avatar reflects the conversational state.

---

# Phase 6 — Persona LoRA

Only after the complete system is stable.

Potential objective:

```text
Base model
+
Persona LoRA
```

Definition of Done:

Persona consistency improves without breaking latency or VRAM constraints.

---

# Phase 7 — Advanced Full-Duplex Research

Only after the practical local architecture is working.

Potential research direction:

```text
Moshi
```

This should be hardware-gated and experimental.

It must not block the main project.

---

# 29. Detailed Milestone Plan

The recommended practical sequence is:

```text
M0  Environment
 ↓
M1  VRAM + Benchmark
 ↓
M2  Runtime Foundation
 ↓
M3  Microphone + VAD
 ↓
M4  STT
 ↓
M5  LLM
 ↓
M6  TTS
 ↓
M7  Voice MVP
 ↓
M8  Streaming + Barge-in
 ↓
M9  PAD
 ↓
M10 Memory
 ↓
M11 Reflection
 ↓
M12 Initiative
 ↓
M13 Full Duplex
 ↓
M14 Avatar
 ↓
M15 Persona LoRA
```

---

# 30. M0 — Environment

**Status: COMPLETE (2026-10-07).** This section records what was actually built,
in order, including the problems hit. `scripts/check_environment.py` is the
executable version of this section — run it to confirm the machine is ready.

## 30.1 Toolchain survey

The first question was whether `llama-cpp-python` could be built from source.
Survey result:

| Tool | Present | Consequence |
|---|---|---|
| CUDA Toolkit (`nvcc`) | NO | cannot compile CUDA kernels |
| CMake | NO | cannot drive the build |
| MSVC / VS Build Tools | NO (only SQL Server Mgmt Studio) | no C++ compiler |
| MSVC runtime (`MSVCP140.dll`) | yes | prebuilt binaries can load |
| NVIDIA driver (`nvcuda.dll`) | yes | GPU reachable |
| ffmpeg | NO | installed below |
| winget | yes | used for ffmpeg |
| Free disk | 59.9 GB | sufficient |

Building from source would have required roughly 5–10 GB of toolchain (CUDA
Toolkit + MSVC build tools + CMake). **Decision: use the prebuilt CUDA wheel
instead.** The MSVC *runtime* is already present, which is all a prebuilt binary
needs.

## 30.2 Virtual environment and Python dependencies

`venv/` already existed with Python 3.12.10, `torch 2.5.1+cu121` and `numpy`.
Everything else was missing.

```bash
venv/Scripts/python.exe -m pip install -e ".[stt,tts,dev]"
```

**Problem 1 — setuptools refused to build.** Flat-layout auto-discovery saw
`amadeus/`, `config/`, `scripts/`, `data/` and `venv/` as competing top-level
packages:

```text
error: Multiple top-level packages discovered in a flat-layout
```

Fixed by declaring packages explicitly in `pyproject.toml`:

```toml
[tool.setuptools.packages.find]
include = ["amadeus*"]
```

The install then succeeded, and critically left `torch 2.5.1+cu121` untouched — an
already-satisfied unpinned requirement is not re-resolved, so the hand-installed
CUDA build survived.

## 30.3 llama-cpp-python with CUDA

```bash
venv/Scripts/python.exe -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

**Problem 2 — pip downloaded the wheel at ~190 kB/s and produced a corrupt file:**

```text
ERROR: Wheel llama-cpp-python located at ...whl is invalid.
```

A throughput test against the same URL showed 4.4 MB/s with HTTP 206 range
support, so the bottleneck was pip, not the network. Fixed by fetching the wheel
directly with resume support, then installing from the local file:

```bash
curl -L -C - --retry 5 -o llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu121/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl
venv/Scripts/python.exe -m pip install ./llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl
```

448,299,770 bytes, verified as a valid zip with 73 members before installing.

**Problem 3 — the wheel installed but would not import:**

```text
RuntimeError: Failed to load shared library ...llama_cpp\lib\llama.dll:
Could not find module ... (or one of its dependencies).
```

The error names `llama.dll`, which is misleading. Reading the PE import tables
gave the real chain:

```text
llama.dll  ->  ggml.dll  ->  ggml-cuda.dll  ->  cudart64_12.dll   MISSING
                                            ->  cublas64_12.dll   MISSING
                                            ->  nvcuda.dll        present (driver)
```

The wheel links against the CUDA runtime but does not bundle it, and there is no
CUDA Toolkit on this machine. However `torch 2.5.1+cu121` ships exactly those
libraries in `venv/Lib/site-packages/torch/lib/`:

| DLL | Size |
|---|---:|
| `cublasLt64_12.dll` | 513.9 MB |
| `cublas64_12.dll` | 93.5 MB |
| `cudart64_12.dll` | 0.5 MB |
| total | **608 MB** |

Copying 608 MB into `llama_cpp/lib/` was rejected as wasteful. Instead
`amadeus/runtime/cuda_dlls.py` adds the torch lib directory to the process DLL
search path via `os.add_dll_directory()`, and `cognition/brain.py` calls it
**before** importing `llama_cpp`. The shim also falls back to the `nvidia-*-cu12`
pip wheels and to `CUDA_PATH` if either ever exists.

Result:

```text
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 2050, compute capability 8.6, VMM: yes
llama_cpp          : 0.3.4
GPU offload support: True
```

**This import order is load-bearing.** Importing `llama_cpp` before the shim
poisons the process — the failed library load is not retried.

## 30.4 ffmpeg

`pydub` shells out to the `ffmpeg` binary to decode Edge-TTS mp3 output.

```powershell
winget install --id Gyan.FFmpeg --scope user
```

Installed ffmpeg 9.0.2 and added its directory to the **user** PATH. Already
running shells do not inherit that, so a terminal restart is required;
`check_environment.py` detects this case and reports WARN rather than FAIL.

Note that `av` (PyAV) is already installed as a `faster-whisper` dependency and
bundles the ffmpeg libraries, so `synthesis/edge.py` could drop both `pydub` and
the external binary. Worth doing when `pydub` becomes a problem — it needs the
stdlib `audioop` module, removed in Python 3.13.

## 30.5 Model weights

```bash
curl -L -C - -o data/models/qwen2.5-3b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
```

2,104,932,768 bytes at ~6.5 MB/s, magic bytes verified as `GGUF`. The filename in
the official Qwen repo matches `config/settings.yaml` exactly, so no rename was
needed.

## 30.6 Final state

```text
Python                3.12.10
torch                 2.5.1+cu121   (CUDA 12.1, sm_86)
llama-cpp-python      0.3.4 cu121   GPU offload: True
faster-whisper        1.2.1         (pulls ctranslate2 4.8.2, av, onnxruntime)
edge-tts              7.2.8
pydub                 0.25.1        + ffmpeg 9.0.2
sounddevice           0.5.6
pyyaml / python-dotenv / pytest / pytest-asyncio
Qwen2.5-3B Q4_K_M     2007 MB, valid GGUF
```

Verify with:

```bash
venv/Scripts/python.exe scripts/check_environment.py
```

All checks pass. Audio devices resolve to Realtek mic in / headphones out —
**use headphones**, `perception/aec.py` is still a pass-through stub.

---

# 31. M1 — Measurement

**Status: VRAM and LLM throughput COMPLETE. End-to-end latency BLOCKED on `main.py`.**

The goal of this milestone is to answer three questions with numbers rather than
estimates:

```text
How much VRAM is available?
How much VRAM does each model use?
How long does each stage take?
```

## 31.1 The bug that made measurement necessary

`runtime/vram.py` originally measured with `torch.cuda.memory_allocated()`. The
first benchmark exposed the problem immediately:

| Stage | Driver reports | `memory_allocated()` reports |
|---|---:|---:|
| baseline | 790 MB | 0.0 MB |
| + Qwen2.5-3B loaded | 3973 MB | **0.0 MB** |

`memory_allocated()` only sees the torch caching allocator. The two largest
consumers in this project do not use it — `llama-cpp-python` allocates through
ggml-cuda and `faster-whisper` through CTranslate2, both straight to the CUDA
driver. **The VRAM guard was blind to the entire budget it existed to protect**,
and would have reported a comfortable margin right up to an OOM crash.

`vram.py` was rewritten to use `torch.cuda.mem_get_info()`, which queries the
driver and therefore sees every allocation in the process regardless of library.
`torch_allocated_mb` is retained as a diagnostic field only, never for budget
decisions.

## 31.2 The first run was void

The first clean-looking benchmark produced TTFT 1732 ms and 5.3 tok/s, roughly 6x
worse than predicted. Cause:

```text
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
15864, MonsterHunterWorld.exe, 2815 MiB
```

A game held 2815 MB of the 4096 MB, leaving llama.cpp ~1.1 GB, so it silently
offloaded most layers to CPU. **No error, no warning.** Those numbers were
discarded, not recorded.

This produced two permanent changes: `require_free()` in `runtime/vram.py`, which
refuses at startup rather than degrading, and a free-VRAM check plus
GPU-process listing in `check_environment.py`.

## 31.3 Measured results, clean GPU

`scripts/bench_vram.py`, 2026-10-07, GPU confirmed at 0 MiB used / 3962 MiB free
before starting:

```text
GPU            : NVIDIA GeForce RTX 2050
total VRAM     : 4096 MB
baseline in use: 790 MB (CUDA context)

faster-whisper small        +370 MB   load 1.81s   free now 2936 MB
Qwen2.5-3B Q4_K_M          +2210 MB   load 1.25s   free now  726 MB

generation:
  TTFT            : 82 ms
  decode          : 83 tok in 1.99s = 41.6 tok/s

peak VRAM        : 3398 MB / 4096 MB (83.0%)
headroom         : 698 MB
budget           : WITHIN target of 3686 MB
```

Estimate versus reality:

| | Estimated | Measured | Verdict |
|---|---:|---:|---|
| CUDA context | 250 MB | **790 MB** | 3.2x worse |
| LLM (n_ctx 2048) | 2350 MB | **2210 MB** | slightly better |
| Whisper small int8 | 600 MB | **370 MB** | 1.6x better |
| Peak total | 3450 MB | **3398 MB** | accurate |
| Decode rate | 30–40 tok/s | **41.6 tok/s** | better |

The aggregate estimate was good; the breakdown was wrong in both directions. The
actionable finding is the CUDA context at 790 MB — see section 8 for why this
argues for a single-process design.

Raw results are written to `data/logs/bench_vram.json` on every run.

## 31.4 Still to measure

```text
scripts/bench_latency.py     - DONE 2026-11-04 (typed input; results in section 38)
scripts/bench_aec.py         - needs real AEC; echo return loss
```

Whisper transcription latency was not isolated in this pass — the benchmark only
forces a dummy decode so CTranslate2 allocates. `bench_latency.py` drives typed
input, so it also does not time Whisper: real STT timing still needs real
utterances, and the first-syllable figures in section 38 exclude it.

## 31.5 An unprompted quality datapoint

Asked to explain GGUF, the 3B model replied:

> GGUF stands for "Google Generalized Quantized Format," a quantization format
> developed by Google

Both claims are wrong — GGUF comes from the ggml / llama.cpp project and has no
connection to Google. This is a clean, unsolicited demonstration of the
hallucination risk in section 50.5, and empirical support for the
quality-recovery strategy in section 11: narrow the job to one persona, cap
output length, and move factual reasoning off the hot path. A 3B model is
adequate for *being a character* and unreliable as a *reference source*, and the
persona design should lean on the former rather than inviting the latter.

Do not optimize blindly.

Measure first.

---

# 32. M2 — Runtime Foundation

Build:

```text
config.py
bus.py
trace.py
state.py
scheduler.py
```

The runtime foundation should support:

- Configuration
- Event publishing
- State transitions
- Logging
- Scheduling
- Cancellation

At this point there is still no need for sophisticated AI.

---

# 33. M3 — Microphone and VAD

Implement:

```text
audio_in.py
aec.py
vad.py
```

Test:

```text
Microphone
 ↓
AEC
 ↓
VAD
```

The system must reliably determine:

```text
speech
vs
silence
```

before adding STT.

---

# 34. M4 — STT

Implement:

```text
stt.py
```

Target:

```text
Microphone
→ VAD
→ Whisper
→ text
```

Measure:

- Recognition latency
- Recognition accuracy
- CPU/GPU usage
- VRAM usage

---

# 35. M5 — LLM

Implement:

```text
cognition/brain.py
```

and a provider abstraction:

```python
class LLMProvider:
    async def stream(self, messages, stop):
        ...
```

The system should support streaming tokens.

Do not tightly couple the rest of the application to a specific model implementation.

---

# 36. M6 — TTS

Implement:

```text
synthesis/base.py
synthesis/kokoro.py
synthesis/chunker.py
synthesis/audio_out.py
```

Target:

```text
Text
→ Clause
→ TTS
→ Audio
```

TTS must be cancellable.

---

# 37. M7 — First Real MVP

The first major milestone is:

```text
MIC
 ↓
VAD
 ↓
STT
 ↓
LLM
 ↓
TTS
 ↓
SPEAKER
```

This is the first version that should be considered a functioning Amadeus prototype.

Do not add memory or avatar before this works reliably.

---

# 38. M8 — Streaming Pipeline

Upgrade:

```text
LLM
 ↓
streaming tokens
 ↓
clause chunker
 ↓
streaming TTS
 ↓
audio
```

The goal is to reduce time-to-first-audio.

Instead of:

```text
Wait for complete response
→ TTS
→ Speak
```

use:

```text
First clause
→ TTS
→ Speak

while:
LLM continues
```

---

## 38.0 Status: built and measured (2026-11-04). Not verified by ear.

What the roadmap diagram above turned into, in order of what mattered:

1. **Pipelined `_respond`** - synthesis of clause N+1 overlaps playback of clause N.
2. **First-clause policy** in the chunker - `first_clause_min_chars` / `first_clause_max_chars`.
3. **A persistent output stream** - the legacy player added ~200 ms of silence per clip.

All three were found and sized by measurement, not by design; the details, the sweep and the
mistakes along the way are in `fix.md` Part 4. This section is the summary.

## 38.1 Local voice: Kokoro, fp32

Cloud TTS was 1909 ms of a 2605 ms first-sound latency. The planned local replacement
(Kokoro int8) was **slower** than the cloud, 5.2 s to speak a 4 s sentence, because 83% of
its runtime is one slow operator (`ConvInteger`). The fp32 model speaks the same sentence in
~1.5 s (real-time factor 0.36, 6 threads), uses 0 MB of VRAM, and works offline. Cost model for
this CPU: **~440 ms fixed + ~19 ms per character**, against speech at ~15 characters per second.

## 38.2 What the numbers say

Baseline (sequential playback, original chunker): first sound p50 1840 / p95 2704 ms, and the
listener heard a pause of a second or more inside 14 of 14 multi-clause replies (worst 3.1 s).

| Config (16 turns, identical seeded replies) | p50 | p95 | worst silence | turns with a >300 ms pause |
|---|---:|---:|---:|---:|
| sequential baseline | 1840 | 2704 | 3110 | 14 of 14 |
| pipelined, original rules | 1896 | 2659 | 795 | 2 of 14 |
| pipelined, `min=0` (fastest, no cap) | 724 | 1966 | 1380 | 7 of 16 |
| **pipelined, min 16 / max 30 + backtracking (default)** | **1043** | **1246** | 1010 | 2 of 16 |

Pipelining removes the stutter but cannot speed the first clause. The first clause's
synthesis time sets first sound, and many replies open with a 60 to 110 character sentence
with no comma, so no minimum length finds a break. The window that satisfies both limits is
an opener of about **19 to 35 characters**: long enough to still be playing when clause 2 is
ready, short enough to synthesise in time. A cap that ends the first clause at a word boundary
reaches it.

## 38.3 The cap and its seams (a quality trade you must judge by ear)

If the first clause reaches 30 characters with no punctuation, it is cut at the next word
boundary. Each clause is synthesised as its own utterance, so the voice falls and resets at the
join. To keep seams on phrase boundaries, a cut that would land after a function word
(`that`, `and`, `the`, `of`...) moves back before it: `a beloved stimulant | that energizes`
rather than `a beloved stimulant that | energizes`. It cannot fix adjective-noun splits
(`personal | preferences`). Turn it off with `first_clause_max_chars: null`; p50 then returns
to about 1700 ms.

## 38.4 Honest limits

* **Typed input only.** Whisper time is not in the 1062 ms.
* **Not listened to.** The seams and the new output path are unjudged.
* **Long run-on clauses still pause ~1 s** (a 150 character unpunctuated clause 3). The cap only
  governs the first clause.
* Eight prompts, same replies repeated: small differences between configs are noise.

## 38.5 The output path

`sd.play()` per clip costs ~202 ms of dead air each on this machine (Windows MME host API).
`StreamingAudioPlayer` holds one WASAPI shared-mode stream open (25 ms latency, 24 kHz needs
`auto_convert`) and queues clips back to back: measured 202 ms -> 0 ms per clip on the real
device. `audio.streaming_output: false` restores the legacy player.

## 38.6 Tooling added

`scripts/bench_latency.py` (`--first-min`, `--first-max`, `--seed`), `--mute` (silent run that
waits out each clip's real duration, so timing stays valid), `NullPlayer`,
`TurnTrace.add_gap`, and `tests/` (59 passing).

# 39. M9 — Emotion

## 39.0 Status: built and tested (2026-11-10). Not judged by ear, and the prompt effect is small.

```text
user text --appraise--> ΔPAD --apply--> PADState (decays to a baseline) --> label
                                              |--> prompt hint   (one system line, only when she feels something)
                                              |--> voice speed   (Kokoro can do nothing else)
                                              `--> "EmotionChanged" on the bus   (the avatar, M14)
```

| File | Role |
|---|---|
| `emotion/pad.py` | `PAD` (a point or a delta) and `PADState`: `tick()` decays toward the baseline, `apply()` adds a delta, `label()` reads it out |
| `emotion/appraisal.py` | rule-based `appraise(text) -> ΔPAD + the tags that explain it` |
| `emotion/labels.py` | nearest-prototype labels (`flustered`, `annoyed`, `angry`, `happy`, `curious`, `concerned`, `sad`...) and the one-line prompt hints |
| `emotion/engine.py` | `EmotionEngine`: owns the state, builds the hint and the voice style, publishes `EmotionChanged`, fades the mood while nobody talks |
| `config/emotion.yaml` | baseline, half-lives, strength, voice-speed mapping (every number is a hand-picked starting point) |

## 39.1 Design decisions

* **Rules, not the LLM.** Appraisal is on the critical path (utterance -> appraisal -> prompt). A second LLM call
  would add a generation's worth of latency on a full GPU. Rules cost microseconds, and every decision comes
  back as a tag (`why: praise`), so "why is she annoyed?" has an answer. Unrecognised text changes nothing.
* **The character's wiring is in the deltas.** A compliment raises arousal and *lowers* dominance (flustered), an
  insult raises dominance (she pushes back). That is the tsundere signature, written as numbers.
* **Neutral is exactly the old behaviour.** Within `NEUTRAL_RADIUS` of the baseline the prompt gets no hint and the
  voice speed is exactly 1.0. The persona you tuned is untouched until she feels something.
* **The hint is in the volatile tail**, one system message just before the user's turn, never in history. The
  system prompt stays byte-identical turn to turn (stable prefix), and a stale "you are angry" cannot survive.
* **Voice: speed only.** Kokoro takes a speed and nothing else. `speed = 1 + 0.10 x dArousal + 0.04 x dPleasure`,
  clamped 0.90-1.12, from the deviation from baseline.

## 39.2 What the live run taught (all fixed, all pinned by tests)

Unit tests passed first time. Replaying the conversation with the real model found three things they could not:

1. **Arousal pinned at +1.0** after two arousing events, so every later mood looked "curious" (`Hey Christina` came out
   curious, not annoyed). Fix: diminishing returns near the bounds (a delta is shrunk by the room left in its direction,
   never enlarged).
2. **A fading mood wandered through unrelated labels** (flustered -> curious -> concerned) because the straight path
   back to the baseline passes near other prototypes. Fix: between events the label's name is frozen and only its
   intensity fades.
3. **The label now follows the event, not the blended state.** A new event renames the mood only if it is at least 60%
   as strong as the mood in force; a weaker one adds to it without renaming it.

Also found by running the phrases: negation was broken ("You're not stupid", "I'm not sad" matched nothing, because the
negator sits inside the phrase). Fixed, and a `can't believe you're so smart` false-negation removed.

Mutation checks were run on the M9 logic (14 deliberate breakages: each one is caught by a named test).

## 39.3 Honest limits

* **The prompt hint helps a little on small talk and not at all on factual answers.** Controlled A/B (same seeds, same
  prompt, with vs without the hint, 7 moods x 3 held-out lines x 3 seeds): 0 of 63 replies leaked the mood or the
  instructions, short social replies took on some colour ("Ugh, thanks. I don't drink caffeine." when angry), but a
  question about neurons opened "Sure!" in every mood. There was no null-control run, so no effect size is claimed.
  The same 3B-model ceiling as the persona (README section 45): the real fix is a persona LoRA (M15).
* **Replies still copy the persona's example lines** word for word on matching prompts, which also hides the hint.
* **Coverage.** Rules fire on English keywords, so a feeling expressed in other words is missed (it fails safe: nothing
  changes). Add a rule in `appraisal.py` plus a sentence in `tests/test_appraisal.py` (a test fails if any rule has none).
* **Hand-tuned numbers.** Prototypes, deltas and half-lives are judgement, not data. Tune `config/emotion.yaml` while
  watching the `[mood: ...]` line printed after each turn.
* **Not persisted.** Mood resets each run (memory is M10). It appraises what the user says, not what she says.
* **Not heard.** The speed change is deterministic and tested, but nobody has judged how it sounds.

---

# 40. M10 — Memory

Recommended first database:

```text
SQLite
```

Use it for structured memory.

Later:

```text
SQLite
+
Qdrant
```

Use Qdrant only when semantic retrieval is needed.

---

# 41. M11 — Reflection

Reflection should not execute on the critical conversation path.

Use the idle lane:

```text
Conversation ends
       ↓
Importance scoring
       ↓
Reflection
       ↓
Insight
       ↓
Memory hook
```

The hook can later feed the initiative system.

---

# 42. M12 — Initiative

Implement:

```text
cognition/drive.py
```

The drive loop can run at approximately:

```text
2 Hz
```

It evaluates whether there is a reason to initiate.

Example conceptual inputs:

```text
time since user spoke
arousal
memory hooks
scheduled events
cooldown
previous response
```

---

# 43. M13 — Full Duplex

Combine:

```text
Listening
+
Speaking
```

with:

```text
Barge-in
Backchannel
AEC
Cancellation
```

Test scenarios:

### Scenario 1

User speaks normally.

Expected:

```text
LISTENING
→ THINKING
→ SPEAKING
→ IDLE
```

### Scenario 2

User interrupts.

Expected:

```text
SPEAKING
→ INTERRUPT
→ cancel TTS
→ cancel LLM
→ LISTENING
```

### Scenario 3

User provides a short acknowledgement.

Expected:

```text
User speech
→ backchannel
```

without requiring a full response.

---

# 44. M14 — Avatar

## 44.0 Status: built and tested (2026-11-10). Browser preview runs and was watched; VTube Studio is mock-tested only.

```text
bus "EmotionChanged" --\
bus "StateChanged"  ----+--> ExpressionModel.step(dt) --> ExpressionFrame --> sinks
player.level -----------/        (30 Hz, smoothed)         (12 face values)    |- browser preview (SVG, no model needed)
 (loudness of the audio                                                        `- VTube Studio (Live2D, needs a model)
  actually being played)
```

The avatar is an **output consumer** (README section 26): it listens to the bus and reads the speaker level, and never
writes to cognition, memory or the state machine. A test runs the same reply with and without an avatar attached and
requires identical speech; another shows a broken avatar (port taken, renderer crash) leaves the voice running.

| File | Role |
|---|---|
| `expression/frame.py` | `ExpressionFrame` and `ExpressionModel`: emotion + system state + speech level -> mouth, smile, eyes, brows, blush, head, gaze; smoothing, blinking, idle sway. Pure maths, no renderer |
| `expression/driver.py` | `AvatarDriver`: fixed-rate loop, bus subscriptions, fans frames out to sinks, isolates a failing sink |
| `expression/server.py` + `web/avatar.html` | local browser preview (aiohttp, 127.0.0.1 only) and a self-contained SVG face |
| `expression/vtube.py` | `VTubeStudioSink`: auth, parameter discovery, injection, hotkeys, automatic reconnect |
| `synthesis/audio_out.py` | each player gained `level` (RMS of the block just sent to the device) for lip-sync |
| `scripts/avatar_demo.py` | shows the whole chain with no LLM, GPU or microphone |

## 44.1 Run it

```text
venv/Scripts/python.exe -m amadeus.main --avatar            # real conversation; open http://127.0.0.1:8765/
venv/Scripts/python.exe scripts/avatar_demo.py              # scripted tour, silent, no GPU needed
venv/Scripts/python.exe scripts/avatar_demo.py --hold "Hey Christina." --speak     # hold one pose
```

Add `?debug` to the URL to see the raw numbers. `config/settings.yaml` `avatar:` sets the port, the frame rate and VTube Studio.

## 44.2 What was measured

| | Result |
|---|---|
| Frame rate to a browser, 9 s of live frames | **30.1 fps**, p99 inter-frame gap 48 ms |
| Cost of one tick (model + browser sink, 180 ticks) | **p50 0.25 ms, p99 0.46 ms, max 0.68 ms** of a 33 ms frame |
| Lip-sync, live, with real Kokoro audio | mouth swept 0 to 0.998 with open/close transitions at word gaps |
| Voice latency with emotion + avatar on (3 real turns) | first sound **p50 1102 ms / p95 1668 ms**, still inside the 1200 / 1800 targets |
| Real app, recorded live from the browser (3 turns, real LLM) | per turn IDLE -> THINKING -> SPEAKING -> IDLE with the right label each time; mouth open in 273 frames (peak 1.0); blush peaked 0.64 |
| Poses watched in the browser | flustered, annoyed, concerned, happy, angry: distinct and legible |

Looking at the rendered avatar and then recording the real app found three bugs the unit tests had passed:

1. A dark band between hair and shoulders (layering).
2. "concerned" drawn with furrowed (angry) brows, because her high resting dominance made the maths read concern as anger.
3. **The face disagreed with the label.** 14 s after a compliment, "Hey Christina" gave the label *annoyed 0.78* but a face
   with blush 0.57, a smile and barely furrowed brows: the face followed the accumulated PAD (still holding the compliment)
   while the label followed the event. Each label now has an expression preset (`frame.py _PRESETS`) blended in by its
   intensity (up to 75%), with PAD supplying the continuous shading. A test replays that exact sequence through the real
   engine and model; turning the presets off makes it fail.

## 44.3 VTube Studio: what is and is not known

The message flow follows the official README (port 8001, token auth approved by the user in-app,
`InputParameterListRequest`, `InjectParameterDataRequest`, custom parameters, hotkeys). The sink asks VTube Studio which
parameters exist and **only drives those**, and scales each value into that parameter's own reported min/max/default,
because the README does not list the default parameter names and I did not want to hard-code guesses.

Tested against a **mock** server that speaks the documented protocol (14 tests: first-run pairing, reuse of the saved
token, a revoked token, access denied, missing parameters, scaling, hotkeys, VTube Studio not running yet, restart and
reconnect). **Never run against a real VTube Studio**, because there is no Live2D model. Expect to need
`vtube_invert` for a parameter that moves the wrong way, and `vtube_hotkeys` to map moods to the model's expressions.
Two custom parameters (`AmadeusBlush`, `AmadeusBrowInner`) appear in VTube Studio for you to map.

## 44.4 Honest limits

* **The browser face is a flat SVG placeholder, not the 3D/Live2D avatar.** It exists so the chain can be seen and tested
  now. A real model is a separate, large piece of work.
* **Mouth is amplitude only.** It opens with loudness; there are no phoneme shapes (visemes).
* **Head turns are faked in the browser** as a small translation plus roll; a flat drawing cannot turn.
* **Lip-sync needs the streaming player** (`audio.streaming_output: true`); the legacy player reports level 0.
* **Numbers are judgement.** Smoothing times, brow angles and the blush formula are hand-tuned by eye on one drawing.
* **Not judged by a person in motion.** I could check frames, rates and stills, not whether the animation feels alive.

---

# 45. M15 — Persona LoRA

Only after collecting enough real interaction data.

Possible training material:

```text
Conversation examples
Persona examples
Style examples
Emotional responses
Interaction patterns
```

LoRA should solve a demonstrated problem.

Do not train it simply because LoRA exists.

---

# 46. Proposed Repository Structure

```text
amadeus/
│
├── README.md
├── implementation.md
├── pyproject.toml
├── .env.example
│
├── config/
│   ├── settings.yaml
│   ├── persona.yaml
│   ├── emotion.yaml
│   └── drive.yaml
│
├── amadeus/
│   ├── main.py
│   ├── config.py
│   ├── bus.py
│   ├── trace.py
│   ├── state.py
│   │
│   ├── perception/
│   │   ├── audio_in.py
│   │   ├── aec.py
│   │   ├── vad.py
│   │   ├── stt.py
│   │   └── backchannel.py
│   │
│   ├── cognition/
│   │   ├── drive.py
│   │   ├── appraisal.py
│   │   ├── prompt.py
│   │   └── brain.py
│   │
│   ├── emotion/
│   │   ├── pad.py
│   │   └── labels.py
│   │
│   ├── memory/
│   │   ├── store.py
│   │   ├── core.py
│   │   ├── working.py
│   │   ├── archival.py
│   │   ├── scoring.py
│   │   ├── hooks.py
│   │   ├── reflection.py
│   │   └── decay.py
│   │
│   ├── synthesis/
│   │   ├── base.py
│   │   ├── kokoro.py
│   │   ├── edge.py
│   │   ├── rvc.py
│   │   ├── chunker.py
│   │   └── audio_out.py
│   │
│   ├── expression/
│   │   ├── server.py
│   │   └── vtube.py
│   │
│   └── runtime/
│       ├── vram.py
│       ├── scheduler.py
│       └── barge_in.py
│
├── scripts/
│   ├── check_environment.py
│   ├── benchmark_latency.py
│   ├── benchmark_vram.py
│   ├── setup_models.py
│   ├── test_aec.py
│   └── train_lora.py
│
├── tests/
│   ├── test_pad.py
│   ├── test_drive.py
│   ├── test_scoring.py
│   ├── test_chunker.py
│   ├── test_backchannel.py
│   └── test_prompt_stability.py
│
└── data/
    ├── sqlite/
    ├── qdrant/
    └── models/
```

---

# 47. Responsibilities of Major Files

## `main.py`

Application entry point.

Responsible for:

- Startup
- Dependency initialization
- Starting runtime
- Connecting components

---

## `bus.py`

Event communication.

Responsible for:

- Publish
- Subscribe
- Event routing

---

## `trace.py`

Observability.

Every important stage should produce trace information.

Example:

```text
VAD start
STT start
STT end
LLM start
First token
First clause
TTS start
First audio
TTS end
```

---

## `state.py`

Central interaction state.

Example:

```text
IDLE
LISTENING
THINKING
SPEAKING
```

---

## `runtime/vram.py`

GPU memory monitoring.

It should help answer:

```text
Current VRAM
Peak VRAM
Model VRAM
Available headroom
```

---

# 48. Testing Strategy

Every major component should have tests.

## PAD

Test:

```text
initial state
delta application
decay
label mapping
```

## Drive

Test:

```text
cooldown
silence
negative response
memory hook
should speak
```

## Memory

Test:

```text
importance
retrieval
decay
consolidation
```

## Chunker

Test:

```text
short sentence
long sentence
punctuation
streaming tokens
```

## Backchannel

Test:

```text
agree
uncertain
interrupt
normal speech
```

## Prompt

Test:

```text
stable prefix
volatile tail
tail length
memory insertion
PAD insertion
```

---

# 49. Observability

Observability is a first-class feature.

For every conversation turn, record enough information to understand:

```text
VAD latency
STT latency
STT result
LLM first token
LLM token rate
First clause
TTS latency
First audio
Total response latency
VRAM
PAD state
Memory retrieval
State transitions
Initiation reason
```

Without this information, optimizing the system becomes guesswork.

---

# 50. Performance Bottlenecks to Watch

The major bottlenecks are expected to be:

## 1. Prompt Prefill

Long prompts can increase time-to-first-token.

Solution:

```text
Stable prefix
+
Short volatile tail
```

---

## 2. VRAM

The 4 GB GPU limit is strict.

Solution:

- Quantization
- CPU offloading where appropriate
- Avoid simultaneous large models
- Keep headroom

---

## 3. Pipeline Latency

Sequential processing creates unnecessary delay.

Bad:

```text
STT
wait
LLM complete
wait
TTS complete
wait
audio
```

Better:

```text
STT
 ↓
LLM streaming
 ↓
Clause 1
 ↓
TTS
 ↓
Audio

while LLM continues
```

---

## 4. Emotional Discontinuity

If emotion changes randomly, the character will feel artificial.

Solution:

```text
PAD
+
Delta updates
+
Decay
```

---

## 5. Persona Drift

A small model may deviate from its intended personality.

Solution:

- Stable system prompt
- Persona file
- User portrait
- Memory
- Short response style
- Later Persona LoRA

---

## 6. Acoustic Self-Hearing

The system may recognize its own voice.

Solution:

```text
AEC
+
residual gating
+
headphones during development
```

---

## 7. Initiative Quality

The agent may speak too often or at inappropriate times.

Solution:

```text
Drive
+
Cooldown
+
Memory hooks
+
Negative response suppression
+
Logging
```

---

# 51. Important Engineering Rules

These rules should be treated as project-wide constraints.

### Rule 1

Never load a model on demand during a conversation turn.

### Rule 2

Never put volatile prompt content before stable prompt content.

### Rule 3

Never perform unnecessary GPU work in the idle lane.

### Rule 4

Every major stage boundary should be traced.

### Rule 5

Emotion is a delta, not an arbitrary absolute label.

### Rule 6

Long asynchronous operations must support cancellation.

### Rule 7

Never mute the microphone simply because Amadeus is speaking.

### Rule 8

The agent never speaks without a reason.

### Rule 9

Every proactive initiation should be logged.

### Rule 10

RTX should not be used for avatar rendering.

---

# 52. What NOT to Build First

Avoid starting with:

```text
❌ Persona LoRA
❌ Complex memory graph
❌ Qdrant optimization
❌ Full Live2D integration
❌ Autonomous initiative
❌ Advanced reflection
❌ RVC voice conversion
❌ Large models
❌ Multi-agent architecture
❌ Distributed services
```

These features are valuable later.

They are not the foundation.

---

# 53. The Correct Starting Point

The first working system should be extremely small:

```text
Microphone
   ↓
VAD
   ↓
STT
   ↓
Qwen 3B
   ↓
Kokoro
   ↓
Speaker
```

Then upgrade it:

```text
Microphone
   ↓
AEC
   ↓
VAD
   ↓
STT
   ↓
Appraisal
   ↓
Memory
   ↓
PAD
   ↓
Prompt
   ↓
LLM Streaming
   ↓
Clause Chunker
   ↓
Streaming TTS
   ↓
Audio
   ↓
Avatar
```

Finally:

```text
Reactive Lane
        ↕
Initiative Lane
        ↕
Idle Lane
        ↕
Memory
        ↕
Emotion
        ↕
Avatar
```

---

# 54. First Practical Implementation Order

If starting the project today, follow this exact order:

```text
STEP 1
Create Python environment

STEP 2
Verify RTX / CUDA / PyTorch

STEP 3
Create repository skeleton

STEP 4
Implement configuration

STEP 5
Implement structured logging / trace

STEP 6
Implement VRAM monitor

STEP 7
Test microphone

STEP 8
Test VAD

STEP 9
Test STT independently

STEP 10
Test Qwen independently

STEP 11
Test TTS independently

STEP 12
Connect STT → LLM → TTS

STEP 13
Add audio streaming

STEP 14
Add cancellation

STEP 15
Add barge-in

STEP 16
Add PAD

STEP 17
Add memory

STEP 18
Add reflection

STEP 19
Add initiative

STEP 20
Add full-duplex behavior

STEP 21
Add avatar

STEP 22
Only then consider Persona LoRA
```

---

# 55. Definition of Done

A phase is not finished merely because its Python files exist.

Each phase must satisfy:

```text
Code exists
+
Component runs
+
Component is tested
+
Latency is measured
+
VRAM is measured
+
Failure cases are handled
+
Logs are available
+
Integration works
```

For example, Phase 1 is complete only when:

```text
User speaks
     ↓
Speech recognized
     ↓
LLM generates
     ↓
TTS speaks
```

reliably on the target laptop.

---

# 56. Recommended Documentation

In addition to this README, maintain:

```text
docs/
├── architecture.md
├── development_plan.md
├── interfaces.md
└── testing_strategy.md
```

The README should explain the project.

`implementation.md` should contain the detailed technical specification.

`architecture.md` should explain how components communicate.

`development_plan.md` should track implementation progress.

`interfaces.md` should define stable module contracts.

`testing_strategy.md` should describe how the system is validated.

---

# 57. Suggested Git Workflow

Use small commits.

Example:

```text
init project structure

add environment checker

add vram monitor

add trace system

add microphone input

add vad

add whisper stt

add qwen provider

add kokoro tts

connect voice pipeline

add streaming

add barge in

add pad emotion

add sqlite memory

add semantic memory

add reflection

add initiative

add avatar
```

Avoid one enormous commit such as:

```text
"finish amadeus"
```

---

# 58. Development Mindset

This project is large enough that trying to understand the entire system before writing any code will become overwhelming.

Instead, understand the system layer by layer.

At every stage ask:

```text
1. What problem does this component solve?

2. What does it receive?

3. What does it output?

4. How fast must it be?

5. Does it use CPU or GPU?

6. How can it be cancelled?

7. How can I test it independently?

8. How does it connect to the next component?
```

This keeps the architecture understandable.

---

# 59. Final Architecture

The intended final system can be visualized as:

```text
                         ┌───────────────────┐
                         │      USER         │
                         └─────────┬─────────┘
                                   │
                              Microphone
                                   │
                                   ▼
                         ┌───────────────────┐
                         │  AUDIO / AEC /    │
                         │       VAD         │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │       STT         │
                         └─────────┬─────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────┐
                  │            APPRAISAL            │
                  └───────────────┬────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              ┌───────────┐               ┌───────────┐
              │   MEMORY  │               │    PAD    │
              └─────┬─────┘               └─────┬─────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                         ┌───────────────────┐
                         │  PROMPT BUILDER   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │    QWEN 3B LLM   │
                         └─────────┬─────────┘
                                   │
                             Streaming
                                   │
                                   ▼
                         ┌───────────────────┐
                         │  CLAUSE CHUNKER   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │       TTS         │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   AUDIO OUTPUT    │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   LIVE2D / VTUBE  │
                         └───────────────────┘


          ┌─────────────────────────────────────┐
          │          INITIATIVE LANE            │
          │                                     │
          │ silence + arousal + memory +       │
          │ schedule + cooldown + response     │
          │                ↓                    │
          │          DRIVE DECISION             │
          └────────────────┬────────────────────┘
                           │
                           ▼
                    Same Brain / TTS


          ┌─────────────────────────────────────┐
          │             IDLE LANE               │
          │                                     │
          │ scoring → embedding → memory        │
          │ reflection → hooks → decay          │
          └────────────────┬────────────────────┘
                           │
                           ▼
                         MEMORY
```

---

# 60. The Most Important Principle

The Amadeus project should not be treated as:

```text
"Build an AI assistant."
```

It should be treated as:

```text
"Build a real-time interactive system whose components
work together under strict latency and hardware constraints."
```

The intelligence comes from the combination of:

```text
LLM
+
Memory
+
Emotion
+
Appraisal
+
Initiative
+
Audio
+
Timing
+
Expression
+
Runtime control
```

A small model inside a well-designed system can feel substantially more coherent than a large model placed inside a poorly designed pipeline.

---

# 61. Immediate Next Action

The recommended first implementation package is:

```text
M0 → M1
```

Specifically:

```text
1. Create project skeleton
2. Create pyproject.toml
3. Create environment checker
4. Create config system
5. Create trace/logging system
6. Create VRAM monitor
7. Create benchmark scripts
8. Verify CUDA / PyTorch
9. Verify audio devices
10. Record the baseline measurements
```

Do not implement memory, emotion, initiative, avatar, or LoRA yet.

Once M0–M1 are stable, continue to:

```text
M2 → M3 → M4 → M5 → M6 → M7
```

and reach the first real voice MVP before expanding the architecture.

---

# 62. Summary

The correct build order is:

```text
FOUNDATION
    ↓
MEASUREMENT
    ↓
AUDIO
    ↓
STT
    ↓
LLM
    ↓
TTS
    ↓
VOICE MVP
    ↓
STREAMING
    ↓
BARGE-IN
    ↓
EMOTION
    ↓
MEMORY
    ↓
REFLECTION
    ↓
INITIATIVE
    ↓
FULL DUPLEX
    ↓
AVATAR
    ↓
PERSONA LORA
```

The project should grow from a **small reliable conversational core** into a **large autonomous embodied agent**.

Do not build the complexity first.

Build the smallest working system, measure it, stabilize it, and then add one intelligence layer at a time.
