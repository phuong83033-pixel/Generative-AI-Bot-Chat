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

| Metric | Target | Measured (2026-10-07) |
|---|---:|---:|
| First audible syllable p50 | ≤ 1200 ms | not yet - needs `main.py` |
| Optimized first syllable | ~900 ms | — |
| First audible syllable p95 | ≤ 1800 ms | not yet |
| Optimized p95 | ~1300 ms | — |
| Barge-in reaction | ≤ 150 ms | not yet |
| Backchannel | ≤ 300 ms | — |
| Sustained LLM decoding | ≥ 25 tok/s | **41.6 tok/s** PASS |
| LLM time to first token | (not set) | **82 ms** |
| Peak VRAM | ≤ 3.6 GB | **3398 MB (3.32 GB)** PASS |

These measurements should be treated as engineering requirements rather than optional optimizations.

End-to-end latency cannot be measured until `amadeus/main.py` exists. The two
numbers that *are* measured both pass, and decode throughput has useful margin:
41.6 tok/s against conversational speech of roughly 4 tok/s means the LLM is
about 10x faster than the mouth it feeds, so the clause chunker will never
starve. See section 31 for the full measurement record and method.

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
| TTS | Kokoro-82M int8 CPU | CPU | CPU |
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
int8
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
scripts/bench_latency.py     - needs main.py; stage-by-stage p50/p95
scripts/bench_aec.py         - needs real AEC; echo return loss
```

Whisper transcription latency was not isolated in this pass — the benchmark only
forces a dummy decode so CTranslate2 allocates. Real STT timing belongs in
`bench_latency.py`, against actual utterances.

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

# 39. M9 — Emotion

Implement:

```text
emotion/pad.py
emotion/appraisal.py
emotion/labels.py
```

The system should maintain state across time.

Example:

```python
pad = PADState(
    pleasure=0.0,
    arousal=0.0,
    dominance=0.0
)
```

Then update it using deltas.

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

Only after the voice system works:

```text
PAD
 ↓
Emotion
 ↓
Expression
 ↓
VTube Studio
```

The avatar is an output consumer.

Keep it isolated from core cognition.

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
