# RUN GUIDE — Amadeus

Everything needed to get the project running, with copy-paste commands.

> **Current stage: M8 (streaming pipeline), on top of the M7 voice loop.** A turn-based voice loop:
> mic → VAD → Whisper → Qwen → clause chunker → Kokoro (local voice) → speakers.
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
| Network | needed for setup only. The default voice is local and works offline (the optional Edge-TTS fallback needs it) | |

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

`pydub` shells out to it to decode Edge-TTS audio. **It is only needed for the optional cloud
voice** (`tts.provider: edge`); the default local Kokoro voice does not use it.

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

### 3.7 Download the local voice (Kokoro-82M)

The default voice (`tts.provider: kokoro`) runs on the CPU, offline, using **0 MB of
VRAM**. It needs two files, 337 MB together, from the official
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) release.

The model (310.5 MB, 325,532,387 bytes):

```bash
curl -L -C - --retry 5 --create-dirs -o data/models/kokoro/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
```

The voices (26.9 MB, 28,214,398 bytes):

```bash
curl -L -C - --retry 5 -o data/models/kokoro/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

The Python package (`kokoro-onnx`) is already part of the `tts` extra from §3.3.
Downloads from GitHub release assets vary a lot in speed (0.3 to 6 MB/s was seen
here); `-C -` resumes after a drop, so just re-run the same command.

Confirm it loads and says it has 54 voices:

```bash
.\venv\Scripts\python.exe -c "from kokoro_onnx import Kokoro; k = Kokoro('data/models/kokoro/kokoro-v1.0.onnx', 'data/models/kokoro/voices-v1.0.bin'); print(len(k.get_voices()), 'voices')"
```

**Use the fp32 file, `kokoro-v1.0.onnx`. Do not "save space" with the 88 MB
`kokoro-v1.0.int8.onnx`.** On this CPU the int8 model is about 3 times slower:
5.2 s to speak a 4 s sentence, slower than real time and slower than the cloud
voice it replaces. onnxruntime's profile showed 83% of the time in one operator
(`ConvInteger`). The fp32 model takes about 1.5 s for the same sentence. Details are
in the comment at the top of `amadeus/synthesis/kokoro.py`.

Prefer the cloud voice instead? Set `tts.provider: edge` in `config/settings.yaml`
(needs network, and ffmpeg from §3.5). Nothing else changes.

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
[PASS] Kokoro voice files - kokoro-v1.0.onnx + voices, sizes verified, fp32
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
| `--mute` | no sound: playback is simulated by waiting out each clip's real duration |
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
[first audio at 1095 ms]
Amadeus > Hello! I'm Amadeus,
Amadeus > a friendly AI here to chat and explore.
Amadeus > How can I assist you today?
  turn 1
    speech_end -> stt_done                   0 ms
    stt_done -> prompt_built                 0 ms
    prompt_built -> first_token            174 ms
    first_token -> first_clause            170 ms
    first_clause -> tts_done               751 ms
    tts_done -> first_audio_out              0 ms
    = first audible syllable              1095 ms
    silence between clauses                  0 ms max  [0, 0]
```

**First audible syllable is about 1.1 s in text mode** (1062 ms median over 16 test turns,
against a 1200 ms target; it was 2605 ms with the cloud voice). Two caveats you should know:

* **Text mode does not include speech-to-text.** In voice mode add Whisper's time to every
  figure. It has not been measured on real speech yet, so voice mode may sit right at the
  1200 ms limit.
* **The first clause is cut short on purpose.** To start speaking sooner, the first clause ends
  at the first punctuation after 16 characters, or at a word boundary at 30 characters if there
  is no punctuation. Occasionally that cuts mid-sentence (`...a beloved stimulant | that
  energizes...`). If it sounds choppy, set `chunker.first_clause_max_chars: null` in
  `config/settings.yaml`. Speech is then continuous but starts ~600 ms later (p50 ~1700 ms).

"silence between clauses" is the longest pause the listener hears inside a reply. 0 is the
normal case; a long unpunctuated sentence can still produce a pause of about a second.

Measured resource use with all three models loaded (the local voice uses **0 MB** of VRAM):

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

### `FileNotFoundError: Kokoro model file not found`

The local voice files are missing or in the wrong folder. Do §3.7, then run
`scripts\check_environment.py`: its "Kokoro voice files" line checks both files by exact size,
so a truncated download is reported instead of failing later when the model loads.

### The voice takes 5 seconds to say a sentence

You are using the int8 model. It is about 3 times slower than fp32 on this CPU and slower than
real time. Set `tts.model_path` to `data/models/kokoro/kokoro-v1.0.onnx` (§3.7).

### Speech sounds choppy, or cut off mid-sentence

That is the first-clause cap (§6). Set `chunker.first_clause_max_chars: null`. If the *sound
itself* crackles or has gaps between sentences, set `audio.streaming_output: false` to use the
older player, and tell me which, since the new output path has not been verified by ear.

### Which speaker is it using?

Windows changes the default output when devices are plugged in, and it may be **Speakers**
rather than headphones (the stream line in the log names the device). On speakers the agent
hears itself, because echo cancellation is still a stub (§7, "talking to itself"). Check:

```bash
.\venv\Scripts\python.exe -c "import sounddevice as sd; print(sd.query_devices(sd.default.device[1])['name'])"
```

Pick a device with `AMADEUS_AUDIO_OUTPUT_DEVICE` in `.env` (a name fragment such as `Headphones`).

### Using the cloud voice instead

Set `tts.provider: edge` in `config/settings.yaml`. It needs network access and ffmpeg, and adds
about 1900 ms before the first sound.

---

## 8. Benchmarks

Measure VRAM per model plus decode throughput. Close other GPU applications
first; results are written to `data/logs/bench_vram.json`.

```bash
.\venv\Scripts\python.exe scripts\bench_vram.py
```

First-sound latency and the longest silence inside a reply, over 8 prompts (silent: playback is
simulated, so it is quiet and repeatable). About 3 minutes; close GPU apps first.

```bash
.\venv\Scripts\python.exe scripts\bench_latency.py --label mytest --repeats 2
```

Replies are seeded so every run sees identical text, which makes two runs comparable. Try other
first-clause settings without editing the config:

```bash
.\venv\Scripts\python.exe scripts\bench_latency.py --label alt --first-min 12 --first-max none
```

Results go to `data/logs/bench_latency_<label>.json`. It uses typed input, so it does not time
Whisper.

Run the automated tests (about 10 s, no GPU or sound device needed):

```bash
.\venv\Scripts\python.exe -m pytest tests -q
```

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
  provider: kokoro          # "kokoro" = local, offline. "edge" = cloud fallback
  kokoro_voice: af_heart    # 54 voices, e.g. af_bella, af_nicole, bf_emma (British)
  speed: 1.0                # Kokoro speaking speed
  threads: 6                # CPU threads for the voice; 6 = physical cores
chunker:
  first_clause_min_chars: 16   # see section 6
  first_clause_max_chars: 30   # null = never cut mid-sentence
audio:
  streaming_output: true       # false = older player (adds ~200 ms of silence per clip)
```

List the Kokoro voices:

```bash
.\venv\Scripts\python.exe -c "from kokoro_onnx import Kokoro; print(sorted(Kokoro('data/models/kokoro/kokoro-v1.0.onnx','data/models/kokoro/voices-v1.0.bin').get_voices()))"
```

Only used when `tts.provider: edge`: `voice: en-US-AriaNeural` and `rate: "+0%"`. List the Edge
voices with `.\venv\Scripts\python.exe -m edge_tts --list-voices`.

---

## 10. Known limits at M8

* **No avatar.** Nothing renders. M14.
* **No emotion engine.** The PAD model is designed, not built. M9.
* **No memory.** The agent forgets everything when you exit; within a session it
  keeps a short rolling history. M10–M11.
* **No proactive speech.** It only answers; it never starts a conversation. The
  state machine now permits it, but `drive.py` does not exist. M12.
* **No barge-in.** You cannot interrupt it mid-sentence. M8.
* **No echo cancellation.** Headphones required. M13.
* **Local voice has no emotion.** Kokoro offers a choice of voice and a speed, not pitch or energy control, so mood can only change which voice and how fast. M9.
* **Never listened to by its author.** The first-clause cap and the new output stream are verified by measurement and tests, not by ear (§6, §7).
* **Voice path unproven.** The microphone opens and the detector listens, but no
  real utterance has yet been captured and transcribed end to end. Text mode is
  verified working.

Full roadmap in README §28–45. Defect history in `fix.md`.
