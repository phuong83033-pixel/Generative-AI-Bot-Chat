import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    channels: int
    chunk_size: int
    output_sample_rate: int
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    # One persistent WASAPI output stream (gapless, ~25 ms latency) instead of an
    # sd.play() per clip (~200 ms of silence per clause boundary). Set false to fall
    # back to the legacy player if the streaming one misbehaves on a given machine.
    streaming_output: bool = True

@dataclass(frozen=True)
class VadConfig:
    model: str
    threshold: float
    min_speech_ms: int
    silence_timeout_ms: int
    pre_speech_pad_ms: int

@dataclass(frozen=True)
class SttConfig:
    model_size: str
    compute_type: str
    language: Optional[str]
    device: str

@dataclass(frozen=True)
class LlmConfig:
    model_path: str
    n_ctx: int
    n_gpu_layers: int
    temperature: float
    max_tokens: int
    top_p: float
    repeat_penalty: float

@dataclass(frozen=True)
class TtsConfig:
    provider: str
    voice: str
    rate: str
    output_sample_rate: int
    # Local Kokoro-82M (provider: kokoro). Defaulted so a settings.yaml that
    # predates the local voice still loads; `rate` applies to Edge only.
    #
    # `voice` is the EDGE voice name and `kokoro_voice` the Kokoro one: the two
    # providers use incompatible naming ("en-US-AriaNeural" vs "af_heart"), so
    # sharing one field made switching provider fail on an unknown voice.
    #
    # fp32, not int8: on this CPU the int8 graph is ~3x slower (README 31.6).
    model_path: str = "data/models/kokoro/kokoro-v1.0.onnx"
    voices_path: str = "data/models/kokoro/voices-v1.0.bin"
    kokoro_voice: str = "af_heart"
    speed: float = 1.0
    threads: int = 6  # measured optimum = physical cores; 12 hyperthreads was slower

@dataclass(frozen=True)
class LatencyTargets:
    first_syllable_p50_ms: int
    first_syllable_p95_ms: int
    barge_in_ms: int
    peak_vram_gb: float

@dataclass(frozen=True)
class PersonaConfig:
    name: str
    personality: Dict[str, Any]
    system_prompt: str

@dataclass(frozen=True)
class ChunkerConfig:
    # Minimum characters before the FIRST clause of a reply may break. None keeps
    # the original rules for every clause. See ClauseChunker.__init__ for why this
    # is a trade between first-sound latency and silence before the second clause.
    first_clause_min_chars: Optional[int] = None
    # Hard cap for the first clause: end it at a word boundary once it reaches this
    # length even without punctuation. None = off (no mid-sentence seams).
    first_clause_max_chars: Optional[int] = None

@dataclass(frozen=True)
class AmadeusConfig:
    audio: AudioConfig
    vad: VadConfig
    stt: SttConfig
    llm: LlmConfig
    tts: TtsConfig
    latency_targets: LatencyTargets
    persona: PersonaConfig
    log_level: str = "INFO"
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)

def load_config(config_dir: Optional[Path] = None) -> AmadeusConfig:
    if config_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        config_dir = project_root / "config"
    
    load_dotenv()
    
    with open(config_dir / "settings.yaml", "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
        
    with open(config_dir / "persona.yaml", "r", encoding="utf-8") as f:
        persona_data = yaml.safe_load(f)
        
    log_level = os.getenv("AMADEUS_LOG_LEVEL", "INFO")
    model_path = os.getenv("AMADEUS_MODEL_PATH", settings["llm"]["model_path"])
    input_device = os.getenv("AMADEUS_AUDIO_INPUT_DEVICE")
    output_device = os.getenv("AMADEUS_AUDIO_OUTPUT_DEVICE")
    
    return AmadeusConfig(
        audio=AudioConfig(
            sample_rate=settings["audio"]["sample_rate"],
            channels=settings["audio"]["channels"],
            chunk_size=settings["audio"]["chunk_size"],
            output_sample_rate=settings["audio"]["output_sample_rate"],
            input_device=input_device if input_device else None,
            output_device=output_device if output_device else None,
            streaming_output=settings["audio"].get("streaming_output", True),
        ),
        vad=VadConfig(**settings["vad"]),
        stt=SttConfig(**settings["stt"]),
        llm=LlmConfig(
            model_path=model_path,
            n_ctx=settings["llm"]["n_ctx"],
            n_gpu_layers=settings["llm"]["n_gpu_layers"],
            temperature=settings["llm"]["temperature"],
            max_tokens=settings["llm"]["max_tokens"],
            top_p=settings["llm"]["top_p"],
            repeat_penalty=settings["llm"]["repeat_penalty"]
        ),
        tts=TtsConfig(**settings["tts"]),
        latency_targets=LatencyTargets(**settings["latency_targets"]),
        persona=PersonaConfig(**persona_data),
        log_level=log_level,
        chunker=ChunkerConfig(**(settings.get("chunker") or {})),
    )
