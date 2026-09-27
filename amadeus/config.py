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
class AmadeusConfig:
    audio: AudioConfig
    vad: VadConfig
    stt: SttConfig
    llm: LlmConfig
    tts: TtsConfig
    latency_targets: LatencyTargets
    persona: PersonaConfig
    log_level: str = "INFO"

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
            output_device=output_device if output_device else None
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
        log_level=log_level
    )
