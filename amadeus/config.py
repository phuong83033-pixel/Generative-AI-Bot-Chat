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
class EmotionConfig:
    """M9. Every default is the value shipped in config/emotion.yaml, so a missing file
    behaves identically to the shipped one."""
    enabled: bool = True
    # Personality baseline the mood relaxes toward: (pleasure, arousal, dominance).
    # Slightly pleasant, calm, and confident: she is sure of herself.
    baseline: tuple = (0.05, 0.05, 0.25)
    # Seconds for the distance to the baseline to halve, per axis (P, A, D).
    half_life_s: tuple = (90.0, 45.0, 120.0)
    appraisal_strength: float = 1.0
    # Moods fainter than this add nothing to the prompt.
    hint_min_intensity: float = 0.25
    tick_interval_s: float = 1.0
    # Voice speed = 1 + speed_per_arousal * ΔA + speed_per_pleasure * ΔP (Δ from baseline).
    speed_per_arousal: float = 0.10
    speed_per_pleasure: float = 0.04
    speed_min: float = 0.90
    speed_max: float = 1.12

@dataclass(frozen=True)
class AvatarConfig:
    """M14. Off by default: the avatar is an optional output layer (README section 26)."""
    enabled: bool = False
    fps: int = 30
    # Built-in browser preview. Needs no Live2D model and no VTube Studio.
    web: bool = True
    web_port: int = 8765
    # VTube Studio (needs the app running with a model loaded and its API enabled).
    vtube: bool = False
    vtube_url: str = "ws://127.0.0.1:8001"
    plugin_name: str = "Amadeus"
    plugin_developer: str = "Amadeus project"
    # emotion label -> name of a VTube Studio hotkey (an expression toggle in the model).
    vtube_hotkeys: Dict[str, str] = field(default_factory=dict)
    # Where the access token VTube Studio issues is kept (git-ignored; delete it to re-pair).
    vtube_token_path: str = "data/vtube_token.txt"
    # Parameter names whose direction is backwards for a given model, e.g. ["FaceAngleY"].
    vtube_invert: List[str] = field(default_factory=list)

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
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)


def _load_emotion(config_dir: Path) -> EmotionConfig:
    """config/emotion.yaml is optional; absent means the defaults above."""
    path = config_dir / "emotion.yaml"
    if not path.is_file():
        return EmotionConfig()
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    base = d.get("baseline") or {}
    half = d.get("half_life_s") or {}
    tts = d.get("tts") or {}
    default = EmotionConfig()
    return EmotionConfig(
        enabled=d.get("enabled", default.enabled),
        baseline=(
            float(base.get("pleasure", default.baseline[0])),
            float(base.get("arousal", default.baseline[1])),
            float(base.get("dominance", default.baseline[2])),
        ),
        half_life_s=(
            float(half.get("pleasure", default.half_life_s[0])),
            float(half.get("arousal", default.half_life_s[1])),
            float(half.get("dominance", default.half_life_s[2])),
        ),
        appraisal_strength=float(d.get("appraisal_strength", default.appraisal_strength)),
        hint_min_intensity=float(d.get("hint_min_intensity", default.hint_min_intensity)),
        tick_interval_s=float(d.get("tick_interval_s", default.tick_interval_s)),
        speed_per_arousal=float(tts.get("speed_per_arousal", default.speed_per_arousal)),
        speed_per_pleasure=float(tts.get("speed_per_pleasure", default.speed_per_pleasure)),
        speed_min=float(tts.get("speed_min", default.speed_min)),
        speed_max=float(tts.get("speed_max", default.speed_max)),
    )

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
        emotion=_load_emotion(config_dir),
        avatar=AvatarConfig(**(settings.get("avatar") or {})),
    )
