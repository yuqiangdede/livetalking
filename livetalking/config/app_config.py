from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SYSTEM_PROMPT_ZH = "\u4f60\u662f\u4e00\u4e2a\u79bb\u7ebf\u6570\u5b57\u4eba\u52a9\u624b\u3002\u8bf7\u5728\u9700\u8981\u65f6\u4f7f\u7528\u4e2d\u6587\u7b80\u6d01\u51c6\u786e\u5730\u56de\u7b54\u3002\u6700\u591a\u4e09\u53e5\u8bdd\uff0c\u4e0d\u8981\u4f7f\u7528\u8868\u60c5\u3002"
DEFAULT_SYSTEM_PROMPT_EN = "You are an offline digital human assistant. Reply concisely and accurately."


def repair_mojibake(text: str) -> str:
    if not text:
        return text
    markers = ("浣犳槸", "璇", "鍥炵瓟", "銆?", "闊宠壊", "寮曟搸")
    if not any(marker in text for marker in markers):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def clean_base_url(value: str) -> str:
    base_url = str(value or "").strip()
    if not base_url:
        return base_url
    return base_url.strip("'\"")


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_project_path(config: "AppConfig", path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((config.project_root / path).resolve())


def ensure_directory(config: "AppConfig", path_value: str) -> str:
    path = Path(resolve_project_path(config, path_value))
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


@dataclass
class ProvidersConfig:
    asr: str = "paraformer"
    tts: str = "sherpa_onnx_vits"
    avatar: str = "wav2lip"
    llm: str = "lm_studio_api"


@dataclass
class RuntimeConfig:
    temp_dir: str = "runtime/tmp"


@dataclass
class ASRParaformerConfig:
    model_dir: str = "models/asr/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    punc_model_dir: str = "models/asr/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    device: str = "cuda:0"
    batch_size_s: int = 300
    hotwords: str = ""
    phonetic_replacements: str = ""


@dataclass
class ASRConfig:
    paraformer: ASRParaformerConfig = field(default_factory=ASRParaformerConfig)


@dataclass
class TTSSherpaOnnxVitsConfig:
    model_dir: str = "models/tts/sherpa-onnx-vits-zh-ll"
    provider: str = "cpu"
    num_threads: int = 2
    speaker_id: int = 4
    speed: float = 1.0
    rule_fsts: list[str] = field(default_factory=lambda: ["phone.fst", "date.fst", "number.fst"])


@dataclass
class TTSSherpaOnnxVitsZhEnConfig:
    model_dir: str = "models/tts/vits-melo-tts-zh_en"
    provider: str = "cpu"
    num_threads: int = 2
    speaker_id: int = 1
    speed: float = 1.0
    rule_fsts: list[str] = field(default_factory=lambda: ["phone.fst", "date.fst", "number.fst"])


@dataclass
class TTSEdgeTTSConfig:
    voice_name: str = "zh-CN-XiaoxiaoNeural"
    speed: str = "0%"


@dataclass
class TTSConfig:
    sherpa_onnx_vits: TTSSherpaOnnxVitsConfig = field(default_factory=TTSSherpaOnnxVitsConfig)
    sherpa_onnx_vits_zh_en: TTSSherpaOnnxVitsZhEnConfig = field(default_factory=TTSSherpaOnnxVitsZhEnConfig)
    edgetts: TTSEdgeTTSConfig = field(default_factory=TTSEdgeTTSConfig)


@dataclass
class AvatarWav2LipConfig:
    model_dir: str = "models/avatar/wav2lip256"
    checkpoint_name: str = "wav2lip256.pth"
    model_path: str = "models/avatar/wav2lip256/wav2lip256.pth"
    avatar_dir: str = "data/avatars"
    avatar_id: str = "avatar_1"
    silence_gate_enabled: bool = False
    silence_gate_by_avatar: dict[str, bool] = field(default_factory=dict)
    legacy_model_paths: list[str] = field(default_factory=lambda: ["models/avatar/wav2lip/wav2lip.pth"])


@dataclass
class AvatarConfig:
    wav2lip: AvatarWav2LipConfig = field(default_factory=AvatarWav2LipConfig)


@dataclass
class LLMOpenAIChatConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = ""
    timeout_s: int = 60
    system_prompt_zh: str = DEFAULT_SYSTEM_PROMPT_ZH
    system_prompt_en: str = DEFAULT_SYSTEM_PROMPT_EN
    web_search_enabled: bool = False
    web_search_max_keyword: int = 2


@dataclass
class LLMChatCompletionsAPIConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_s: int = 60
    system_prompt_zh: str = DEFAULT_SYSTEM_PROMPT_ZH
    system_prompt_en: str = DEFAULT_SYSTEM_PROMPT_EN
    stream: bool = False


@dataclass
class LLMLMStudioAPIConfig:
    base_url: str = "http://127.0.0.1:1234/api/v1/chat"
    api_key: str = ""
    model: str = ""
    timeout_s: int = 60
    system_prompt_zh: str = DEFAULT_SYSTEM_PROMPT_ZH
    system_prompt_en: str = DEFAULT_SYSTEM_PROMPT_EN
    stream: bool = False


@dataclass
class LLMConfig:
    openai_chat: LLMOpenAIChatConfig = field(default_factory=LLMOpenAIChatConfig)
    chat_completions_api: LLMChatCompletionsAPIConfig = field(default_factory=LLMChatCompletionsAPIConfig)
    lm_studio_api: LLMLMStudioAPIConfig = field(default_factory=LLMLMStudioAPIConfig)


@dataclass
class AppConfig:
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    project_root: Path = field(default_factory=Path.cwd, repr=False, compare=False)


def normalize_tts_engine(value: str) -> str:
    engine = str(value or "").strip().lower().replace("-", "_")
    if engine in {"", "sherpa_onnx_vits"}:
        return "sherpa_onnx_vits"
    if engine in {"sherpa_onnx_vits_zh_en", "vits_melo_tts_zh_en", "sherpa_onnx_melo_tts_zh_en"}:
        return "sherpa_onnx_vits_zh_en"
    if engine == "edgetts":
        return "edgetts"
    return engine


def normalize_llm_mode(value: str) -> str:
    mode = str(value or "").strip().lower().replace("-", "_")
    if mode in {"", "openai_chat", "openai_responses", "responses", "responses_api"}:
        return "openai_chat"
    if mode in {"chat_completions_api", "chat_completions"}:
        return "chat_completions_api"
    if mode in {"lm_studio_api", "lm_studio", "lmstudio", "lmstudio_api"}:
        return "lm_studio_api"
    return "openai_chat"


def load_app_config(config_path: str | None = None, project_root: str | Path | None = None) -> AppConfig:
    root = Path(project_root or Path.cwd()).resolve()
    path = Path(config_path or (root / "configs" / "app.yaml")).resolve()

    defaults = asdict(AppConfig())
    incoming: dict[str, Any] = {}
    if path.exists():
        incoming = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    merged = _deep_merge(defaults, incoming)
    for llm_key in ("openai_chat", "chat_completions_api", "lm_studio_api"):
        if llm_key in merged.get("llm", {}):
            merged["llm"][llm_key]["base_url"] = clean_base_url(merged["llm"][llm_key].get("base_url", ""))
    config = AppConfig(
        providers=ProvidersConfig(**merged["providers"]),
        runtime=RuntimeConfig(**merged["runtime"]),
        asr=ASRConfig(paraformer=ASRParaformerConfig(**merged["asr"]["paraformer"])),
        tts=TTSConfig(
            sherpa_onnx_vits=TTSSherpaOnnxVitsConfig(**merged["tts"]["sherpa_onnx_vits"]),
            sherpa_onnx_vits_zh_en=TTSSherpaOnnxVitsZhEnConfig(**merged["tts"]["sherpa_onnx_vits_zh_en"]),
            edgetts=TTSEdgeTTSConfig(**merged["tts"]["edgetts"]),
        ),
        avatar=AvatarConfig(wav2lip=AvatarWav2LipConfig(**merged["avatar"]["wav2lip"])),
        llm=LLMConfig(
            openai_chat=LLMOpenAIChatConfig(**merged["llm"]["openai_chat"]),
            chat_completions_api=LLMChatCompletionsAPIConfig(**merged["llm"]["chat_completions_api"]),
            lm_studio_api=LLMLMStudioAPIConfig(**merged["llm"]["lm_studio_api"]),
        ),
        project_root=root,
    )

    config.providers.avatar = "wav2lip"
    config.providers.asr = "paraformer"
    config.providers.llm = normalize_llm_mode(config.providers.llm)

    env_api_key = os.getenv("LIVETALKING_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if env_api_key:
        config.llm.openai_chat.api_key = env_api_key
        config.llm.chat_completions_api.api_key = env_api_key
        config.llm.lm_studio_api.api_key = env_api_key

    config.llm.openai_chat.system_prompt_zh = repair_mojibake(config.llm.openai_chat.system_prompt_zh)
    config.llm.chat_completions_api.system_prompt_zh = repair_mojibake(config.llm.chat_completions_api.system_prompt_zh)
    config.llm.lm_studio_api.system_prompt_zh = repair_mojibake(config.llm.lm_studio_api.system_prompt_zh)

    return config


def save_app_config(config: AppConfig, config_path: str | None = None) -> str:
    path = Path(config_path or (config.project_root / "configs" / "app.yaml")).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload.pop("project_root", None)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(path)


def apply_config_to_opt(opt, config: AppConfig):
    opt.model = "wav2lip"
    opt.tts = getattr(opt, "tts", config.providers.tts) or config.providers.tts
    opt.tts = normalize_tts_engine(opt.tts)
    config.providers.tts = opt.tts
    opt.avatar_id = getattr(opt, "avatar_id", None) or config.avatar.wav2lip.avatar_id
    config.avatar.wav2lip.avatar_id = opt.avatar_id
    silence_gate_map = config.avatar.wav2lip.silence_gate_by_avatar or {}
    opt.SILENCE_GATE_ENABLED = bool(
        silence_gate_map.get(config.avatar.wav2lip.avatar_id, config.avatar.wav2lip.silence_gate_enabled)
    )
    opt.ASR_MODEL_DIR = resolve_project_path(config, config.asr.paraformer.model_dir)
    opt.ASR_PUNC_MODEL_DIR = resolve_project_path(config, config.asr.paraformer.punc_model_dir)
    opt.ASR_DEVICE = config.asr.paraformer.device
    opt.ASR_BATCH_SIZE_S = config.asr.paraformer.batch_size_s
    opt.ASR_HOTWORDS = config.asr.paraformer.hotwords
    opt.ASR_PHONETIC_REPLACEMENTS = config.asr.paraformer.phonetic_replacements
    opt.RUNTIME_TEMP_DIR = ensure_directory(config, config.runtime.temp_dir)
    opt.WAV2LIP_MODEL_PATH = resolve_project_path(config, config.avatar.wav2lip.model_path)
    opt.WAV2LIP_FACE_SIZE = 256
    opt.AVATAR_DIR = resolve_project_path(config, config.avatar.wav2lip.avatar_dir)
    if opt.tts == "sherpa_onnx_vits":
        sherpa_config = config.tts.sherpa_onnx_vits
    elif opt.tts == "sherpa_onnx_vits_zh_en":
        sherpa_config = config.tts.sherpa_onnx_vits_zh_en
    else:
        sherpa_config = None

    if sherpa_config is not None:
        opt.REF_FILE = resolve_project_path(config, sherpa_config.model_dir)
        opt.REF_TEXT = None
        opt.TTS_SERVER = ""
        opt.TTS_MODEL_DIR = resolve_project_path(config, sherpa_config.model_dir)
        opt.TTS_PROVIDER = sherpa_config.provider
        opt.TTS_NUM_THREADS = sherpa_config.num_threads
        opt.TTS_SPEAKER_ID = sherpa_config.speaker_id
        opt.TTS_SPEED = sherpa_config.speed
        opt.TTS_RULE_FSTS = list(sherpa_config.rule_fsts)
    elif opt.tts == "edgetts":
        opt.REF_FILE = config.tts.edgetts.voice_name
        opt.REF_TEXT = None
        opt.TTS_SERVER = ""
        opt.TTS_MODEL_DIR = ""
        opt.TTS_PROVIDER = ""
        opt.TTS_NUM_THREADS = 1
        opt.TTS_SPEAKER_ID = 0
        opt.TTS_SPEED = config.tts.edgetts.speed
        opt.TTS_RULE_FSTS = []
    else:
        raise ValueError(f"Unsupported TTS provider: {opt.tts}")
    opt.CONFIG_PATH = str(Path(getattr(opt, "config", "configs/app.yaml")).resolve())
    return opt


def load_custom_actions(opt) -> None:
    opt.customopt = []
    if not opt.customvideo_config:
        return
    with open(opt.customvideo_config, "r", encoding="utf-8") as file:
        opt.customopt = json.load(file)
