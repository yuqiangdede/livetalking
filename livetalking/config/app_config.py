from __future__ import annotations

import copy
import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SYSTEM_PROMPT_ZH = "\u4f60\u662f\u4e00\u4e2a\u79bb\u7ebf\u6570\u5b57\u4eba\u52a9\u624b\u3002\u8bf7\u5728\u9700\u8981\u65f6\u4f7f\u7528\u4e2d\u6587\u7b80\u6d01\u51c6\u786e\u5730\u56de\u7b54\u3002\u6700\u591a\u4e09\u53e5\u8bdd\uff0c\u4e0d\u8981\u4f7f\u7528\u8868\u60c5\u3002"
DEFAULT_SYSTEM_PROMPT_EN = "You are an offline digital human assistant. Reply concisely and accurately."
SETTINGS_DB_RELATIVE_PATH = Path("data") / "runtime" / "settings.sqlite3"


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


def resolve_coqui_reference_audio_path(config: "AppConfig", path_value: str) -> tuple[str, str]:
    configured_rel = str(path_value or "").strip().replace("\\", "/")
    configured_abs = Path(resolve_project_path(config, configured_rel)) if configured_rel else None
    candidates: list[tuple[str, Path]] = []

    def add_candidate(rel_value: str) -> None:
        rel_norm = str(rel_value or "").strip().replace("\\", "/")
        if not rel_norm:
            return
        abs_path = Path(resolve_project_path(config, rel_norm))
        candidates.append((rel_norm, abs_path))

    if configured_rel:
        add_candidate(configured_rel)
        configured_path = Path(configured_rel)
        if configured_path.suffix.lower() == ".wav":
            add_candidate(configured_path.with_suffix(".mp3").as_posix())
        elif configured_path.suffix.lower() == ".mp3":
            add_candidate(configured_path.with_suffix(".wav").as_posix())

    add_candidate("data/tts/coqui_xtts_v2/reference.wav")
    add_candidate("data/tts/coqui_xtts_v2/reference.mp3")

    seen: set[str] = set()
    for rel_norm, abs_path in candidates:
        key = f"{rel_norm}|{abs_path}"
        if key in seen:
            continue
        seen.add(key)
        if abs_path.is_file():
            return rel_norm, str(abs_path)

    reference_root = Path(resolve_project_path(config, "data/tts/coqui_xtts_v2"))
    if reference_root.is_dir():
        files = [
            path for path in reference_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".wav", ".mp3"}
        ]
        if files:
            selected = sorted(files, key=lambda item: str(item.relative_to(reference_root)).lower())[0]
            rel_norm = selected.resolve().relative_to(config.project_root.resolve()).as_posix()
            return rel_norm, str(selected.resolve())

    fallback_rel = configured_rel or "data/tts/coqui_xtts_v2/reference.wav"
    fallback_abs = str(configured_abs or Path(resolve_project_path(config, fallback_rel)))
    return fallback_rel, fallback_abs


def get_settings_db_path(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / SETTINGS_DB_RELATIVE_PATH


def _app_config_payload(config: AppConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("project_root", None)
    return payload


def _build_app_config_from_payload(payload: dict[str, Any], project_root: Path) -> AppConfig:
    tts_payload = payload["tts"]
    return AppConfig(
        providers=ProvidersConfig(**payload["providers"]),
        runtime=RuntimeConfig(**payload["runtime"]),
        asr=ASRConfig(paraformer=ASRParaformerConfig(**payload["asr"]["paraformer"])),
        tts=TTSConfig(
            vits_zh=TTSSherpaOnnxVitsConfig(**tts_payload["vits_zh"]),
            vits_melo_zh_en=TTSSherpaOnnxVitsZhEnConfig(**tts_payload["vits_melo_zh_en"]),
            qwen3_customvoice=TTSQwenCustomVoiceConfig(**tts_payload["qwen3_customvoice"]),
            coqui_xtts_v2=TTSCoquiXTTSV2Config(**tts_payload["coqui_xtts_v2"]),
            edgetts=TTSEdgeTTSConfig(**tts_payload["edgetts"]),
            pyttsx3=TTSPyttsx3Config(**tts_payload["pyttsx3"]),
        ),
        avatar=AvatarConfig(wav2lip=AvatarWav2LipConfig(**payload["avatar"]["wav2lip"])),
        llm=LLMConfig(
            openai_chat=LLMOpenAIChatConfig(**payload["llm"]["openai_chat"]),
            chat_completions_api=LLMChatCompletionsAPIConfig(**payload["llm"]["chat_completions_api"]),
            lm_studio_api=LLMLMStudioAPIConfig(**payload["llm"]["lm_studio_api"]),
        ),
        project_root=project_root,
    )


def _load_settings_snapshot(db_path: Path) -> dict[str, Any] | None:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT config_json FROM runtime_settings WHERE id = 1"
            ).fetchone()
            if row is None:
                return None
            return json.loads(row["config_json"])
    except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError):
        return None


def _strip_legacy_llm_fields(payload: dict[str, Any]) -> dict[str, Any]:
    llm_payload = payload.get("llm")
    if not isinstance(llm_payload, dict):
        return payload

    cleaned_llm_payload = copy.deepcopy(llm_payload)
    openai_chat = cleaned_llm_payload.get("openai_chat")
    if isinstance(openai_chat, dict):
        openai_chat.pop("web_search_enabled", None)
        openai_chat.pop("web_search_max_keyword", None)
    chat_completions_api = cleaned_llm_payload.get("chat_completions_api")
    if isinstance(chat_completions_api, dict):
        chat_completions_api.pop("stream", None)
    lm_studio_api = cleaned_llm_payload.get("lm_studio_api")
    if isinstance(lm_studio_api, dict):
        lm_studio_api.pop("stream", None)

    cleaned_payload = copy.deepcopy(payload)
    cleaned_payload["llm"] = cleaned_llm_payload
    return cleaned_payload


def _ensure_settings_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            config_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _save_settings_snapshot(db_path: Path, payload: dict[str, Any]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        _ensure_settings_schema(connection)
        connection.execute(
            """
            INSERT INTO runtime_settings (id, config_json, updated_at)
            VALUES (1, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                config_json = excluded.config_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (json.dumps(payload, ensure_ascii=False),),
        )
        connection.commit()


@dataclass
class ProvidersConfig:
    asr: str = "paraformer"
    tts: str = "vits_zh"
    avatar: str = "wav2lip"
    llm: str = "lm_studio_api"


@dataclass
class RuntimeConfig:
    temp_dir: str = "runtime/tmp"
    log_level: str = "INFO"


@dataclass
class ASRParaformerConfig:
    model_dir: str = "models/asr/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    punc_model_dir: str = "models/asr/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    device: str = "cuda:0"
    batch_size_s: int = 300
    hotwords: str = ""
    phonetic_replacements: str = ""
    segment_by_silence: bool = True
    segment_min_silence_ms: int = 650
    segment_min_speech_ms: int = 240
    segment_padding_ms: int = 120
    segment_max_duration_s: float = 18.0
    segment_split_overlap_ms: int = 150


@dataclass
class ASRConfig:
    paraformer: ASRParaformerConfig = field(default_factory=ASRParaformerConfig)


@dataclass
class TTSSherpaOnnxVitsConfig:
    model_dir: str = "models/tts/sherpa-onnx-vits-zh-ll"
    provider: str = "cuda"
    num_threads: int = 2
    speaker_id: int = 4
    speed: float = 1.0
    rule_fsts: list[str] = field(default_factory=lambda: ["phone.fst", "date.fst", "number.fst"])


@dataclass
class TTSSherpaOnnxVitsZhEnConfig:
    model_dir: str = "models/tts/vits-melo-tts-zh_en"
    provider: str = "cuda"
    num_threads: int = 2
    speaker_id: int = 1
    speed: float = 1.0
    rule_fsts: list[str] = field(default_factory=lambda: ["phone.fst", "date.fst", "number.fst"])


@dataclass
class TTSEdgeTTSConfig:
    voice_name: str = "zh-CN-XiaoxiaoNeural"
    speed: str = "0%"


@dataclass
class TTSPyttsx3Config:
    voice_id: str = ""
    rate: int = 175
    volume: float = 1.0
    driver_name: str = ""


@dataclass
class TTSQwenCustomVoiceConfig:
    model_dir: str = "models/tts/qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    tokenizer_dir: str = "models/tts/qwen/Qwen3-TTS-Tokenizer-12Hz"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    speaker: str = "Vivian"
    instruct: str = ""
    language: str = "Auto"
    speed: float = 1.0


@dataclass
class TTSCoquiXTTSV2Config:
    model_dir: str = "models/tts/coqui/xtts_v2"
    speaker_wav_path: str = "data/tts/coqui_xtts_v2/reference.wav"
    language: str = "zh-cn"
    device: str = "cuda:0"
    speed: float = 1.1


@dataclass
class TTSConfig:
    vits_zh: TTSSherpaOnnxVitsConfig = field(default_factory=TTSSherpaOnnxVitsConfig)
    vits_melo_zh_en: TTSSherpaOnnxVitsZhEnConfig = field(default_factory=TTSSherpaOnnxVitsZhEnConfig)
    qwen3_customvoice: TTSQwenCustomVoiceConfig = field(default_factory=TTSQwenCustomVoiceConfig)
    coqui_xtts_v2: TTSCoquiXTTSV2Config = field(default_factory=TTSCoquiXTTSV2Config)
    edgetts: TTSEdgeTTSConfig = field(default_factory=TTSEdgeTTSConfig)
    pyttsx3: TTSPyttsx3Config = field(default_factory=TTSPyttsx3Config)


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


@dataclass
class LLMChatCompletionsAPIConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_s: int = 60
    system_prompt_zh: str = DEFAULT_SYSTEM_PROMPT_ZH
    system_prompt_en: str = DEFAULT_SYSTEM_PROMPT_EN


@dataclass
class LLMLMStudioAPIConfig:
    base_url: str = "http://127.0.0.1:1234/api/v1/chat"
    api_key: str = ""
    model: str = ""
    timeout_s: int = 60
    system_prompt_zh: str = DEFAULT_SYSTEM_PROMPT_ZH
    system_prompt_en: str = DEFAULT_SYSTEM_PROMPT_EN


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
    if engine in {"", "sherpa_onnx_vits", "vits_zh"}:
        return "vits_zh"
    if engine in {"sherpa_onnx_vits_zh_en", "vits_melo_tts_zh_en", "sherpa_onnx_melo_tts_zh_en", "vits_melo_zh_en"}:
        return "vits_melo_zh_en"
    if engine in {"qwen3_tts", "qwen3_customvoice", "qwen3_tts_customvoice", "qwen_customvoice"}:
        return "qwen3_customvoice"
    if engine in {"coqui_tts", "coqui_xtts_v2", "xtts", "xtts_v2"}:
        return "coqui_xtts_v2"
    if engine == "edgetts":
        return "edgetts"
    if engine in {"pyttsx3", "pyttsx3_tts", "pyttsx3_sapi5", "sapi5"}:
        return "pyttsx3"
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
    db_path = get_settings_db_path(root)

    defaults = asdict(AppConfig())
    incoming: dict[str, Any] = {}
    if path.exists():
        incoming = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    merged = _deep_merge(defaults, incoming)
    stored_snapshot = _load_settings_snapshot(db_path)
    if stored_snapshot:
        merged = _deep_merge(merged, stored_snapshot)
    merged = _strip_legacy_llm_fields(merged)
    for llm_key in ("openai_chat", "chat_completions_api", "lm_studio_api"):
        if llm_key in merged.get("llm", {}):
            merged["llm"][llm_key]["base_url"] = clean_base_url(merged["llm"][llm_key].get("base_url", ""))
    config = _build_app_config_from_payload(merged, root)

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
    db_path = get_settings_db_path(config.project_root)
    payload = _app_config_payload(config)
    _save_settings_snapshot(db_path, payload)
    return str(db_path)


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
    if opt.tts == "vits_zh":
        sherpa_config = config.tts.vits_zh
    elif opt.tts == "vits_melo_zh_en":
        sherpa_config = config.tts.vits_melo_zh_en
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
    elif opt.tts == "qwen3_customvoice":
        qwen_config = config.tts.qwen3_customvoice
        opt.REF_FILE = resolve_project_path(config, qwen_config.model_dir)
        opt.REF_TEXT = qwen_config.instruct or None
        opt.TTS_SERVER = ""
        opt.TTS_MODEL_DIR = resolve_project_path(config, qwen_config.model_dir)
        opt.TTS_TOKENIZER_DIR = resolve_project_path(config, qwen_config.tokenizer_dir)
        opt.TTS_DEVICE = qwen_config.device
        opt.TTS_DTYPE = qwen_config.dtype
        opt.TTS_ATTN_IMPLEMENTATION = qwen_config.attn_implementation
        opt.TTS_QWEN_SPEAKER = qwen_config.speaker
        opt.TTS_QWEN_INSTRUCT = qwen_config.instruct
        opt.TTS_QWEN_LANGUAGE = qwen_config.language
        opt.TTS_QWEN_SPEED = qwen_config.speed
        opt.TTS_PROVIDER = ""
        opt.TTS_NUM_THREADS = 1
        opt.TTS_SPEAKER_ID = 0
        opt.TTS_SPEED = 1.0
        opt.TTS_RULE_FSTS = []
    elif opt.tts == "coqui_xtts_v2":
        coqui_config = config.tts.coqui_xtts_v2
        resolved_rel, resolved_abs = resolve_coqui_reference_audio_path(config, coqui_config.speaker_wav_path)
        coqui_config.speaker_wav_path = resolved_rel
        opt.REF_FILE = resolved_abs
        opt.REF_TEXT = None
        opt.TTS_SERVER = ""
        opt.TTS_MODEL_DIR = resolve_project_path(config, coqui_config.model_dir)
        opt.TTS_SPEAKER_WAV_PATH = resolved_abs
        opt.TTS_LANGUAGE = coqui_config.language
        opt.TTS_DEVICE = coqui_config.device
        opt.TTS_PROVIDER = ""
        opt.TTS_NUM_THREADS = 1
        opt.TTS_SPEAKER_ID = 0
        opt.TTS_SPEED = coqui_config.speed
        opt.TTS_RULE_FSTS = []
    elif opt.tts == "pyttsx3":
        pyttsx3_config = config.tts.pyttsx3
        opt.REF_FILE = ""
        opt.REF_TEXT = None
        opt.TTS_SERVER = ""
        opt.TTS_MODEL_DIR = ""
        opt.TTS_TOKENIZER_DIR = ""
        opt.TTS_PROVIDER = ""
        opt.TTS_NUM_THREADS = 1
        opt.TTS_SPEAKER_ID = 0
        opt.TTS_SPEED = 1.0
        opt.TTS_RULE_FSTS = []
        opt.TTS_PYTTSX3_DRIVER_NAME = pyttsx3_config.driver_name
        opt.TTS_PYTTSX3_VOICE_ID = ""
        opt.TTS_PYTTSX3_RATE = pyttsx3_config.rate
        opt.TTS_PYTTSX3_VOLUME = pyttsx3_config.volume
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
