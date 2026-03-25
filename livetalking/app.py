###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import logging
import random
import re
import shutil
import socket
import os
import time
import tempfile
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote
from typing import Any, Dict

import aiohttp_cors
import cv2
import torch.multiprocessing as mp
from aioice import ice as aioice_ice
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCRtpSender
from .core.base_real import BaseReal
from .config.app_config import (
    apply_config_to_opt,
    get_settings_db_path,
    load_app_config,
    resolve_project_path,
    save_app_config,
)
from .providers.llm_client import configure_llm, llm_response
from .providers.local_asr import ParaformerProvider, decode_audio_to_pcm16
from .utils.app_logger import configure_logging, logger
from .providers.sherpa_tts import SherpaOnnxVitsTTS
from .realtime.webrtc import HumanPlayer


nerfreals: Dict[int, BaseReal] = {}
pcs = set()
opt = None
model = None
avatar = None
app_config = None
speech_recognizer: ParaformerProvider | None = None
_avatar_cache: dict[tuple[str, str], tuple[Any, Any, Any, Any]] = {}
_avatar_generation_jobs: dict[str, dict[str, Any]] = {}
_ORIGINAL_GET_HOST_ADDRESSES = aioice_ice.get_host_addresses
_preferred_ice_addresses: list[str] | None = None
_AVATAR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_AVATAR_JOB_PROGRESS_PATTERN = re.compile(r"(\d{1,3})%\|")
_AVATAR_JOB_LOG_LIMIT = 120

logging.getLogger("aioice").setLevel(logging.INFO)
logging.getLogger("aiortc").setLevel(logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[1]


class InvalidSessionError(RuntimeError):
    pass


SUPPORTED_TTS_ENGINES = {
    "vits_zh",
    "vits_melo_zh_en",
    "edgetts",
}

SUPPORTED_LLM_MODES = {
    "openai_chat",
    "chat_completions_api",
    "lm_studio_api",
}

SHERPA_TTS_ENGINES = {
    "vits_zh",
    "vits_melo_zh_en",
}

TTS_VOICE_OPTIONS_BY_ENGINE = {
    "vits_zh": [
        {"id": 0, "label": "音色 1"},
        {"id": 1, "label": "音色 2"},
        {"id": 2, "label": "音色 3"},
        {"id": 3, "label": "音色 4"},
        {"id": 4, "label": "音色 5"},
    ],
    "vits_melo_zh_en": [
        {"id": 1, "label": "默认音色"},
    ],
    "edgetts": [],
}

def get_tts_voice_options(engine: str | None) -> list[dict]:
    normalized = normalize_tts_engine(engine)
    return list(TTS_VOICE_OPTIONS_BY_ENGINE.get(normalized, TTS_VOICE_OPTIONS_BY_ENGINE["vits_zh"]))

TTS_ENGINE_OPTIONS = [
    {"id": "vits_zh", "label": "vits-zh", "desc": "本地中文 TTS，模型为 k2-fsa/sherpa-onnx-vits-zh-ll。"},
    {"id": "vits_melo_zh_en", "label": "vits-melo-zh_en", "desc": "本地中英 TTS，模型为 vits-melo-tts-zh_en。"},
    {"id": "edgetts", "label": "EdgeTTS", "desc": "在线 TTS，适合英文单词和中英混读。"},
]

LLM_MODE_OPTIONS = [
    {"id": "openai_chat", "label": "OpenAI Responses", "desc": "OpenAI Responses 协议。"},
    {"id": "chat_completions_api", "label": "OpenAI Chat Completions", "desc": "OpenAI chat/completions 协议。"},
    {"id": "lm_studio_api", "label": "LM Studio API", "desc": "LM Studio 原生 /api/v1/chat 接口。"},
]


def normalize_tts_engine(value: str | None) -> str:
    engine = (value or "").strip().lower().replace("-", "_")
    if engine in {"", "sherpa_onnx_vits", "vits_zh"}:
        return "vits_zh"
    if engine in {"sherpa_onnx_vits_zh_en", "vits_melo_tts_zh_en", "sherpa_onnx_melo_tts_zh_en", "vits_melo_zh_en"}:
        return "vits_melo_zh_en"
    if engine == "edgetts":
        return "edgetts"
    return engine


def normalize_llm_mode(value: str | None) -> str:
    mode = (value or "").strip().lower().replace("-", "_")
    if mode in {"", "openai_chat", "openai_responses", "responses", "responses_api"}:
        return "openai_chat"
    if mode in {"chat_completions_api", "chat_completions"}:
        return "chat_completions_api"
    if mode in {"lm_studio_api", "lm_studio", "lmstudio", "lmstudio_api"}:
        return "lm_studio_api"
    return "openai_chat"


def get_llm_mode_meta(mode: str | None) -> dict:
    normalized = normalize_llm_mode(mode)
    for item in LLM_MODE_OPTIONS:
        if item["id"] == normalized:
            return item
    return {"id": normalized, "label": normalized, "desc": normalized}


def get_current_llm_config():
    if app_config is None:
        return None
    mode = normalize_llm_mode(getattr(app_config.providers, "llm", "openai_chat"))
    return getattr(app_config.llm, mode, app_config.llm.openai_chat)


def get_current_llm_model_description() -> str:
    if app_config is None:
        return "OpenAI Responses"
    mode = normalize_llm_mode(getattr(app_config.providers, "llm", "openai_chat"))
    return str(get_llm_mode_meta(mode).get("label") or mode)


def get_tts_engine_meta(engine: str | None) -> dict:
    normalized = normalize_tts_engine(engine)
    for item in TTS_ENGINE_OPTIONS:
        if item["id"] == normalized:
            return item
    return {"id": normalized, "label": normalized, "desc": normalized}


def _sorted_avatar_images(directory: str) -> list[Path]:
    path = Path(directory)
    if not path.is_dir():
        return []
    images = [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".webp"}]

    def _image_sort_key(item: Path) -> tuple[int, int | str]:
        try:
            return (0, int(item.stem))
        except ValueError:
            return (1, item.stem.lower())

    return sorted(images, key=_image_sort_key)


def _resolve_avatar_thumbnail_path(avatar_root: str, avatar_id: str) -> str:
    candidate_ids = [avatar_id]
    if "_" in avatar_id:
        candidate_ids.append(avatar_id.replace("_", ""))

    avatar_path = ""
    for candidate in candidate_ids:
        if not candidate:
            continue
        maybe_path = os.path.join(avatar_root, candidate)
        if os.path.isdir(maybe_path):
            avatar_path = maybe_path
            break
    if not avatar_path:
        return ""

    full_imgs_path = os.path.join(avatar_path, "full_imgs")
    full_images = _sorted_avatar_images(full_imgs_path)
    if full_images:
        return str(full_images[0])
    return ""


def _normalize_avatar_id(value: str) -> str:
    return str(value or "").strip()


def _is_valid_avatar_id(avatar_id: str) -> bool:
    return bool(_AVATAR_ID_PATTERN.fullmatch(avatar_id))


def _avatar_root_path() -> Path:
    avatar_root = _resolve_avatar_root()
    return Path(avatar_root) if avatar_root else Path()


def _avatar_target_path(avatar_id: str) -> Path:
    return _avatar_root_path() / avatar_id


def _list_existing_avatar_ids() -> list[str]:
    return [item["id"] for item in list_available_avatar_ids()]


def _next_avatar_id(existing_ids: list[str] | None = None) -> str:
    ids = existing_ids if existing_ids is not None else _list_existing_avatar_ids()
    max_suffix = 0
    for avatar_id in ids:
        match = re.fullmatch(r"avatar_(\d+)", avatar_id)
        if match:
            max_suffix = max(max_suffix, int(match.group(1)))
    return f"avatar_{max_suffix + 1}"


def _invalidate_avatar_cache(avatar_id: str | None = None) -> None:
    if avatar_id:
        keys_to_remove = [key for key in _avatar_cache if key[1] == avatar_id]
        for key in keys_to_remove:
            _avatar_cache.pop(key, None)
        return
    _avatar_cache.clear()


def avatar_materials_payload() -> dict:
    avatar_options = list_available_avatar_ids()
    existing_ids = [item["id"] for item in avatar_options]
    if app_config is not None:
        current_avatar_id = getattr(app_config.avatar.wav2lip, "avatar_id", getattr(opt, "avatar_id", "avatar_1"))
    else:
        current_avatar_id = getattr(opt, "avatar_id", "avatar_1")
    return {
        "avatar_options": avatar_options,
        "current_avatar_id": str(current_avatar_id or "avatar_1"),
        "next_avatar_id": _next_avatar_id(existing_ids),
    }


def _avatar_generation_job_payload(job_id: str) -> dict:
    job = _avatar_generation_jobs.get(job_id)
    if job is None:
        raise KeyError(job_id)
    payload = {key: value for key, value in job.items() if not key.startswith("_")}
    payload["log_tail"] = list(job.get("log_tail", []))
    return payload


def _create_avatar_generation_job(avatar_id: str, temp_video_path: Path) -> str:
    job_id = f"{avatar_id}-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    now = time.time()
    _avatar_generation_jobs[job_id] = {
        "job_id": job_id,
        "avatar_id": avatar_id,
        "status": "queued",
        "phase": "等待启动",
        "progress": 0,
        "message": "已接收素材生成请求",
        "log_tail": [],
        "created_at": now,
        "updated_at": now,
        "_temp_video_path": temp_video_path,
        "_target_dir": _avatar_target_path(avatar_id),
        "_task": None,
    }
    return job_id


def _update_avatar_generation_job(job_id: str, **fields: Any) -> None:
    job = _avatar_generation_jobs.get(job_id)
    if job is None:
        return
    job.update(fields)
    job["updated_at"] = time.time()


def _append_avatar_generation_log(job_id: str, text: str) -> None:
    job = _avatar_generation_jobs.get(job_id)
    if job is None:
        return
    lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
    if not lines:
        return
    log_tail = job.setdefault("log_tail", [])
    for line in lines:
        log_tail.append(line)
        if len(log_tail) > _AVATAR_JOB_LOG_LIMIT:
            del log_tail[:-_AVATAR_JOB_LOG_LIMIT]
        progress_match = _AVATAR_JOB_PROGRESS_PATTERN.search(line)
        if progress_match:
            try:
                progress = max(0, min(100, int(progress_match.group(1))))
            except ValueError:
                progress = None
            if progress is not None:
                job["progress"] = min(progress, 99)
                job["phase"] = "生成中"
                job["message"] = line
        else:
            lower_line = line.lower()
            if "reading images" in lower_line:
                job["phase"] = "读取图片"
            elif "prepared " in lower_line and " full frames" in lower_line:
                job["phase"] = "准备检测"
            elif "face detect" in lower_line:
                job["phase"] = "人脸检测"
            elif "writing face crops" in lower_line:
                job["phase"] = "写入 face_imgs"
            elif "face crop" in lower_line:
                job["phase"] = "写入 face_imgs"
            elif "recovering from oom error" in lower_line:
                job["phase"] = "显存不足，降低 batch"
            elif "using " in lower_line and "for inference" in lower_line:
                job["phase"] = "初始化模型"
            elif "traceback" in lower_line:
                job["phase"] = "生成失败"
            else:
                job["message"] = line
    job["updated_at"] = time.time()


def _consume_avatar_generation_output(job_id: str, buffer: str) -> str:
    if not buffer:
        return buffer
    normalized = buffer.replace("\r", "\n")
    if "\n" not in normalized:
        return buffer[-4096:]
    parts = normalized.split("\n")
    for line in parts[:-1]:
        _append_avatar_generation_log(job_id, line)
    return parts[-1]


async def _run_avatar_generation_job(job_id: str) -> None:
    job = _avatar_generation_jobs.get(job_id)
    if job is None:
        return

    avatar_id = str(job["avatar_id"])
    target_dir: Path = job["_target_dir"]
    temp_video_path: Path = job["_temp_video_path"]
    python_exe = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    script_path = REPO_ROOT / "wav2lip" / "genavatar.py"
    process = None

    try:
        _update_avatar_generation_job(
            job_id,
            status="running",
            phase="启动生成器",
            progress=2,
            message="正在启动素材生成任务",
        )
        process = await asyncio.create_subprocess_exec(
            str(python_exe),
            str(script_path),
            "--video_path",
            str(temp_video_path),
            "--img_size",
            "256",
            "--face_det_batch_size",
            "4",
            "--avatar_id",
            avatar_id,
            cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        buffer = ""
        while True:
            chunk = await process.stdout.read(4096) if process.stdout is not None else b""
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="ignore")
            buffer = _consume_avatar_generation_output(job_id, buffer)
        if buffer.strip():
            _append_avatar_generation_log(job_id, buffer)
        returncode = await process.wait()
        if returncode != 0:
            log_tail = _avatar_generation_jobs.get(job_id, {}).get("log_tail", [])
            tail_text = "\n".join(log_tail[-20:]).strip()
            message = tail_text or f"Avatar generation failed with code {returncode}"
            raise RuntimeError(message)

        _invalidate_avatar_cache(avatar_id)
        _update_avatar_generation_job(
            job_id,
            status="success",
            phase="完成",
            progress=100,
            message="素材生成完成",
        )
    except Exception as exc:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        _invalidate_avatar_cache(avatar_id)
        _update_avatar_generation_job(
            job_id,
            status="error",
            phase="失败",
            progress=100,
            message=str(exc),
            error=str(exc),
        )
        logger.exception("avatar generation job failed: job_id=%s avatar_id=%s", job_id, avatar_id)
    finally:
        if temp_video_path.exists():
            try:
                temp_video_path.unlink()
            except OSError:
                pass
        if process is not None and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass


def _encode_thumbnail(image_path: str, max_size: int = 320) -> tuple[bytes, str]:
    source = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if source is None:
        raise FileNotFoundError(image_path)

    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    elif source.ndim == 3 and source.shape[2] == 4:
        source = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)

    height, width = source.shape[:2]
    if height > 0 and width > 0:
        scale = min(1.0, float(max_size) / float(max(height, width)))
        if scale < 1.0:
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            source = cv2.resize(source, (new_width, new_height), interpolation=cv2.INTER_AREA)

    suffix = Path(image_path).suffix.lower()
    if suffix == ".png":
        success, encoded = cv2.imencode(".png", source)
        content_type = "image/png"
    else:
        success, encoded = cv2.imencode(".jpg", source, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        content_type = "image/jpeg"
    if not success:
        raise RuntimeError(f"Failed to encode thumbnail for {image_path}")
    return encoded.tobytes(), content_type


def get_current_tts_model_description() -> str:
    if app_config is None:
        return "k2-fsa/sherpa-onnx-vits-zh-ll"
    engine = getattr(app_config.providers, "tts", "vits_zh")
    meta = get_tts_engine_meta(engine)
    return str(meta.get("desc") or meta.get("label") or engine)


def list_available_avatar_ids() -> list[dict]:
    avatar_root = _resolve_avatar_root()
    silence_gate_map = {}
    if app_config is not None:
        silence_gate_map = getattr(app_config.avatar.wav2lip, "silence_gate_by_avatar", {}) or {}
    items: list[dict] = []
    if not avatar_root or not os.path.isdir(avatar_root):
        return items

    for entry in sorted(os.scandir(avatar_root), key=lambda item: item.name.lower()):
        if not entry.is_dir():
            continue
        items.append({
            "id": entry.name,
            "label": entry.name,
            "thumbnail_url": f"/api/avatar-thumbnail/{quote(entry.name, safe='')}",
            "silence_gate_enabled": bool(
                silence_gate_map.get(
                    entry.name,
                    getattr(app_config.avatar.wav2lip, "silence_gate_enabled", False) if app_config else False,
                )
            ),
        })
    return items


def _resolve_avatar_root() -> str:
    avatar_root = ""
    if app_config is not None:
        avatar_root = resolve_project_path(app_config, getattr(app_config.avatar.wav2lip, "avatar_dir", ""))
    if not avatar_root:
        avatar_root = getattr(opt, "AVATAR_DIR", "")
    return avatar_root


def get_avatar_data(avatar_id: str) -> tuple[Any, Any, Any, Any]:
    avatar_root = _resolve_avatar_root()
    cache_key = (avatar_root, avatar_id)
    cached_avatar = _avatar_cache.get(cache_key)
    if cached_avatar is not None:
        return cached_avatar

    from .avatar.wav2lip_real import load_avatar

    if not avatar_root:
        avatar_root = "./data/avatars"
    avatar_data = load_avatar(avatar_id, avatar_root)
    _avatar_cache[cache_key] = avatar_data
    return avatar_data


def warmup_runtime_resources() -> None:
    if app_config is None or opt is None:
        return

    avatar_id = getattr(app_config.avatar.wav2lip, "avatar_id", getattr(opt, "avatar_id", "avatar_1"))
    try:
        get_avatar_data(avatar_id)
        logger.info("Avatar assets warmed up: avatar_id=%s", avatar_id)
    except Exception as exc:
        logger.warning("Avatar assets warmup skipped: avatar_id=%s error=%s", avatar_id, exc)

    if opt.tts in SHERPA_TTS_ENGINES:
        tts_warmup = SherpaOnnxVitsTTS(opt, None)
        tts_warmup.warmup()
        logger.info("TTS model warmed up: engine=%s", opt.tts)
def randN(length: int) -> int:
    minimum = pow(10, length - 1)
    maximum = pow(10, length)
    return random.randint(minimum, maximum - 1)


def json_response(payload: dict, status: int = 200) -> web.Response:
    return web.Response(
        status=status,
        content_type="application/json",
        text=json.dumps(payload, ensure_ascii=False),
    )


def _get_local_non_loopback_ipv4_addresses() -> list[str]:
    preferred: list[str] = []
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = sockaddr[0]
            ip = ipaddress.ip_address(address)
            if ip.is_loopback or ip.is_link_local:
                continue
            if address not in preferred:
                preferred.append(address)
    except socket.gaierror:
        pass
    return preferred


def patched_get_host_addresses(use_ipv4: bool, use_ipv6: bool) -> list[str]:
    if use_ipv4 and _preferred_ice_addresses:
        return list(_preferred_ice_addresses)
    return _ORIGINAL_GET_HOST_ADDRESSES(use_ipv4, use_ipv6)


aioice_ice.get_host_addresses = patched_get_host_addresses


def get_session_real(sessionid: int) -> BaseReal:
    if sessionid <= 0:
        raise InvalidSessionError("invalid sessionid: 0, start WebRTC first")

    nerfreal = nerfreals.get(sessionid)
    if nerfreal is None:
        raise InvalidSessionError(f"invalid sessionid: {sessionid}, session not ready or already closed")
    return nerfreal


def _extract_offer_udp_host_ipv4_candidates(sdp: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in sdp.splitlines():
        line = raw_line.strip()
        if not line.startswith("a=candidate:"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        transport = parts[2].lower()
        address = parts[4].strip()
        candidate_type = parts[7].lower()
        if transport != "udp" or candidate_type != "host":
            continue
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if ip.version != 4 or ip.is_link_local:
            continue
        if address not in candidates:
            candidates.append(address)
    return candidates


def resolve_preferred_ice_addresses(request: web.Request, offer_sdp: str = "") -> list[str]:
    host = request.host.split(":")[0].strip().lower()

    if host in {"127.0.0.1", "localhost"}:
        offer_candidates = _extract_offer_udp_host_ipv4_candidates(offer_sdp)
        if "127.0.0.1" in offer_candidates:
            return ["127.0.0.1"]
        non_loopback_offer_candidates = [addr for addr in offer_candidates if addr != "127.0.0.1"]
        if non_loopback_offer_candidates:
            logger.info(
                "Using localhost remote-offer IPv4 candidate(s) as preferred ICE addresses: %s",
                non_loopback_offer_candidates,
            )
            return non_loopback_offer_candidates
        return ["127.0.0.1"]

    preferred = _get_local_non_loopback_ipv4_addresses()

    if host and host not in {"0.0.0.0"}:
        try:
            ip = ipaddress.ip_address(host)
            if ip.version == 4 and not ip.is_loopback and not ip.is_link_local and host not in preferred:
                preferred.insert(0, host)
        except ValueError:
            pass

    if preferred:
        return preferred
    if host in {"127.0.0.1", "localhost"}:
        return ["127.0.0.1"]
    return _ORIGINAL_GET_HOST_ADDRESSES(use_ipv4=True, use_ipv6=False)


def should_force_loopback_mdns(request: web.Request) -> bool:
    host = request.host.split(":")[0].strip().lower()
    return host in {"127.0.0.1", "localhost"}


def rewrite_local_mdns_candidates(sdp: str) -> str:
    line_ending = "\r\n" if "\r\n" in sdp else "\n"
    rewritten_mdns = re.sub(r"(?<=\s)([0-9a-f-]+\.local)(?=\s)", "127.0.0.1", sdp, flags=re.IGNORECASE)

    rewritten_lines: list[str] = []
    dropped_candidates = 0
    kept_loopback_candidates = 0

    for raw_line in rewritten_mdns.splitlines():
        line = raw_line.strip()
        if not line:
            rewritten_lines.append(raw_line)
            continue

        if line.startswith("c=IN IP4 "):
            rewritten_lines.append("c=IN IP4 127.0.0.1")
            continue

        if line.startswith("a=candidate:"):
            parts = line.split()
            if len(parts) >= 8:
                transport = parts[2].lower()
                address = parts[4].lower()
                candidate_type = parts[7].lower()
                if transport == "udp" and candidate_type == "host" and address == "127.0.0.1":
                    kept_loopback_candidates += 1
                    rewritten_lines.append(" ".join(parts))
                    continue
                dropped_candidates += 1
                continue
            rewritten_lines.append(line)
            continue

        rewritten_lines.append(raw_line)

    if kept_loopback_candidates <= 0:
        logger.warning(
            "No loopback UDP host ICE candidate found in localhost offer; keeping rewritten original SDP candidates"
        )
        return rewritten_mdns

    if dropped_candidates:
        logger.info("Filtered %s non-loopback ICE candidate(s) from localhost offer", dropped_candidates)

    return line_ending.join(rewritten_lines) + line_ending


def build_nerfreal(sessionid: int) -> BaseReal:
    opt.sessionid = sessionid
    from .avatar.wav2lip_real import LipReal

    avatar_data = get_avatar_data(opt.avatar_id)
    return LipReal(opt, model, avatar_data)


async def offer(request: web.Request) -> web.Response:
    global _preferred_ice_addresses
    params = await request.json()
    raw_sdp = params["sdp"]
    if should_force_loopback_mdns(request):
        rewritten_sdp = rewrite_local_mdns_candidates(raw_sdp)
        if rewritten_sdp != raw_sdp:
            logger.info("Rewrote localhost mDNS ICE candidates to 127.0.0.1 for local debugging")
        raw_sdp = rewritten_sdp
    remote_offer = RTCSessionDescription(sdp=raw_sdp, type=params["type"])
    _preferred_ice_addresses = resolve_preferred_ice_addresses(request, raw_sdp)
    logger.info("Preferred ICE addresses for %s -> %s", request.host, _preferred_ice_addresses)
    offer_candidates = [
        line.strip()
        for line in raw_sdp.splitlines()
        if line.startswith("a=candidate:") or line.startswith("c=")
    ]
    logger.info("Remote offer candidates: %s", offer_candidates)

    sessionid = randN(6)
    nerfreals[sessionid] = None
    logger.info("sessionid=%d, session num=%d", sessionid, len(nerfreals))
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed"}:
            nerfreal.deactivate_session()
            await pc.close()
            pcs.discard(pc)
            nerfreals.pop(sessionid, None)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info("ICE connection state is %s", pc.iceConnectionState)

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        logger.info("ICE gathering state is %s", pc.iceGatheringState)

    player = HumanPlayer(nerfreals[sessionid])
    pc.addTrack(player.audio)
    pc.addTrack(player.video)

    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = [codec for codec in capabilities.codecs if codec.name in {"H264", "VP8", "rtx"}]
    pc.getTransceivers()[1].setCodecPreferences(preferences)

    await pc.setRemoteDescription(remote_offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    answer_candidates = [
        line.strip()
        for line in pc.localDescription.sdp.splitlines()
        if line.startswith("a=candidate:") or line.startswith("c=")
    ]
    logger.info(
        "Created answer for sessionid=%d, iceGatheringState=%s, iceConnectionState=%s, candidates=%s",
        sessionid,
        pc.iceGatheringState,
        pc.iceConnectionState,
        answer_candidates,
    )

    return json_response(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "sessionid": sessionid,
        }
    )


async def human(request: web.Request) -> web.Response:
    try:
        params = await request.json()
        sessionid = int(params.get("sessionid", 0))
        nerfreal = get_session_real(sessionid)

        if params.get("interrupt"):
            nerfreal.invalidate_pending_responses()
            nerfreal.flush_talk()

        if not params.get("skip_dialog_log") and hasattr(nerfreal, "append_dialog"):
            nerfreal.append_dialog(
                "user",
                params.get("text", ""),
                str(params.get("type", "chat")),
                params.get("meta", {}) if isinstance(params.get("meta", {}), dict) else {},
            )

        if params["type"] == "echo":
            nerfreal.put_msg_txt(params["text"])
        elif params["type"] == "chat":
            continuous_dialogue = bool(params.get("continuous_dialogue", False))
            dialog_start_ts = time.perf_counter()
            response_token = nerfreal.begin_response()
            if response_token < 0:
                raise InvalidSessionError(f"invalid sessionid: {sessionid}, session already closed")
            asyncio.get_event_loop().run_in_executor(
                None,
                safe_llm_response,
                params["text"],
                nerfreal,
                continuous_dialogue,
                dialog_start_ts,
                response_token,
            )

        return json_response({"code": 0, "msg": "ok"})
    except InvalidSessionError as exc:
        return json_response({"code": -1, "msg": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("human route failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


async def interrupt_talk(request: web.Request) -> web.Response:
    try:
        params = await request.json()
        sessionid = int(params.get("sessionid", 0))
        get_session_real(sessionid).flush_talk()
        return json_response({"code": 0, "msg": "ok"})
    except InvalidSessionError as exc:
        return json_response({"code": -1, "msg": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("interrupt_talk route failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


async def humanaudio(request: web.Request) -> web.Response:
    try:
        form = await request.post()
        sessionid = int(form.get("sessionid", 0))
        fileobj = form["file"]
        filebytes = fileobj.file.read()
        get_session_real(sessionid).put_audio_file(filebytes)
        return json_response({"code": 0, "msg": "ok"})
    except InvalidSessionError as exc:
        return json_response({"code": -1, "msg": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("humanaudio route failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


async def transcribe_audio(request: web.Request) -> web.Response:
    try:
        if speech_recognizer is None:
            raise RuntimeError("Paraformer ASR is not initialized")

        start = time.perf_counter()
        form = await request.post()
        if "file" not in form:
            raise RuntimeError("missing form field: file")

        fileobj = form["file"]
        filebytes = fileobj.file.read()
        logger.info(
            "ASR upload received filename=%s size=%s bytes",
            getattr(fileobj, "filename", ""),
            len(filebytes),
        )
        decode_start = time.perf_counter()
        try:
            pcm_s16le, sample_rate = await asyncio.get_event_loop().run_in_executor(None, decode_audio_to_pcm16, filebytes)
        except Exception as exc:
            logger.warning("ASR decode failed filename=%s: %s", getattr(fileobj, "filename", ""), exc)
            return json_response({"code": -1, "msg": f"audio decode failed: {exc}"}, status=400)
        logger.info(
            "ASR decode done in %.3fs, sample_rate=%s, pcm_samples=%s",
            time.perf_counter() - decode_start,
            sample_rate,
            len(pcm_s16le),
        )
        infer_start = time.perf_counter()
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            speech_recognizer.transcribe_pcm16,
            pcm_s16le,
            sample_rate,
        )
        logger.info(
            "ASR transcribe done in %.3fs, sample_rate=%s, pcm_samples=%s, text=%s",
            time.perf_counter() - infer_start,
            sample_rate,
            len(pcm_s16le),
            result.get("text", ""),
        )
        logger.info("ASR total elapsed %.3fs", time.perf_counter() - start)
        return json_response({"code": 0, "msg": "ok", "data": result})
    except Exception as exc:
        logger.exception("transcribe_audio route failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


async def set_silence_gate(request: web.Request) -> web.Response:
    try:
        params = await request.json()
        sessionid = int(params.get("sessionid", 0))
        enabled = bool(params.get("enabled", False))
        nerfreal = get_session_real(sessionid)
        nerfreal.set_silence_gate(enabled)
        return json_response({"code": 0, "msg": "ok", "data": {"enabled": enabled}})
    except InvalidSessionError as exc:
        return json_response({"code": -1, "msg": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("set_silence_gate route failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


async def record(request: web.Request) -> web.Response:
    try:
        params = await request.json()
        sessionid = int(params.get("sessionid", 0))
        nerfreal = get_session_real(sessionid)
        if params["type"] == "start_record":
            nerfreal.start_recording()
        elif params["type"] == "end_record":
            nerfreal.stop_recording()
        return json_response({"code": 0, "msg": "ok"})
    except InvalidSessionError as exc:
        return json_response({"code": -1, "msg": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("record route failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


async def is_speaking(request: web.Request) -> web.Response:
    try:
        params = await request.json()
        sessionid = int(params.get("sessionid", 0))
        return json_response({"code": 0, "data": get_session_real(sessionid).is_speaking()})
    except InvalidSessionError as exc:
        return json_response({"code": -1, "msg": str(exc)}, status=400)


async def add_dialog_entry(request: web.Request) -> web.Response:
    try:
        params = await request.json()
        sessionid = int(params.get("sessionid", 0))
        nerfreal = get_session_real(sessionid)
        if hasattr(nerfreal, "append_dialog"):
            nerfreal.append_dialog(
                str(params.get("role", "user")),
                str(params.get("text", "")),
                str(params.get("source", "")),
                params.get("meta", {}) if isinstance(params.get("meta", {}), dict) else {},
            )
        return json_response({"code": 0, "msg": "ok"})
    except InvalidSessionError as exc:
        return json_response({"code": -1, "msg": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("add_dialog_entry route failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


async def get_dialog_history(request: web.Request) -> web.Response:
    try:
        sessionid = int(request.query.get("sessionid", "0"))
        limit = int(request.query.get("limit", "50"))
        nerfreal = get_session_real(sessionid)
        history = nerfreal.get_dialog_history(limit) if hasattr(nerfreal, "get_dialog_history") else []
        return json_response({"code": 0, "msg": "ok", "data": history})
    except InvalidSessionError as exc:
        return json_response({"code": -1, "msg": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("get_dialog_history route failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


async def clear_dialog_history(request: web.Request) -> web.Response:
    try:
        params = await request.json()
        sessionid = int(params.get("sessionid", 0))
        nerfreal = get_session_real(sessionid)
        if hasattr(nerfreal, "clear_dialog_history"):
            nerfreal.clear_dialog_history()
        return json_response({"code": 0, "msg": "ok"})
    except InvalidSessionError as exc:
        return json_response({"code": -1, "msg": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("clear_dialog_history route failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


def runtime_config_payload() -> dict:
    if app_config is None:
        raise RuntimeError("app config is not initialized")

    payload = asdict(app_config)
    payload.pop("project_root", None)
    payload["meta"] = {
        "config_path": getattr(opt, "CONFIG_PATH", ""),
        "settings_db_path": str(get_settings_db_path(REPO_ROOT)),
        "tts_voice_options": get_tts_voice_options(app_config.providers.tts),
        "tts_voice_options_by_engine": TTS_VOICE_OPTIONS_BY_ENGINE,
        "tts_engine_options": TTS_ENGINE_OPTIONS,
        "llm_mode_options": LLM_MODE_OPTIONS,
        "avatar_options": list_available_avatar_ids(),
        "model_descriptions": {
            "asr": "Paraformer-large + punctuation restoration",
            "tts": get_current_tts_model_description(),
            "llm": get_current_llm_model_description(),
        },
        "current_tts_engine": getattr(app_config.providers, "tts", "vits_zh"),
        "current_llm_mode": normalize_llm_mode(getattr(app_config.providers, "llm", "openai_chat")),
        "current_avatar_id": getattr(app_config.avatar.wav2lip, "avatar_id", getattr(opt, "avatar_id", "avatar_1")),
        "current_silence_gate_enabled": bool(
            getattr(app_config.avatar.wav2lip, "silence_gate_by_avatar", {}).get(
                getattr(app_config.avatar.wav2lip, "avatar_id", getattr(opt, "avatar_id", "avatar_1")),
                getattr(app_config.avatar.wav2lip, "silence_gate_enabled", False),
            )
        ),
    }
    return payload


def apply_runtime_settings(payload: dict) -> None:
    global speech_recognizer
    if app_config is None or opt is None:
        raise RuntimeError("runtime is not initialized")

    providers_config = payload.get("providers", {})
    runtime_config = payload.get("runtime", {})
    tts_payload = payload.get("tts", {})
    edge_config = tts_payload.get("edgetts", {})
    llm_payload = payload.get("llm", {})
    asr_config = payload.get("asr", {}).get("paraformer", {})
    avatar_id = str(payload.get("avatar_id", getattr(app_config.avatar.wav2lip, "avatar_id", "avatar_1"))).strip()
    silence_gate_by_avatar = dict(getattr(app_config.avatar.wav2lip, "silence_gate_by_avatar", {}) or {})
    if "avatar_silence_gate_by_avatar" in payload and isinstance(payload.get("avatar_silence_gate_by_avatar"), dict):
        incoming_map = payload.get("avatar_silence_gate_by_avatar", {})
        silence_gate_by_avatar = {str(key): bool(value) for key, value in incoming_map.items()}
    legacy_silence_gate = bool(payload.get("silence_gate_enabled", getattr(app_config.avatar.wav2lip, "silence_gate_enabled", False)))
    tts_engine_changed = False
    avatar_changed = False
    selected_tts = normalize_tts_engine(str(providers_config.get("tts", app_config.providers.tts)) if "tts" in providers_config else app_config.providers.tts)
    selected_llm = normalize_llm_mode(str(providers_config.get("llm", app_config.providers.llm)) if "llm" in providers_config else app_config.providers.llm)

    if "tts" in providers_config:
        if selected_tts not in SUPPORTED_TTS_ENGINES:
            raise ValueError(f"Unsupported TTS provider: {selected_tts}")
        tts_engine_changed = selected_tts != app_config.providers.tts
        app_config.providers.tts = selected_tts
        opt.tts = selected_tts

    if "llm" in providers_config:
        if selected_llm not in SUPPORTED_LLM_MODES:
            raise ValueError(f"Unsupported LLM provider: {selected_llm}")
        app_config.providers.llm = selected_llm

    if "log_level" in runtime_config:
        log_level = str(runtime_config["log_level"]).strip().upper() or "INFO"
        app_config.runtime.log_level = log_level
        configure_logging(log_level)
        logger.info("Logging level updated to %s", log_level)

    if avatar_id:
        available_avatar_ids = {item["id"] for item in list_available_avatar_ids()}
        if available_avatar_ids and avatar_id not in available_avatar_ids:
            raise ValueError(f"Unsupported avatar id: {avatar_id}")
        avatar_changed = avatar_id != getattr(app_config.avatar.wav2lip, "avatar_id", "")
        app_config.avatar.wav2lip.avatar_id = avatar_id
        opt.avatar_id = avatar_id

    if not silence_gate_by_avatar and payload.get("silence_gate_enabled") is not None:
        silence_gate_by_avatar[avatar_id] = legacy_silence_gate
    app_config.avatar.wav2lip.silence_gate_by_avatar = silence_gate_by_avatar
    app_config.avatar.wav2lip.silence_gate_enabled = bool(silence_gate_by_avatar.get(avatar_id, legacy_silence_gate))
    opt.SILENCE_GATE_ENABLED = bool(silence_gate_by_avatar.get(avatar_id, legacy_silence_gate))

    sherpa_config = tts_payload.get(selected_tts, {}) if selected_tts in SHERPA_TTS_ENGINES else {}
    if sherpa_config:
        if "speaker_id" in sherpa_config:
            speaker_id = int(sherpa_config["speaker_id"])
            getattr(app_config.tts, selected_tts).speaker_id = speaker_id
            opt.TTS_SPEAKER_ID = speaker_id
        if "speed" in sherpa_config:
            speed = float(sherpa_config["speed"])
            getattr(app_config.tts, selected_tts).speed = speed
            opt.TTS_SPEED = speed
        if "num_threads" in sherpa_config:
            num_threads = int(sherpa_config["num_threads"])
            getattr(app_config.tts, selected_tts).num_threads = num_threads
            opt.TTS_NUM_THREADS = num_threads
            SherpaOnnxVitsTTS.reset_shared_tts()
        if "provider" in sherpa_config:
            provider = str(sherpa_config["provider"])
            getattr(app_config.tts, selected_tts).provider = provider
            opt.TTS_PROVIDER = provider
            SherpaOnnxVitsTTS.reset_shared_tts()

    if edge_config:
        if "voice_name" in edge_config:
            app_config.tts.edgetts.voice_name = str(edge_config["voice_name"])
        if "speed" in edge_config:
            app_config.tts.edgetts.speed = str(edge_config["speed"])

    llm_config = llm_payload.get(selected_llm, {})
    target_llm_config = getattr(app_config.llm, selected_llm, None)
    if target_llm_config is not None and llm_config:
        for field_name in ("base_url", "api_key", "model", "system_prompt_zh", "system_prompt_en"):
            if field_name in llm_config:
                setattr(target_llm_config, field_name, str(llm_config[field_name]))
        for field_name in ("timeout_s",):
            if field_name in llm_config and hasattr(target_llm_config, field_name):
                setattr(target_llm_config, field_name, int(llm_config[field_name]))
    if target_llm_config is not None:
        configure_llm(target_llm_config, selected_llm)

    if asr_config:
        should_reload_asr = False
        hotwords_changed = False
        phonetic_replacements_changed = False
        segment_settings_changed = False
        if "batch_size_s" in asr_config:
            batch_size_s = int(asr_config["batch_size_s"])
            app_config.asr.paraformer.batch_size_s = batch_size_s
            opt.ASR_BATCH_SIZE_S = batch_size_s
            if speech_recognizer is not None:
                speech_recognizer.batch_size_s = batch_size_s
        if "device" in asr_config:
            device = str(asr_config["device"])
            if device != app_config.asr.paraformer.device:
                app_config.asr.paraformer.device = device
                opt.ASR_DEVICE = device
                should_reload_asr = True
        if "model_dir" in asr_config:
            model_dir = str(asr_config["model_dir"])
            if model_dir != app_config.asr.paraformer.model_dir:
                app_config.asr.paraformer.model_dir = model_dir
                opt.ASR_MODEL_DIR = model_dir
                should_reload_asr = True
        if "punc_model_dir" in asr_config:
            punc_model_dir = str(asr_config["punc_model_dir"])
            if punc_model_dir != app_config.asr.paraformer.punc_model_dir:
                app_config.asr.paraformer.punc_model_dir = punc_model_dir
                opt.ASR_PUNC_MODEL_DIR = punc_model_dir
                should_reload_asr = True
        if "hotwords" in asr_config:
            hotwords = str(asr_config["hotwords"])
            if hotwords != app_config.asr.paraformer.hotwords:
                app_config.asr.paraformer.hotwords = hotwords
                opt.ASR_HOTWORDS = hotwords
                hotwords_changed = True
        if "phonetic_replacements" in asr_config:
            phonetic_replacements = str(asr_config["phonetic_replacements"])
            if phonetic_replacements != app_config.asr.paraformer.phonetic_replacements:
                app_config.asr.paraformer.phonetic_replacements = phonetic_replacements
                opt.ASR_PHONETIC_REPLACEMENTS = phonetic_replacements
                phonetic_replacements_changed = True
        for field_name in (
            "segment_by_silence",
            "segment_min_silence_ms",
            "segment_min_speech_ms",
            "segment_padding_ms",
            "segment_max_duration_s",
            "segment_split_overlap_ms",
        ):
            if field_name in asr_config:
                new_value = asr_config[field_name]
                current_value = getattr(app_config.asr.paraformer, field_name)
                if field_name == "segment_by_silence":
                    new_value = bool(new_value)
                elif field_name == "segment_max_duration_s":
                    new_value = float(new_value)
                else:
                    new_value = int(new_value)
                if new_value != current_value:
                    setattr(app_config.asr.paraformer, field_name, new_value)
                    segment_settings_changed = True
        if should_reload_asr:
            speech_recognizer = ParaformerProvider(
                model_dir=opt.ASR_MODEL_DIR,
                device=opt.ASR_DEVICE,
                batch_size_s=opt.ASR_BATCH_SIZE_S,
                hotwords=app_config.asr.paraformer.hotwords,
                phonetic_replacements=app_config.asr.paraformer.phonetic_replacements,
                punc_model_dir=opt.ASR_PUNC_MODEL_DIR,
                temp_dir=opt.RUNTIME_TEMP_DIR,
                segment_by_silence=app_config.asr.paraformer.segment_by_silence,
                segment_min_silence_ms=app_config.asr.paraformer.segment_min_silence_ms,
                segment_min_speech_ms=app_config.asr.paraformer.segment_min_speech_ms,
                segment_padding_ms=app_config.asr.paraformer.segment_padding_ms,
                segment_max_duration_s=app_config.asr.paraformer.segment_max_duration_s,
                segment_split_overlap_ms=app_config.asr.paraformer.segment_split_overlap_ms,
            )
            speech_recognizer.warmup()
        elif speech_recognizer is not None and (
            hotwords_changed or phonetic_replacements_changed or segment_settings_changed
        ):
            if segment_settings_changed:
                speech_recognizer.segment_by_silence = app_config.asr.paraformer.segment_by_silence
                speech_recognizer.segment_min_silence_ms = app_config.asr.paraformer.segment_min_silence_ms
                speech_recognizer.segment_min_speech_ms = app_config.asr.paraformer.segment_min_speech_ms
                speech_recognizer.segment_padding_ms = app_config.asr.paraformer.segment_padding_ms
                speech_recognizer.segment_max_duration_s = app_config.asr.paraformer.segment_max_duration_s
                speech_recognizer.segment_split_overlap_ms = app_config.asr.paraformer.segment_split_overlap_ms
            speech_recognizer.update_asr_rules(
                hotwords=app_config.asr.paraformer.hotwords,
                phonetic_replacements=app_config.asr.paraformer.phonetic_replacements,
            )

    apply_config_to_opt(opt, app_config)

    if tts_engine_changed:
        for nerfreal in list(nerfreals.values()):
            if nerfreal is None or not hasattr(nerfreal, "reload_tts"):
                continue
            nerfreal.reload_tts()
    else:
        for nerfreal in list(nerfreals.values()):
            if nerfreal is None or not hasattr(nerfreal, "tts"):
                continue
            if hasattr(nerfreal.tts, "speaker_id"):
                nerfreal.tts.speaker_id = getattr(opt, "TTS_SPEAKER_ID", 0)
            if hasattr(nerfreal.tts, "speed"):
                nerfreal.tts.speed = getattr(opt, "TTS_SPEED", 1.0)
            if hasattr(nerfreal.tts, "num_threads"):
                nerfreal.tts.num_threads = getattr(opt, "TTS_NUM_THREADS", 1)
            if hasattr(nerfreal.tts, "provider"):
                nerfreal.tts.provider = getattr(opt, "TTS_PROVIDER", "")
            if hasattr(nerfreal.tts, "_tts") and nerfreal.tts._tts is None:
                continue
            if hasattr(nerfreal.tts, "_tts"):
                nerfreal.tts._tts = None

    if avatar_changed:
        for nerfreal in list(nerfreals.values()):
            if nerfreal is None or not hasattr(nerfreal, "reload_avatar"):
                continue
            nerfreal.reload_avatar(avatar_id)
    for nerfreal in list(nerfreals.values()):
        if nerfreal is None or not hasattr(nerfreal, "set_silence_gate"):
            continue
        nerfreal.set_silence_gate(bool(silence_gate_by_avatar.get(avatar_id, legacy_silence_gate)))

    save_app_config(app_config, getattr(opt, "CONFIG_PATH", None))


async def get_runtime_config(request: web.Request) -> web.Response:
    try:
        return json_response({"code": 0, "msg": "ok", "data": runtime_config_payload()})
    except Exception as exc:
        logger.exception("get_runtime_config failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


async def get_avatar_thumbnail(request: web.Request) -> web.Response:
    try:
        avatar_id = str(request.match_info.get("avatar_id", "")).strip()
        if not avatar_id:
            return json_response({"code": -1, "msg": "Missing avatar id"}, status=400)

        avatar_root = ""
        if app_config is not None:
            avatar_root = resolve_project_path(app_config, getattr(app_config.avatar.wav2lip, "avatar_dir", ""))
        if not avatar_root:
            avatar_root = getattr(opt, "AVATAR_DIR", "")
        if not avatar_root or not os.path.isdir(avatar_root):
            return json_response({"code": -1, "msg": "Avatar directory is unavailable"}, status=404)

        thumbnail_path = _resolve_avatar_thumbnail_path(avatar_root, avatar_id)
        if not thumbnail_path or not os.path.isfile(thumbnail_path):
            return json_response({"code": -1, "msg": "Thumbnail not found"}, status=404)

        image_bytes, content_type = _encode_thumbnail(thumbnail_path)
        return web.Response(
            body=image_bytes,
            content_type=content_type,
            headers={
                "Cache-Control": "public, max-age=3600",
            },
        )
    except Exception as exc:
        logger.exception("get_avatar_thumbnail failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


async def get_avatar_materials(request: web.Request) -> web.Response:
    try:
        return json_response({"code": 0, "msg": "ok", "data": avatar_materials_payload()})
    except Exception as exc:
        logger.exception("get_avatar_materials failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


async def create_avatar_material(request: web.Request) -> web.Response:
    target_dir: Path | None = None
    temp_video_path: Path | None = None
    try:
        if app_config is None or opt is None:
            raise RuntimeError("Runtime is not initialized")

        form = await request.post()
        avatar_id = _normalize_avatar_id(form.get("avatar_id", ""))
        if not avatar_id:
            raise ValueError("Missing avatar id")
        if not _is_valid_avatar_id(avatar_id):
            raise ValueError("Invalid avatar id. Use letters, numbers, '_' or '-' only.")

        avatar_root = _avatar_root_path()
        if not str(avatar_root):
            raise RuntimeError("Avatar directory is unavailable")
        avatar_root.mkdir(parents=True, exist_ok=True)

        existing_ids = set(_list_existing_avatar_ids())
        if avatar_id in existing_ids:
            raise ValueError(f"Avatar already exists: {avatar_id}")

        fileobj = form.get("file")
        if fileobj is None:
            raise ValueError("Missing mp4 file")

        filename = _normalize_avatar_id(getattr(fileobj, "filename", ""))
        if not filename.lower().endswith(".mp4"):
            raise ValueError("Only mp4 files are supported")

        temp_dir = Path(getattr(opt, "RUNTIME_TEMP_DIR", str(REPO_ROOT / "runtime" / "tmp")))
        temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=str(temp_dir)) as temp_file:
            shutil.copyfileobj(fileobj.file, temp_file)
            temp_video_path = Path(temp_file.name)

        target_dir = _avatar_target_path(avatar_id)
        python_exe = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
        if not python_exe.is_file():
            raise RuntimeError(f"Python executable not found: {python_exe}")

        job_id = _create_avatar_generation_job(avatar_id, temp_video_path)
        task = asyncio.create_task(_run_avatar_generation_job(job_id))
        _avatar_generation_jobs[job_id]["_task"] = task
        temp_video_path = None
        return json_response({"code": 0, "msg": "accepted", "data": _avatar_generation_job_payload(job_id)})
    except ValueError as exc:
        if target_dir is not None and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
            _invalidate_avatar_cache(target_dir.name)
        logger.warning("create_avatar_material validation failed: %s", exc)
        return json_response({"code": -1, "msg": str(exc)}, status=400)
    except Exception as exc:
        if target_dir is not None and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
            _invalidate_avatar_cache(target_dir.name)
        logger.exception("create_avatar_material failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)
    finally:
        if temp_video_path is not None and temp_video_path.exists():
            try:
                temp_video_path.unlink()
            except OSError:
                pass


async def get_avatar_generation_job(request: web.Request) -> web.Response:
    try:
        job_id = str(request.match_info.get("job_id", "")).strip()
        if not job_id:
            return json_response({"code": -1, "msg": "Missing job id"}, status=400)
        job = _avatar_generation_jobs.get(job_id)
        if job is None:
            return json_response({"code": -1, "msg": "Job not found"}, status=404)
        return json_response({"code": 0, "msg": "ok", "data": _avatar_generation_job_payload(job_id)})
    except Exception as exc:
        logger.exception("get_avatar_generation_job failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


async def delete_avatar_material(request: web.Request) -> web.Response:
    try:
        if app_config is None or opt is None:
            raise RuntimeError("Runtime is not initialized")
        avatar_id = _normalize_avatar_id(request.match_info.get("avatar_id", ""))
        if not avatar_id:
            return json_response({"code": -1, "msg": "Missing avatar id"}, status=400)
        if not _is_valid_avatar_id(avatar_id):
            return json_response({"code": -1, "msg": "Invalid avatar id"}, status=400)
        current_avatar_id = getattr(app_config.avatar.wav2lip, "avatar_id", getattr(opt, "avatar_id", "avatar_1"))
        if avatar_id == current_avatar_id:
            return json_response({"code": -1, "msg": "Cannot delete the current avatar"}, status=400)

        target_dir = _avatar_target_path(avatar_id)
        if not target_dir.is_dir():
            return json_response({"code": -1, "msg": "Avatar not found"}, status=404)

        shutil.rmtree(target_dir)
        _invalidate_avatar_cache(avatar_id)
        return json_response({"code": 0, "msg": "ok", "data": avatar_materials_payload()})
    except ValueError as exc:
        logger.warning("delete_avatar_material validation failed: %s", exc)
        return json_response({"code": -1, "msg": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("delete_avatar_material failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


async def update_runtime_config(request: web.Request) -> web.Response:
    try:
        params = await request.json()
        apply_runtime_settings(params)
        return json_response({"code": 0, "msg": "ok", "data": runtime_config_payload()})
    except Exception as exc:
        logger.exception("update_runtime_config failed")
        return json_response({"code": -1, "msg": str(exc)}, status=500)


def safe_llm_response(
    message: str,
    nerfreal: BaseReal,
    continuous_dialogue: bool = False,
    dialog_start_ts: float | None = None,
    response_token: int | None = None,
) -> None:
    try:
        llm_response(message, nerfreal, continuous_dialogue, dialog_start_ts, response_token)
    except Exception:
        logger.exception("llm_response failed")


async def on_shutdown(app: web.Application) -> None:
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/app.yaml", help="project config yaml")
    parser.add_argument("--fps", type=int, default=50, help="audio fps, must be 50")
    parser.add_argument("-l", type=int, default=10)
    parser.add_argument("-m", type=int, default=8)
    parser.add_argument("-r", type=int, default=10)
    parser.add_argument("--W", type=int, default=450, help="GUI width")
    parser.add_argument("--H", type=int, default=450, help="GUI height")
    parser.add_argument("--avatar_id", type=str, default=None, help="avatar id in data/avatars")
    parser.add_argument("--batch_size", type=int, default=16, help="infer batch")
    parser.add_argument("--tts", type=str, default="", help="TTS 引擎：vits_zh / vits_melo_zh_en / edgetts；留空时使用配置文件")
    parser.add_argument("--model", type=str, default="wav2lip", help="only wav2lip is supported")
    parser.add_argument("--listenport", type=int, default=8010, help="web listen port")
    return parser


def validate_supported_modes(args) -> None:
    if args.model != "wav2lip":
        raise ValueError("This build only supports --model wav2lip")
    if getattr(args, "tts", ""):
        tts_engine = normalize_tts_engine(args.tts)
        if tts_engine not in SUPPORTED_TTS_ENGINES:
            raise ValueError("This build only supports --tts vits_zh / vits_melo_zh_en / edgetts")
        args.tts = tts_engine
    args.transport = "webrtc"


def bootstrap_runtime(args):
    global app_config, opt, model, avatar, speech_recognizer

    app_config = load_app_config(args.config, project_root=REPO_ROOT)
    configure_logging(getattr(app_config.runtime, "log_level", "INFO"))
    logger.info("Logging level configured: %s", getattr(app_config.runtime, "log_level", "INFO"))
    if not getattr(args, "tts", ""):
        args.tts = app_config.providers.tts
    validate_supported_modes(args)
    opt = apply_config_to_opt(args, app_config)
    opt.transport = "webrtc"
    opt.customopt = []

    configure_llm(get_current_llm_config(), normalize_llm_mode(getattr(app_config.providers, "llm", "openai_chat")))
    speech_recognizer = ParaformerProvider(
        model_dir=opt.ASR_MODEL_DIR,
        device=opt.ASR_DEVICE,
        batch_size_s=opt.ASR_BATCH_SIZE_S,
        hotwords=app_config.asr.paraformer.hotwords,
        phonetic_replacements=app_config.asr.paraformer.phonetic_replacements,
        punc_model_dir=opt.ASR_PUNC_MODEL_DIR,
        temp_dir=opt.RUNTIME_TEMP_DIR,
        segment_by_silence=app_config.asr.paraformer.segment_by_silence,
        segment_min_silence_ms=app_config.asr.paraformer.segment_min_silence_ms,
        segment_min_speech_ms=app_config.asr.paraformer.segment_min_speech_ms,
        segment_padding_ms=app_config.asr.paraformer.segment_padding_ms,
        segment_max_duration_s=app_config.asr.paraformer.segment_max_duration_s,
        segment_split_overlap_ms=app_config.asr.paraformer.segment_split_overlap_ms,
    )
    speech_recognizer.warmup()

    from .avatar.wav2lip_real import load_model, warm_up

    logger.info(opt)
    model = load_model(opt.WAV2LIP_MODEL_PATH)
    warm_up(opt.batch_size, model, opt.WAV2LIP_FACE_SIZE)
    warmup_runtime_resources()


def create_web_app() -> web.Application:
    appasync = web.Application(client_max_size=1024**2 * 100)
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_get("/", lambda request: web.FileResponse(REPO_ROOT / "web" / "index.html"))
    appasync.router.add_get("/webrtcapi-asr.html", lambda request: web.FileResponse(REPO_ROOT / "web" / "webrtcapi-asr.html"))
    appasync.router.add_get("/client.js", lambda request: web.FileResponse(REPO_ROOT / "web" / "client.js"))
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/api/asr/transcribe", transcribe_audio)
    appasync.router.add_post("/api/dialog/add", add_dialog_entry)
    appasync.router.add_get("/api/dialog/history", get_dialog_history)
    appasync.router.add_post("/api/dialog/clear", clear_dialog_history)
    appasync.router.add_post("/set_silence_gate", set_silence_gate)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/interrupt_talk", interrupt_talk)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_get("/api/runtime/config", get_runtime_config)
    appasync.router.add_post("/api/runtime/config", update_runtime_config)
    appasync.router.add_get("/api/avatar-thumbnail/{avatar_id}", get_avatar_thumbnail)
    appasync.router.add_get("/api/avatar-materials", get_avatar_materials)
    appasync.router.add_get("/api/avatar-materials/jobs/{job_id}", get_avatar_generation_job)
    appasync.router.add_post("/api/avatar-materials", create_avatar_material)
    appasync.router.add_delete("/api/avatar-materials/{avatar_id}", delete_avatar_material)
    appasync.router.add_static("/web/", path=str(REPO_ROOT / "web"))

    cors = aiohttp_cors.setup(
        appasync,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        },
    )
    for route in list(appasync.router.routes()):
        cors.add(route)
    return appasync


def run_server(runner: web.AppRunner) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "0.0.0.0", opt.listenport)
    loop.run_until_complete(site.start())
    loop.run_forever()


def main() -> None:
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    args = build_arg_parser().parse_args()
    bootstrap_runtime(args)
    logger.info("start http server; port=%s", opt.listenport)
    logger.info("主控台页面: http://127.0.0.1:%s/webrtcapi-asr.html", opt.listenport)
    run_server(web.AppRunner(create_web_app(), access_log=None))


if __name__ == "__main__":
    main()
