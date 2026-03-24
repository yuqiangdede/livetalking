from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

import av
import numpy as np

from ..utils.app_logger import logger
from .asr_enhancements import apply_phonetic_replacements, parse_hotword_lines, parse_phonetic_replacements
from .local_punc import CTTransformerPunctuationRestorer

try:
    from funasr import AutoModel
except Exception:  # pragma: no cover
    AutoModel = None


TARGET_SAMPLE_RATE = 16000


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return ""
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？!?、,.;:])", r"\1", text)
    return text


def _ensure_terminal_punctuation(text: str, lang: str) -> str:
    if not text:
        return ""
    if re.search(r"[。！？.!?…]$", text):
        return text
    if lang == "zh":
        return text + "。"
    return text + "."


def _post_process_text(text: str, lang: str) -> str:
    text = _normalize_whitespace(text)
    return _ensure_terminal_punctuation(text, lang)


def _finalize_punctuated_text(text: str, lang: str) -> str:
    text = str(text or "").strip()
    return _ensure_terminal_punctuation(text, lang)


def _normalize_lang(raw: str | None, text: str) -> str:
    if raw:
        lang = raw.lower()
        if lang.startswith(("zh", "cn")):
            return "zh"
        if lang.startswith(("en", "english")):
            return "en"
    zh_count = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    return "zh" if zh_count > 0 else "en"


def _pick_text(result: Any) -> str:
    if isinstance(result, dict):
        text = result.get("text")
        if isinstance(text, str):
            return text.strip()
        sentence_info = result.get("sentence_info")
        if isinstance(sentence_info, list):
            parts = [item.get("text", "") for item in sentence_info if isinstance(item, dict)]
            return "".join(parts).strip()
    return ""


def _write_pcm_to_wav(audio_bytes: bytes, sample_rate: int, temp_dir: str | None = None) -> str:
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".wav", dir=temp_dir)
    os.close(fd)
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise
    return path


def _decode_audio_with_ffmpeg(audio_bytes: bytes, sample_rate: int) -> tuple[bytes, int]:
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg is not available in PATH")

    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed: {stderr or 'unknown error'}")
    return proc.stdout, sample_rate


def _trim_silence_pcm16(audio_bytes: bytes, sample_rate: int) -> tuple[bytes, int, int]:
    if not audio_bytes:
        return b"", 0, 0

    if len(audio_bytes) % 2:
        audio_bytes = audio_bytes[:-1]

    pcm = np.frombuffer(audio_bytes, dtype=np.int16)
    if pcm.size == 0:
        return b"", 0, 0

    frame_samples = max(int(sample_rate * 0.02), 1)
    if pcm.size < frame_samples:
        return audio_bytes, 0, 0

    usable_samples = (pcm.size // frame_samples) * frame_samples
    frames = pcm[:usable_samples].reshape(-1, frame_samples)
    if frames.size == 0:
        return audio_bytes, 0, 0

    frame_energy = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    peak = float(np.max(np.abs(pcm)))
    if peak <= 0:
        return b"", pcm.size, 0

    threshold = max(500.0, peak * 0.02)
    speech_frames = np.flatnonzero(frame_energy > threshold)
    if speech_frames.size == 0:
        return b"", pcm.size, 0

    pad_frames = max(int(round(0.2 / 0.02)), 1)
    start_frame = max(int(speech_frames[0]) - pad_frames, 0)
    end_frame = min(int(speech_frames[-1]) + pad_frames + 1, frames.shape[0])

    trimmed = frames[start_frame:end_frame].reshape(-1).astype(np.int16).tobytes()
    if usable_samples < pcm.size:
        trimmed += pcm[usable_samples:].tobytes()

    removed_leading = start_frame * frame_samples
    removed_trailing = (pcm.size - usable_samples) + (frames.shape[0] - end_frame) * frame_samples
    return trimmed, removed_leading, removed_trailing


def decode_audio_to_pcm16(audio_bytes: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[bytes, int]:
    if not audio_bytes:
        return b"", sample_rate

    try:
        container = av.open(io.BytesIO(audio_bytes))
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        pcm_chunks: list[bytes] = []
        try:
            for frame in container.decode(audio=0):
                resampled = resampler.resample(frame)
                if resampled is None:
                    continue
                if not isinstance(resampled, list):
                    resampled = [resampled]
                for out_frame in resampled:
                    pcm_chunks.append(bytes(out_frame.planes[0]))
            flushed = resampler.resample(None)
            if flushed is not None:
                if not isinstance(flushed, list):
                    flushed = [flushed]
                for out_frame in flushed:
                    pcm_chunks.append(bytes(out_frame.planes[0]))
        finally:
            container.close()
        return b"".join(pcm_chunks), sample_rate
    except Exception as exc:
        logger.warning("PyAV decode failed, falling back to ffmpeg: %s", exc)
        return _decode_audio_with_ffmpeg(audio_bytes, sample_rate)


class ParaformerProvider:
    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        batch_size_s: int = 300,
        hotwords: str = "",
        phonetic_replacements: str = "",
        punc_model_dir: str = "models/asr/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        temp_dir: str | None = None,
    ):
        self.model_dir = model_dir
        self.device = device
        self.batch_size_s = int(batch_size_s)
        self._hotwords_text = ""
        self._hotword_lines: list[str] = []
        self._phonetic_replacements_text = ""
        self._phonetic_rules: list[tuple[str, str]] = []
        self._hotword_file_path = ""
        self.punc_model_dir = punc_model_dir
        self.temp_dir = temp_dir
        self._model = None
        self._punc_restorer: CTTransformerPunctuationRestorer | None = None
        self._config_lock = threading.Lock()
        self._lock = threading.Lock()
        self.update_asr_rules(hotwords=hotwords, phonetic_replacements=phonetic_replacements)
        self._init_punctuation_restorer()

    def _resolve_hotword_file_path(self) -> Path:
        base_dir = Path(self.temp_dir) if self.temp_dir else Path.cwd() / "runtime" / "tmp"
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / "asr_hotwords.txt"

    def _refresh_hotword_file_locked(self) -> None:
        hotword_path = self._resolve_hotword_file_path()
        if not self._hotword_lines:
            hotword_path.unlink(missing_ok=True)
            self._hotword_file_path = ""
            return

        content = "\n".join(self._hotword_lines) + "\n"
        if not hotword_path.exists() or hotword_path.read_text(encoding="utf-8") != content:
            hotword_path.write_text(content, encoding="utf-8")
        self._hotword_file_path = str(hotword_path)

    def update_asr_rules(self, hotwords: str | None = None, phonetic_replacements: str | None = None) -> None:
        with self._config_lock:
            if hotwords is not None:
                self._hotwords_text = str(hotwords or "")
            if phonetic_replacements is not None:
                self._phonetic_replacements_text = str(phonetic_replacements or "")
            self._hotword_lines = parse_hotword_lines(self._hotwords_text)
            self._phonetic_rules = parse_phonetic_replacements(self._phonetic_replacements_text)
            self._refresh_hotword_file_locked()
            logger.info(
                "ASR enhancement rules updated: hotwords=%s phonetic_rules=%s",
                len(self._hotword_lines),
                len(self._phonetic_rules),
            )

    def _snapshot_asr_rules(self) -> tuple[str | None, list[tuple[str, str]]]:
        with self._config_lock:
            hotword_file = self._hotword_file_path if self._hotword_lines else None
            phonetic_rules = list(self._phonetic_rules)
        return hotword_file, phonetic_rules

    def _validate_model_dir(self) -> Path:
        model_path = Path(self.model_dir)
        required_files = [
            model_path / "model.pt",
            model_path / "config.yaml",
            model_path / "tokens.json",
        ]
        missing = [path.name for path in required_files if not path.exists()]
        if missing:
            raise RuntimeError(
                "paraformer local model directory is incomplete: "
                f"model_dir={model_path} missing={missing}"
            )
        return model_path

    def _validate_punc_model_dir(self) -> Path:
        model_path = Path(self.punc_model_dir)
        required_files = [
            model_path / "model.int8.onnx",
            model_path / "config.yaml",
            model_path / "tokens.json",
        ]
        missing = [path.name for path in required_files if not path.exists()]
        if missing:
            raise RuntimeError(
                "punctuation model directory is incomplete: "
                f"model_dir={model_path} missing={missing}"
            )
        return model_path

    def _init_punctuation_restorer(self) -> None:
        try:
            punc_model_path = self._validate_punc_model_dir()
            self._punc_restorer = CTTransformerPunctuationRestorer(str(punc_model_path))
            logger.info("Punctuation model loaded: model_dir=%s", punc_model_path)
        except Exception as exc:
            self._punc_restorer = None
            logger.warning("Punctuation model is unavailable, fallback to terminal punctuation only: %s", exc)

    def _get_model(self):
        if AutoModel is None:
            raise RuntimeError("funasr is not installed, cannot use paraformer")
        if self._model is None:
            with self._lock:
                if self._model is None:
                    model_path = self._validate_model_dir()
                    self._model = AutoModel(model=str(model_path), device=self.device)
                    logger.info("Paraformer model loaded: model_dir=%s device=%s", self.model_dir, self.device)
        return self._model

    def warmup(self) -> None:
        self._get_model()
        if self._punc_restorer is None:
            self._init_punctuation_restorer()

    def _restore_punctuation(self, text: str, lang: str) -> str:
        restorer = self._punc_restorer
        if restorer is None:
            return _finalize_punctuated_text(text, lang)
        try:
            punctuated = restorer.punctuate(text)
        except Exception as exc:
            logger.warning("Punctuation restoration failed, fallback to terminal punctuation only: %s", exc)
            return _finalize_punctuated_text(text, lang)
        if not punctuated:
            return _finalize_punctuated_text(text, lang)
        return _finalize_punctuated_text(punctuated, lang)

    def _cleanup_temp_file(self, wav_path: str) -> None:
        path = Path(wav_path)
        for _ in range(5):
            try:
                path.unlink(missing_ok=True)
                return
            except PermissionError:
                time.sleep(0.2)
            except FileNotFoundError:
                return
        logger.warning("Paraformer temp wav cleanup skipped because file is still in use: %s", wav_path)

    def transcribe_pcm16(self, audio_bytes: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> dict[str, Any]:
        if not audio_bytes:
            return {"text": "", "raw_text": "", "lang": "zh", "segments": []}

        audio_bytes, removed_leading, removed_trailing = _trim_silence_pcm16(audio_bytes, sample_rate)
        if not audio_bytes:
            return {"text": "", "raw_text": "", "lang": "zh", "segments": []}
        if removed_leading or removed_trailing:
            logger.info(
                "ASR trimmed silence: leading_samples=%s trailing_samples=%s sample_rate=%s",
                removed_leading,
                removed_trailing,
                sample_rate,
            )

        model = self._get_model()
        hotword_file, phonetic_rules = self._snapshot_asr_rules()
        wav_path = _write_pcm_to_wav(audio_bytes, sample_rate, self.temp_dir)
        try:
            generate_kwargs: dict[str, Any] = {"batch_size_s": self.batch_size_s}
            if hotword_file:
                generate_kwargs["hotword"] = hotword_file
            result = model.generate(input=wav_path, **generate_kwargs)
            item = result[0] if isinstance(result, list) and result else result
            text = _pick_text(item)
            lang = _normalize_lang(str(item.get("lang") or item.get("language") or ""), text) if isinstance(item, dict) else _normalize_lang("", text)
            text = _normalize_whitespace(text)
            text, _ = apply_phonetic_replacements(text, phonetic_rules)
            raw_text = text
            text = self._restore_punctuation(text, lang)
            segments = item.get("sentence_info", []) if isinstance(item, dict) else []
            return {"text": text, "raw_text": raw_text, "lang": lang, "segments": segments}
        finally:
            self._cleanup_temp_file(wav_path)
