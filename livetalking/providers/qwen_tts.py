from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading
import time
from pathlib import Path

import librosa
import numpy as np
import resampy

from ..utils.app_logger import logger
from .tts_engines import BaseTTS, State
from .tts_segments import split_tts_segments

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from qwen_tts import Qwen3TTSModel
except Exception:  # pragma: no cover
    Qwen3TTSModel = None


_DTYPE_MAP = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
}

_SPEAKER_LANGUAGE_MAP = {
    "vivian": "Chinese",
    "serena": "Chinese",
    "uncle_fu": "Chinese",
    "dylan": "Chinese",
    "eric": "Chinese",
    "ryan": "English",
    "aiden": "English",
    "ono_anna": "Japanese",
    "sohee": "Korean",
}


def is_qwen_tts_runtime_available() -> bool:
    return Qwen3TTSModel is not None


class QwenCustomVoiceTTS(BaseTTS):
    _shared_model = None
    _shared_key = None
    _shared_lock = threading.Lock()

    @classmethod
    def reset_shared_model(cls) -> None:
        with cls._shared_lock:
            cls._shared_model = None
            cls._shared_key = None

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.model_dir = Path(opt.TTS_MODEL_DIR)
        self.tokenizer_dir = Path(getattr(opt, "TTS_TOKENIZER_DIR", ""))
        self.device = str(getattr(opt, "TTS_DEVICE", "cuda:0") or "cuda:0")
        self.dtype = self._resolve_dtype(str(getattr(opt, "TTS_DTYPE", "float16") or "float16"))
        requested_attn = str(getattr(opt, "TTS_ATTN_IMPLEMENTATION", "eager") or "eager").strip().lower()
        if requested_attn in {"", "sdpa"}:
            self.attn_implementation = "sdpa"
        elif requested_attn in {"flash_attention_2", "flash_attn", "flash-attn"}:
            self.attn_implementation = "eager"
        else:
            self.attn_implementation = requested_attn
        self.speaker = str(getattr(opt, "TTS_QWEN_SPEAKER", "Vivian") or "Vivian")
        self.instruct = str(getattr(opt, "TTS_QWEN_INSTRUCT", "") or "")
        self.language = str(getattr(opt, "TTS_QWEN_LANGUAGE", "Auto") or "Auto")
        self.speed = float(getattr(opt, "TTS_QWEN_SPEED", 1.0) or 1.0)
        self.max_new_tokens = int(getattr(opt, "TTS_QWEN_MAX_NEW_TOKENS", 768) or 768)
        self.do_sample = bool(getattr(opt, "TTS_QWEN_DO_SAMPLE", False))
        self.top_k = int(getattr(opt, "TTS_QWEN_TOP_K", 50) or 50)
        self.top_p = float(getattr(opt, "TTS_QWEN_TOP_P", 1.0) or 1.0)
        self.temperature = float(getattr(opt, "TTS_QWEN_TEMPERATURE", 0.9) or 0.9)
        self.repetition_penalty = float(getattr(opt, "TTS_QWEN_REPETITION_PENALTY", 1.05) or 1.05)
        self.subtalker_dosample = bool(getattr(opt, "TTS_QWEN_SUBTALKER_DOSAMPLE", True))
        self.subtalker_top_k = int(getattr(opt, "TTS_QWEN_SUBTALKER_TOP_K", 50) or 50)
        self.subtalker_top_p = float(getattr(opt, "TTS_QWEN_SUBTALKER_TOP_P", 1.0) or 1.0)
        self.subtalker_temperature = float(getattr(opt, "TTS_QWEN_SUBTALKER_TEMPERATURE", 0.9) or 0.9)
        self._warmed_up = False

    def _resolve_dtype(self, value: str):
        if torch is None:
            return value
        normalized = _DTYPE_MAP.get(value.strip().lower(), "float16")
        resolved = getattr(torch, normalized, torch.float16)
        if resolved in {torch.float16, torch.bfloat16} and torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        return resolved

    def _build_model_key(self) -> tuple[str, str, str, str]:
        return (
            str(self.model_dir.resolve()),
            str(self.tokenizer_dir.resolve()),
            self.device,
            str(self.attn_implementation),
        )

    def _sync_cuda(self) -> None:
        if torch is None or not torch.cuda.is_available():
            return
        if not str(self.device).lower().startswith("cuda"):
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            return

    def _resolve_max_new_tokens(self, text: str) -> int:
        length_based_cap = max(128, min(512, len(text) * 12))
        return min(self.max_new_tokens, length_based_cap)

    def _split_text_segments(self, text: str) -> list[str]:
        return split_tts_segments(text)

    def _apply_speed(self, stream: np.ndarray) -> tuple[np.ndarray, float]:
        speed = float(self.speed or 1.0)
        if abs(speed - 1.0) < 1e-3:
            return stream, 0.0
        if speed <= 0:
            logger.warning("Invalid Qwen TTS speed %.3f, fallback to 1.0", speed)
            return stream, 0.0
        start = time.perf_counter()
        stretched = librosa.effects.time_stretch(stream.astype(np.float32), rate=speed)
        return np.asarray(stretched, dtype=np.float32), time.perf_counter() - start

    def _validate_paths(self) -> None:
        if not self.model_dir.is_dir():
            raise RuntimeError(f"Qwen TTS model directory does not exist: {self.model_dir}")
        if self.tokenizer_dir and not self.tokenizer_dir.is_dir():
            logger.warning("Qwen TTS tokenizer directory does not exist: %s", self.tokenizer_dir)

    def _get_model_size(self, model=None) -> str:
        if model is None:
            model = self.__class__._shared_model
        if model is None:
            return ""
        return str(getattr(getattr(model, "model", None), "tts_model_size", "") or "").lower()

    def _is_06b_model(self, model=None) -> bool:
        size = self._get_model_size(model)
        return "0.6" in size or "0b6" in size or "0.6b" in size

    def _infer_language(self, speaker: str) -> str:
        speaker_key = str(speaker or "").strip().lower()
        if self.language and self.language.strip().lower() != "auto":
            return self.language
        return _SPEAKER_LANGUAGE_MAP.get(speaker_key, "Chinese")

    def _get_model(self):
        if torch is None or Qwen3TTSModel is None:
            raise RuntimeError("qwen-tts or torch is not installed, cannot use qwen3_customvoice")

        self._validate_paths()
        key = self._build_model_key()
        with self.__class__._shared_lock:
            if self.__class__._shared_model is None or self.__class__._shared_key != key:
                load_start = time.perf_counter()
                logger.info(
                    "Loading Qwen TTS model: model_dir=%s tokenizer_dir=%s device=%s attn=%s",
                    self.model_dir,
                    self.tokenizer_dir,
                    self.device,
                    self.attn_implementation,
                )
                kwargs = {
                    "device_map": self.device,
                    "dtype": self.dtype,
                    "attn_implementation": self.attn_implementation,
                    "tokenizer_path": str(self.tokenizer_dir),
                }
                try:
                    self.__class__._shared_model = Qwen3TTSModel.from_pretrained(str(self.model_dir), **kwargs)
                except TypeError:
                    kwargs.pop("tokenizer_path", None)
                    self.__class__._shared_model = Qwen3TTSModel.from_pretrained(str(self.model_dir), **kwargs)
                self.__class__._shared_key = key
                load_elapsed = time.perf_counter() - load_start
                logger.info(
                    "Qwen TTS model loaded in %.3fs, generate_defaults=%s",
                    load_elapsed,
                    getattr(self.__class__._shared_model, "generate_defaults", {}),
                )
        return self.__class__._shared_model

    def warmup(self) -> None:
        model = self._get_model()
        if self._warmed_up:
            return
        try:
            logger.info("Qwen TTS warmup synthesis start")
            self._synthesize("hello", warmup=True)
            self._warmed_up = True
            logger.info("Qwen TTS warmup synthesis done")
        except Exception:
            logger.exception("Qwen TTS warmup synthesis failed")

    def _synthesize(self, text: str, warmup: bool = False) -> tuple[np.ndarray, int, dict[str, float]]:
        model = self._get_model()
        language = self._infer_language(self.speaker)
        instruct = self.instruct or ""
        if self._is_06b_model(model) and instruct:
            logger.warning("Qwen 0.6B CustomVoice ignores instruct for stability: %s", instruct)
            instruct = ""
        prepared_start = time.perf_counter()
        input_ids = model._tokenize_texts([model._build_assistant_text(text)])
        instruct_ids = [None]
        if instruct:
            instruct_ids = [model._tokenize_texts([model._build_instruct_text(instruct)])[0]]
        gen_kwargs = model._merge_generate_kwargs(
            max_new_tokens=self._resolve_max_new_tokens(text),
            do_sample=self.do_sample,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            subtalker_dosample=self.subtalker_dosample,
            subtalker_top_k=self.subtalker_top_k,
            subtalker_top_p=self.subtalker_top_p,
            subtalker_temperature=self.subtalker_temperature,
        )
        prepared_elapsed = time.perf_counter() - prepared_start

        self._sync_cuda()
        generate_start = time.perf_counter()
        talker_codes_list, _ = model.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=[language],
            speakers=[self.speaker],
            non_streaming_mode=True,
            **gen_kwargs,
        )
        self._sync_cuda()
        generate_elapsed = time.perf_counter() - generate_start

        self._sync_cuda()
        decode_start = time.perf_counter()
        wavs, sample_rate = model.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        self._sync_cuda()
        decode_elapsed = time.perf_counter() - decode_start
        if not wavs:
            logger.warning("Qwen TTS returned empty audio for text=%s warmup=%s", text, warmup)
            return np.zeros(0, dtype=np.float32), 24000, {
                "prepare": prepared_elapsed,
                "generate": generate_elapsed,
                "decode": decode_elapsed,
            }
        pcm = np.asarray(wavs[0], dtype=np.float32)
        if self.speed and abs(float(self.speed) - 1.0) >= 1e-3:
            pcm, stretch_elapsed = self._apply_speed(pcm)
        else:
            stretch_elapsed = 0.0
        logger.info(
            "Qwen TTS timings: prepare=%.3fs generate=%.3fs decode=%.3fs stretch=%.3fs sample_rate=%s samples=%s speaker=%s language=%s speed=%.2f text_len=%s max_new_tokens=%s warmup=%s",
            prepared_elapsed,
            generate_elapsed,
            decode_elapsed,
            stretch_elapsed,
            int(sample_rate),
            len(pcm),
            self.speaker,
            language,
            float(self.speed or 1.0),
            len(text),
            gen_kwargs.get("max_new_tokens"),
            warmup,
        )
        return pcm, int(sample_rate), {
            "prepare": prepared_elapsed,
            "generate": generate_elapsed,
            "decode": decode_elapsed,
            "stretch": stretch_elapsed,
        }

    def txt_to_audio(self, msg: tuple[str, dict]) -> None:
        text, textevent = msg
        text = (text or "").strip()
        if not text:
            return

        total_start = time.perf_counter()
        segments = self._split_text_segments(text)
        if not segments:
            return

        textevent = dict(textevent or {})
        queued_chunks = 0
        queued_samples = 0
        first_chunk_elapsed = None
        total_audio_samples = 0
        total_resample_elapsed = 0.0
        segment_timings: list[dict[str, float | int]] = []

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen_tts_prefetch") as executor:
            future: Future[tuple[np.ndarray, int, dict[str, float]]] | None = executor.submit(
                self._synthesize,
                segments[0],
                False,
            )
            next_segment_index = 1

            for segment_index, segment_text in enumerate(segments, start=1):
                if future is None:
                    break
                stream, sample_rate, timings = future.result()
                future = None

                if next_segment_index < len(segments) and self.state == State.RUNNING:
                    future = executor.submit(self._synthesize, segments[next_segment_index], False)
                    next_segment_index += 1

                if stream.size == 0:
                    continue

                segment_resample_elapsed = 0.0
                if sample_rate != self.sample_rate:
                    resample_start = time.perf_counter()
                    stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
                    segment_resample_elapsed = time.perf_counter() - resample_start
                    total_resample_elapsed += segment_resample_elapsed

                total_audio_samples += int(stream.shape[0])
                segment_timings.append(
                    {
                        "index": segment_index,
                        "text_len": len(segment_text),
                        "prepare": float(timings.get("prepare", 0.0)),
                        "generate": float(timings.get("generate", 0.0)),
                        "decode": float(timings.get("decode", 0.0)),
                        "stretch": float(timings.get("stretch", 0.0)),
                        "resample": segment_resample_elapsed,
                        "duration": float(len(stream)) / float(self.sample_rate),
                    }
                )
                logger.info(
                    "Qwen TTS segment[%s/%s]: text_len=%s prepare=%.3fs generate=%.3fs decode=%.3fs stretch=%.3fs resample=%.3fs duration=%.3fs",
                    segment_index,
                    len(segments),
                    len(segment_text),
                    float(timings.get("prepare", 0.0)),
                    float(timings.get("generate", 0.0)),
                    float(timings.get("decode", 0.0)),
                    float(timings.get("stretch", 0.0)),
                    segment_resample_elapsed,
                    float(len(stream)) / float(self.sample_rate),
                )

                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk and self.state == State.RUNNING:
                    eventpoint = {}
                    streamlen -= self.chunk
                    if queued_chunks == 0:
                        eventpoint = {"status": "start", "text": text}
                        eventpoint.update(**textevent)
                    elif segment_index == len(segments) and streamlen < self.chunk:
                        eventpoint = {"status": "end", "text": text}
                        eventpoint.update(**textevent)
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                    if first_chunk_elapsed is None:
                        first_chunk_elapsed = time.perf_counter() - total_start
                    queued_chunks += 1
                    queued_samples += self.chunk
                    idx += self.chunk

        total_elapsed = time.perf_counter() - total_start
        textevent.setdefault("tts_elapsed", total_elapsed)
        textevent.setdefault("tts_duration", float(total_audio_samples) / float(self.sample_rate))
        logger.info(
            "Qwen TTS stream timings: total=%.3fs first_chunk=%.3fs resample=%.3fs queued_chunks=%s queued_samples=%s segments=%s text_len=%s",
            total_elapsed,
            float(first_chunk_elapsed or total_elapsed),
            total_resample_elapsed,
            queued_chunks,
            queued_samples,
            len(segment_timings),
            len(text),
        )
