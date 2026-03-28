from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import re
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np
import resampy

from ..utils.app_logger import logger
from .tts_engines import BaseTTS, State
from .tts_segments import split_tts_segments

try:
    import sherpa_onnx
except Exception:  # pragma: no cover
    sherpa_onnx = None


def _repair_mojibake(text: str) -> str:
    suspicious_markers = ("Ã", "Â", "ä", "å", "æ", "ç", "ï", "¢", "€", "™")
    if not text or not any(marker in text for marker in suspicious_markers):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


_TTS_GARBAGE_CHARS = {
    "锛",
    "銆",
    "閿",
    "鍥",
    "璇",
    "娴",
    "浣",
    "闂",
    "鎼",
    "鎯",
    "闊",
    "鈧",
    "鈩",
}


def _sanitize_tts_text(text: str) -> str:
    text = _repair_mojibake((text or "").strip())
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text).replace("\uFFFD", " ")
    cleaned: list[str] = []
    for char in text:
        if char in _TTS_GARBAGE_CHARS:
            continue
        if char.isspace():
            cleaned.append(" ")
            continue
        if char.isalnum():
            cleaned.append(char)
            continue
        if "\u4e00" <= char <= "\u9fff" or "\u3400" <= char <= "\u4dbf":
            cleaned.append(char)
            continue
        if unicodedata.category(char).startswith("P"):
            cleaned.append(char)
            continue
    return re.sub(r"\s+", " ", "".join(cleaned)).strip()


class SherpaOnnxVitsTTS(BaseTTS):
    _shared_tts = None
    _shared_lock = threading.Lock()

    @classmethod
    def reset_shared_tts(cls) -> None:
        with cls._shared_lock:
            cls._shared_tts = None

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.model_dir = Path(opt.TTS_MODEL_DIR)
        self.provider = opt.TTS_PROVIDER
        self.num_threads = int(opt.TTS_NUM_THREADS)
        self.speaker_id = int(opt.TTS_SPEAKER_ID)
        self.speed = float(opt.TTS_SPEED)
        self.rule_fsts = list(getattr(opt, "TTS_RULE_FSTS", []))
        self._tts = None

    def _resolve_rule_fsts(self) -> str:
        resolved: list[str] = []
        for item in self.rule_fsts:
            path = Path(item)
            if not path.is_absolute():
                path = self.model_dir / item
            if path.exists():
                resolved.append(str(path.resolve()))
            else:
                logger.warning("Sherpa TTS rule fst skipped because file does not exist: %s", path)
        return ",".join(resolved)

    def _get_tts(self):
        if sherpa_onnx is None:
            raise RuntimeError("sherpa-onnx is not installed, cannot use sherpa_onnx_vits")
        if self._tts is None:
            with self.__class__._shared_lock:
                if self.__class__._shared_tts is None:
                    model_file = self.model_dir / "model.onnx"
                    lexicon_file = self.model_dir / "lexicon.txt"
                    tokens_file = self.model_dir / "tokens.txt"
                    missing = [str(path) for path in (model_file, lexicon_file, tokens_file) if not path.exists()]
                    if missing:
                        raise RuntimeError(
                            "Sherpa-ONNX TTS model directory is incomplete, missing files: " + ", ".join(missing)
                        )
                    config = sherpa_onnx.OfflineTtsConfig(
                        model=sherpa_onnx.OfflineTtsModelConfig(
                            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                                model=str(model_file),
                                lexicon=str(lexicon_file),
                                tokens=str(tokens_file),
                                data_dir="",
                            ),
                            provider=self.provider,
                            num_threads=self.num_threads,
                            debug=False,
                        ),
                        rule_fsts=self._resolve_rule_fsts(),
                        max_num_sentences=1,
                    )
                    if not config.validate():
                        raise RuntimeError("Invalid sherpa-onnx TTS config")
                    self.__class__._shared_tts = sherpa_onnx.OfflineTts(config)
                    logger.info(
                        "Sherpa-ONNX TTS loaded: model_dir=%s provider=%s num_threads=%s speaker_id=%s",
                        self.model_dir,
                        self.provider,
                        self.num_threads,
                        self.speaker_id,
                    )
                self._tts = self.__class__._shared_tts
        return self._tts

    def warmup(self) -> None:
        self._get_tts()

    def put_msg_txt(self, msg: str, datainfo: dict | None = None) -> None:
        cleaned = _sanitize_tts_text(msg)
        if not cleaned:
            return
        super().put_msg_txt(cleaned, datainfo)

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        tts = self._get_tts()
        start = time.perf_counter()
        if hasattr(sherpa_onnx, "GenerationConfig"):
            gen_config = sherpa_onnx.GenerationConfig()
            gen_config.sid = self.speaker_id
            gen_config.speed = self.speed
            gen_config.silence_scale = 0.2
            audio = tts.generate(text, gen_config)
        else:
            audio = tts.generate(text, sid=self.speaker_id, speed=self.speed)
        elapsed = time.perf_counter() - start
        if len(audio.samples) == 0:
            logger.warning("Sherpa TTS synth produced empty audio in %.3fs for text=%s", elapsed, text)
            return np.zeros(0, dtype=np.float32), int(audio.sample_rate or 24000)
        pcm = np.asarray(audio.samples, dtype=np.float32)
        duration_s = float(len(pcm)) / float(audio.sample_rate or 24000)
        logger.info(
            "Sherpa TTS synth done in %.3fs, sample_rate=%s, samples=%s, duration=%.3fs, text_len=%s",
            elapsed,
            int(audio.sample_rate),
            len(pcm),
            duration_s,
            len(text),
        )
        return pcm, int(audio.sample_rate)

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        text = (text or "").strip()
        if not text:
            return

        segments = split_tts_segments(text)
        if not segments:
            return
        sanitized_segments = [_sanitize_tts_text(segment) for segment in segments]
        sanitized_segments = [segment for segment in sanitized_segments if segment]
        if not sanitized_segments:
            return

        total_start = time.perf_counter()
        textevent = dict(textevent or {})
        queued_chunks = 0
        queued_samples = 0
        first_chunk_elapsed = None
        total_audio_samples = 0

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="sherpa_tts_prefetch") as executor:
            future: Future[tuple[np.ndarray, int]] | None = executor.submit(self._synthesize, sanitized_segments[0])
            next_segment_index = 1

            for segment_index, _segment_text in enumerate(sanitized_segments, start=1):
                if future is None:
                    break
                stream, sample_rate = future.result()
                future = None

                if next_segment_index < len(sanitized_segments) and self.state == State.RUNNING:
                    future = executor.submit(self._synthesize, sanitized_segments[next_segment_index])
                    next_segment_index += 1

                if stream.size == 0:
                    continue

                if sample_rate != self.sample_rate:
                    stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
                    logger.info(
                        "Sherpa TTS resampled to %s Hz, samples=%s, duration=%.3fs",
                        self.sample_rate,
                        len(stream),
                        float(len(stream)) / float(self.sample_rate),
                    )

                total_audio_samples += int(stream.shape[0])
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk and self.state == State.RUNNING:
                    eventpoint = {}
                    streamlen -= self.chunk
                    if queued_chunks == 0:
                        eventpoint = {"status": "start", "text": text}
                        eventpoint.update(**textevent)
                    elif segment_index == len(sanitized_segments) and streamlen < self.chunk:
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
            "Sherpa TTS total elapsed %.3fs, first_chunk=%.3fs, audio_duration=%.3fs, text_len=%s, llm_elapsed=%s, queued_chunks=%s, queued_samples=%s, segments=%s",
            total_elapsed,
            float(first_chunk_elapsed or total_elapsed),
            float(total_audio_samples) / float(self.sample_rate),
            len(text),
            textevent.get("llm_elapsed"),
            queued_chunks,
            queued_samples,
            len(segments),
        )

        dialog_id = str(textevent.get("dialog_id", ""))
        if dialog_id and hasattr(self.parent, "update_dialog_meta"):
            self.parent.update_dialog_meta(
                dialog_id,
                {
                    "tts_elapsed": total_elapsed,
                    "tts_duration": float(len(stream)) / float(self.sample_rate),
                },
            )
