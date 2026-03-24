from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import resampy

from ..utils.app_logger import logger
from .tts_engines import BaseTTS, State

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
        text = _repair_mojibake((text or "").strip())
        if not text:
            return

        total_start = time.perf_counter()
        stream, sample_rate = self._synthesize(text)
        if stream.size == 0:
            logger.warning("Sherpa-ONNX TTS returned empty audio for text=%s", text)
            return

        if sample_rate != self.sample_rate:
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
            logger.info(
                "Sherpa TTS resampled to %s Hz, samples=%s, duration=%.3fs",
                self.sample_rate,
                len(stream),
                float(len(stream)) / float(self.sample_rate),
            )

        total_elapsed = time.perf_counter() - total_start
        textevent = dict(textevent or {})
        textevent.setdefault("tts_elapsed", total_elapsed)
        textevent.setdefault("tts_duration", float(len(stream)) / float(self.sample_rate))

        streamlen = stream.shape[0]
        idx = 0
        while streamlen >= self.chunk and self.state == State.RUNNING:
            eventpoint = {}
            streamlen -= self.chunk
            if idx == 0:
                eventpoint = {"status": "start", "text": text}
                eventpoint.update(**textevent)
            elif streamlen < self.chunk:
                eventpoint = {"status": "end", "text": text}
                eventpoint.update(**textevent)
            self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
            idx += self.chunk

        logger.info(
            "Sherpa TTS total elapsed %.3fs, audio_duration=%.3fs, text_len=%s, llm_elapsed=%s",
            total_elapsed,
            float(len(stream)) / float(self.sample_rate),
            len(text),
            textevent.get("llm_elapsed"),
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
