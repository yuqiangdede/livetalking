from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import resampy
import soundfile as sf

from ..utils.app_logger import logger
from .tts_engines import BaseTTS, State
from .tts_segments import split_tts_segments

try:
    import pyttsx3
except Exception:  # pragma: no cover
    pyttsx3 = None

PYTTSX3_KNOWN_VOICE_CHOICES = (
    (
        "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ZH-CN_HUIHUI_11.0",
        "Huihui（中文）",
        "Microsoft Huihui Desktop - Chinese (Simplified)",
    ),
    (
        "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0",
        "Zira（英文）",
        "Microsoft Zira Desktop - English (United States)",
    ),
)


def list_pyttsx3_voice_options(driver_name: str | None = None) -> list[dict]:
    if pyttsx3 is None:
        return []

    engine = None
    try:
        engine = pyttsx3.init(driverName=driver_name) if driver_name else pyttsx3.init()
        voices = list(engine.getProperty("voices") or [])
    except Exception:
        logger.exception("pyttsx3 voice discovery failed")
        return []
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    discovered_voices: dict[str, str] = {}
    for voice in voices:
        voice_id = str(getattr(voice, "id", "") or "").strip()
        if not voice_id:
            continue
        discovered_voices[voice_id] = str(getattr(voice, "name", "") or "").strip()

    options: list[dict] = []
    for voice_id, label, fallback_name in PYTTSX3_KNOWN_VOICE_CHOICES:
        voice_name = discovered_voices.get(voice_id, fallback_name)
        options.append(
            {
                "id": voice_id,
                "label": label,
                "desc": f"{voice_name} / {voice_id}",
            }
        )
    return options


def _collect_voice_map(engine) -> dict[str, str]:
    voices = engine.getProperty("voices") or []
    return {
        str(getattr(voice, "id", "") or "").strip(): str(getattr(voice, "name", "") or "").strip()
        for voice in voices
        if str(getattr(voice, "id", "") or "").strip()
    }


class Pyttsx3TTS(BaseTTS):
    _shared_lock = threading.Lock()

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.driver_name = str(getattr(opt, "TTS_PYTTSX3_DRIVER_NAME", "") or "")
        self.voice_id = str(getattr(opt, "TTS_PYTTSX3_VOICE_ID", "") or "")
        self.rate = int(getattr(opt, "TTS_PYTTSX3_RATE", 175) or 175)
        self.volume = float(getattr(opt, "TTS_PYTTSX3_VOLUME", 1.0) or 1.0)

    def _build_engine(self):
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is not installed, cannot use pyttsx3")
        try:
            return pyttsx3.init(driverName=self.driver_name or None)
        except TypeError:
            return pyttsx3.init()

    def _configure_engine(self, engine, voice_id: str | None = None) -> str:
        try:
            voice_map = _collect_voice_map(engine)
            requested_voice_id = str(voice_id if voice_id is not None else self.voice_id).strip()
            selected_voice_id = ""
            if requested_voice_id:
                if requested_voice_id in voice_map:
                    engine.setProperty("voice", requested_voice_id)
                else:
                    logger.warning("pyttsx3 voice_id not found, using default voice: %s", requested_voice_id)
            engine.setProperty("rate", int(self.rate))
            engine.setProperty("volume", max(0.0, min(1.0, float(self.volume))))
            selected_voice_id = str(engine.getProperty("voice") or "").strip()
            logger.info(
                "pyttsx3 engine configured: driver=%s requested_voice_id=%s selected_voice_id=%s selected_voice_name=%s rate=%s volume=%.2f available_voices=%s",
                self.driver_name or "default",
                requested_voice_id or "<system-default>",
                selected_voice_id or "<unknown>",
                voice_map.get(selected_voice_id, ""),
                int(self.rate),
                max(0.0, min(1.0, float(self.volume))),
                list(voice_map.items()),
            )
            return selected_voice_id
        except Exception:
            logger.exception("pyttsx3 engine configuration failed")
            raise

    def warmup(self) -> None:
        list_pyttsx3_voice_options(self.driver_name or None)

    def _synthesize_segment(
        self,
        segment_text: str,
        segment_index: int,
        segment_count: int,
        voice_id: str | None,
    ) -> tuple[np.ndarray, int, int, str]:
        temp_path = None
        engine = None
        try:
            engine = self._build_engine()
            selected_voice_id = self._configure_engine(engine, voice_id=voice_id)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_path = temp_file.name
            logger.info(
                "pyttsx3 segment synth start: segment=%s/%s text=%s temp_wav=%s requested_voice_id=%s",
                segment_index,
                segment_count,
                segment_text,
                temp_path,
                (voice_id or "").strip() or "<system-default>",
            )
            engine.save_to_file(segment_text, temp_path)
            engine.runAndWait()
            wav_size = Path(temp_path).stat().st_size if Path(temp_path).exists() else -1
            logger.info(
                "pyttsx3 segment synth file ready: segment=%s/%s wav_size=%s text=%s selected_voice_id=%s",
                segment_index,
                segment_count,
                wav_size,
                segment_text,
                selected_voice_id or "<unknown>",
            )

            stream, sample_rate = sf.read(temp_path, dtype="float32")
            if stream.ndim > 1:
                stream = stream[:, 0]
            if sample_rate != self.sample_rate and stream.shape[0] > 0:
                stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
            logger.info(
                "pyttsx3 segment audio loaded: segment=%s/%s sample_rate=%s samples=%s duration=%.3fs selected_voice_id=%s",
                segment_index,
                segment_count,
                sample_rate,
                int(stream.shape[0]),
                float(stream.shape[0]) / float(self.sample_rate) if stream.shape[0] > 0 else 0.0,
                selected_voice_id or "<unknown>",
            )
            return stream, int(sample_rate), wav_size, selected_voice_id
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    def txt_to_audio(self, msg: tuple[str, dict]) -> None:
        text, textevent = msg
        text = (text or "").strip()
        if not text:
            return

        segments = split_tts_segments(text)
        if not segments:
            return

        total_start = time.perf_counter()
        textevent = dict(textevent or {})
        queued_chunks = 0
        queued_samples = 0
        first_chunk_elapsed = None
        total_audio_samples = 0

        try:
            for segment_index, segment_text in enumerate(segments, start=1):
                stream, _sample_rate, wav_size, used_voice_id = self._synthesize_segment(
                    segment_text,
                    segment_index,
                    len(segments),
                    self.voice_id,
                )
                if stream.size == 0 and self.voice_id:
                    logger.warning(
                        "pyttsx3 segment fallback to system default voice: segment=%s/%s requested_voice_id=%s wav_size=%s text=%s",
                        segment_index,
                        len(segments),
                        self.voice_id,
                        wav_size,
                        segment_text,
                    )
                    stream, _sample_rate, wav_size, used_voice_id = self._synthesize_segment(
                        segment_text,
                        segment_index,
                        len(segments),
                        "",
                    )
                if stream.size == 0:
                    logger.warning(
                        "pyttsx3 returned empty audio for segment=%s/%s voice_id=%s text=%s",
                        segment_index,
                        len(segments),
                        used_voice_id or "<system-default>",
                        segment_text,
                    )
                    continue

                total_audio_samples += int(stream.shape[0])
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
                "pyttsx3 synth done in %.3fs, first_chunk=%.3fs, duration=%.3fs, text_len=%s, queued_chunks=%s, queued_samples=%s, segments=%s",
                total_elapsed,
                float(first_chunk_elapsed or total_elapsed),
                float(total_audio_samples) / float(self.sample_rate),
                len(text),
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
                        "tts_duration": float(total_audio_samples) / float(self.sample_rate),
                    },
                )
        except Exception:
            logger.exception("pyttsx3 synth failed")
