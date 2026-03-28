from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import os
import shutil
import subprocess
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
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from TTS.api import TTS
except Exception:  # pragma: no cover
    TTS = None

try:
    from TTS.utils.synthesizer import Synthesizer
except Exception:  # pragma: no cover
    Synthesizer = None

try:
    import TTS.tts.models.xtts as xtts_module
except Exception:  # pragma: no cover
    xtts_module = None


def is_coqui_xtts_runtime_available() -> bool:
    return TTS is not None and Synthesizer is not None


class CoquiXTTSV2TTS(BaseTTS):
    _shared_model = None
    _shared_key = None
    _shared_lock = threading.Lock()
    _prepared_reference_cache: dict[tuple[str, float, int], str] = {}
    _prepared_reference_lock = threading.Lock()
    _xtts_audio_patch_done = False

    @classmethod
    def reset_shared_model(cls) -> None:
        with cls._shared_lock:
            cls._shared_model = None
            cls._shared_key = None

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.model_dir = Path(getattr(opt, "TTS_MODEL_DIR", "") or "")
        self.speaker_wav_path = Path(getattr(opt, "TTS_SPEAKER_WAV_PATH", "") or "")
        self.language = str(getattr(opt, "TTS_LANGUAGE", "zh-cn") or "zh-cn").strip().lower()
        self.device = str(getattr(opt, "TTS_DEVICE", "cuda:0") or "cuda:0").strip()
        self.speed = float(getattr(opt, "TTS_SPEED", 1.1) or 1.1)
        self.use_cuda = bool(
            self.device.lower().startswith("cuda")
            and torch is not None
            and hasattr(torch, "cuda")
            and torch.cuda.is_available()
        )
        self._model = None

    def _validate_paths(self) -> None:
        if not self.model_dir.is_dir():
            raise RuntimeError(f"Coqui XTTS v2 model directory does not exist: {self.model_dir}")
        config_path = self.model_dir / "config.json"
        if not config_path.is_file():
            raise RuntimeError(f"Coqui XTTS v2 config.json not found: {config_path}")
        checkpoint_files = sorted(
            path
            for path in self.model_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".pth", ".pt"}
        )
        if not checkpoint_files:
            raise RuntimeError(f"Coqui XTTS v2 checkpoint not found in: {self.model_dir}")
        if not self.speaker_wav_path.is_file():
            raise RuntimeError(f"Coqui XTTS v2 speaker wav does not exist: {self.speaker_wav_path}")

    def _resolve_checkpoint_path(self) -> Path:
        preferred = self.model_dir / "model.pth"
        if preferred.is_file():
            return preferred
        for candidate in sorted(self.model_dir.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() in {".pth", ".pt"}:
                return candidate
        raise RuntimeError(f"Coqui XTTS v2 checkpoint not found in: {self.model_dir}")

    def _build_model_key(self) -> tuple[str, str, str, bool]:
        return (
            str(self.model_dir.resolve()),
            str(self.speaker_wav_path.resolve()),
            self.device,
            self.use_cuda,
        )

    def _get_model(self):
        if TTS is None or Synthesizer is None:
            raise RuntimeError("coqui-tts is not installed, cannot use coqui_xtts_v2")

        self._ensure_xtts_audio_patch()
        self._validate_paths()
        key = self._build_model_key()
        with self.__class__._shared_lock:
            if self.__class__._shared_model is None or self.__class__._shared_key != key:
                load_start = time.perf_counter()
                logger.info(
                    "Loading Coqui XTTS v2 model: model_dir=%s speaker_wav=%s language=%s device=%s use_cuda=%s",
                    self.model_dir,
                    self.speaker_wav_path,
                    self.language,
                    self.device,
                    self.use_cuda,
                )
                model = TTS(progress_bar=False, gpu=False)
                model.synthesizer = Synthesizer(
                    model_dir=str(self.model_dir),
                    use_cuda=self.use_cuda,
                )
                model.config = getattr(model.synthesizer, "tts_config", None)
                model.model_name = "xtts_v2_local"
                self.__class__._shared_model = model
                self.__class__._shared_key = key
                logger.info("Coqui XTTS v2 model loaded in %.3fs", time.perf_counter() - load_start)
            self._model = self.__class__._shared_model
        return self._model

    def _resolve_ffmpeg_executable(self) -> str:
        bundled = Path(__file__).resolve().parents[2] / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
        if bundled.is_file():
            return str(bundled)
        ffmpeg_on_path = shutil.which("ffmpeg")
        if ffmpeg_on_path:
            return ffmpeg_on_path
        raise RuntimeError("ffmpeg not found; Coqui XTTS v2 mp3 reference audio requires ffmpeg")

    def _prepare_reference_audio_path(self) -> Path:
        source_path = self.speaker_wav_path.resolve()
        if source_path.suffix.lower() == ".wav":
            return source_path
        if source_path.suffix.lower() != ".mp3":
            raise RuntimeError(f"Unsupported Coqui reference audio format: {source_path.suffix}")

        stat = source_path.stat()
        cache_key = (str(source_path), float(stat.st_mtime), int(stat.st_size))
        with self.__class__._prepared_reference_lock:
            cached = self.__class__._prepared_reference_cache.get(cache_key)
            if cached and Path(cached).is_file():
                return Path(cached)

            temp_dir = Path(tempfile.gettempdir()) / "livetalking_coqui_refs"
            temp_dir.mkdir(parents=True, exist_ok=True)
            target_path = temp_dir / f"{source_path.stem}_{abs(hash(cache_key)) & 0xffffffff:08x}.wav"
            ffmpeg_exe = self._resolve_ffmpeg_executable()
            command = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                "24000",
                str(target_path),
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=os.environ.copy(),
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"Failed to convert Coqui reference mp3 to wav: {exc.stderr or exc}") from exc
            self.__class__._prepared_reference_cache[cache_key] = str(target_path)
            logger.info(
                "Coqui XTTS v2 prepared reference audio: source=%s prepared_wav=%s",
                source_path,
                target_path,
            )
            return target_path

    @classmethod
    def _ensure_xtts_audio_patch(cls) -> None:
        if cls._xtts_audio_patch_done or xtts_module is None or torch is None:
            return

        def _patched_load_audio(audiopath, sampling_rate):
            stream, loaded_sample_rate = sf.read(str(audiopath), dtype="float32")
            if getattr(stream, "ndim", 1) > 1:
                stream = stream.mean(axis=1)
            if int(loaded_sample_rate) != int(sampling_rate):
                stream = resampy.resample(x=stream, sr_orig=int(loaded_sample_rate), sr_new=int(sampling_rate))
            stream = np.asarray(stream, dtype=np.float32)
            if stream.size == 0:
                return torch.zeros((1, 0), dtype=torch.float32)
            stream = np.clip(stream, -1.0, 1.0)
            return torch.from_numpy(stream).unsqueeze(0)

        xtts_module.load_audio = _patched_load_audio
        cls._xtts_audio_patch_done = True
        logger.info("Coqui XTTS v2 patched XTTS load_audio to avoid torchaudio/torchcodec dependency")

    def warmup(self) -> None:
        self._get_model()
        self._synthesize("hello", warmup=True)

    def _synthesize(self, text: str, warmup: bool = False) -> tuple[np.ndarray, int]:
        model = self._get_model()
        prepared_reference_path = self._prepare_reference_audio_path()
        start = time.perf_counter()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = Path(temp_file.name)
        try:
            synth_kwargs = {
                "text": text,
                "speaker_wav": str(prepared_reference_path),
                "language": self.language,
                "file_path": str(temp_path),
                "speed": max(float(self.speed or 1.0), 0.05),
            }
            try:
                model.tts_to_file(split_sentences=False, **synth_kwargs)
            except TypeError:
                try:
                    model.tts_to_file(**synth_kwargs)
                except TypeError:
                    synth_kwargs.pop("speed", None)
                    model.tts_to_file(**synth_kwargs)
            stream, sample_rate = sf.read(str(temp_path), dtype="float32")
            if stream.ndim > 1:
                stream = stream[:, 0]
            if stream.size == 0:
                logger.warning("Coqui XTTS v2 returned empty audio for text=%s warmup=%s", text, warmup)
                return np.zeros(0, dtype=np.float32), 24000
            if sample_rate != self.sample_rate:
                stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
            pcm = np.asarray(stream, dtype=np.float32)
            logger.info(
                "Coqui XTTS v2 synth done in %.3fs, sample_rate=%s, duration=%.3fs, text_len=%s, language=%s warmup=%s",
                time.perf_counter() - start,
                int(sample_rate),
                float(len(pcm)) / float(self.sample_rate),
                len(text),
                self.language,
                warmup,
            )
            return pcm, int(sample_rate)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def txt_to_audio(self, msg: tuple[str, dict]) -> None:
        text, textevent = msg
        text = (text or "").strip()
        if not text:
            return

        total_start = time.perf_counter()
        segments = split_tts_segments(text)
        if not segments:
            return

        textevent = dict(textevent or {})
        total_audio_samples = 0
        queued_chunks = 0
        queued_samples = 0
        first_chunk_elapsed = None
        total_segments = len(segments)

        logger.info(
            "Coqui XTTS v2 segmented text into %s parts, text_len=%s, language=%s",
            total_segments,
            len(text),
            self.language,
        )

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="coqui_tts_prefetch") as executor:
            future: Future[tuple[np.ndarray, int]] | None = executor.submit(self._synthesize, segments[0], False)
            next_segment_index = 1
            logger.info(
                "Coqui XTTS v2 prefetch submit segment[1/%s] at +%.3fs text_len=%s",
                total_segments,
                time.perf_counter() - total_start,
                len(segments[0]),
            )

            for segment_index, _segment_text in enumerate(segments, start=1):
                if future is None:
                    break
                stream, sample_rate = future.result()
                future = None
                segment_ready_elapsed = time.perf_counter() - total_start
                logger.info(
                    "Coqui XTTS v2 segment[%s/%s] ready at +%.3fs sample_rate=%s samples=%s duration=%.3fs",
                    segment_index,
                    total_segments,
                    segment_ready_elapsed,
                    sample_rate,
                    int(stream.shape[0]),
                    float(stream.shape[0]) / float(self.sample_rate) if stream.size > 0 else 0.0,
                )

                if next_segment_index < len(segments) and self.state == State.RUNNING:
                    future = executor.submit(self._synthesize, segments[next_segment_index], False)
                    logger.info(
                        "Coqui XTTS v2 prefetch submit segment[%s/%s] at +%.3fs while queueing current segment[%s/%s]",
                        next_segment_index + 1,
                        total_segments,
                        time.perf_counter() - total_start,
                        segment_index,
                        total_segments,
                    )
                    next_segment_index += 1

                if stream.size == 0:
                    logger.warning(
                        "Coqui XTTS v2 segment[%s/%s] produced empty audio",
                        segment_index,
                        total_segments,
                    )
                    continue

                total_audio_samples += int(stream.shape[0])
                streamlen = stream.shape[0]
                idx = 0
                segment_chunk_count = 0
                segment_queue_start_elapsed = None
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
                    if segment_queue_start_elapsed is None:
                        segment_queue_start_elapsed = time.perf_counter() - total_start
                    queued_chunks += 1
                    segment_chunk_count += 1
                    queued_samples += self.chunk
                    idx += self.chunk
                logger.info(
                    "Coqui XTTS v2 segment[%s/%s] queued at +%.3fs chunks=%s duration=%.3fs",
                    segment_index,
                    total_segments,
                    float(segment_queue_start_elapsed or (time.perf_counter() - total_start)),
                    segment_chunk_count,
                    float(stream.shape[0]) / float(self.sample_rate),
                )

        total_elapsed = time.perf_counter() - total_start
        textevent.setdefault("tts_elapsed", total_elapsed)
        textevent.setdefault("tts_duration", float(total_audio_samples) / float(self.sample_rate))
        logger.info(
            "Coqui XTTS v2 stream timings: total=%.3fs first_chunk=%.3fs queued_chunks=%s queued_samples=%s segments=%s text_len=%s language=%s",
            total_elapsed,
            float(first_chunk_elapsed or total_elapsed),
            queued_chunks,
            queued_samples,
            len(segments),
            len(text),
            self.language,
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
