from __future__ import annotations

import asyncio
import queue
import time
from enum import Enum
from io import BytesIO
from threading import Thread
from typing import TYPE_CHECKING

import edge_tts
import numpy as np
import resampy
import soundfile as sf

from ..utils.app_logger import logger

if TYPE_CHECKING:
    from ..core.base_real import BaseReal


class State(Enum):
    RUNNING = 0
    PAUSE = 1


class BaseTTS:
    def __init__(self, opt, parent: BaseReal):
        self.opt = opt
        self.parent = parent
        self.fps = opt.fps
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps
        self.input_stream = BytesIO()
        self.msgqueue = queue.Queue()
        self.state = State.RUNNING

    def warmup(self) -> None:
        return None

    def flush_talk(self) -> None:
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self, msg: str, datainfo: dict | None = None) -> None:
        if not msg:
            return
        logger.info("TTS enqueue text=%s", msg)
        self.msgqueue.put((msg, dict(datainfo or {})))

    def render(self, quit_event) -> None:
        process_thread = Thread(target=self.process_tts, args=(quit_event,), daemon=True, name="tts_process")
        process_thread.start()

    def process_tts(self, quit_event) -> None:
        while not quit_event.is_set():
            try:
                msg: tuple[str, dict] = self.msgqueue.get(block=True, timeout=1)
                self.state = State.RUNNING
                logger.info("TTS dequeued text=%s", msg[0])
            except queue.Empty:
                continue

            start = time.perf_counter()
            self.txt_to_audio(msg)
            elapsed = time.perf_counter() - start
            text, textevent = msg
            dialog_id = str(textevent.get("dialog_id") or textevent.get("id") or "")
            if dialog_id and hasattr(self.parent, "update_dialog_meta"):
                self.parent.update_dialog_meta(dialog_id, {"tts_elapsed": elapsed})
            logger.info("TTS total elapsed %.4fs text=%s", elapsed, text)
        logger.info("tts process thread stop")

    def txt_to_audio(self, msg: tuple[str, dict]) -> None:
        raise NotImplementedError


class EdgeTTS(BaseTTS):
    def txt_to_audio(self, msg: tuple[str, dict]) -> None:
        voicename = self.opt.REF_FILE
        text, textevent = msg
        start = time.perf_counter()
        asyncio.new_event_loop().run_until_complete(self._stream_edge_tts(voicename, text))
        logger.info("EdgeTTS synth done in %.4fs", time.perf_counter() - start)
        if self.input_stream.getbuffer().nbytes <= 0:
            logger.error("EdgeTTS returned empty audio")
            return

        self.input_stream.seek(0)
        stream = self._create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx = 0
        queued_chunks = 0
        queued_samples = 0
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
            queued_chunks += 1
            queued_samples += self.chunk
            idx += self.chunk
        logger.info(
            "EdgeTTS audio queued chunks=%s queued_samples=%s text_len=%s",
            queued_chunks,
            queued_samples,
            len(text),
        )
        self.input_stream.seek(0)
        self.input_stream.truncate()

    def _create_bytes_stream(self, byte_stream: BytesIO) -> np.ndarray:
        stream, sample_rate = sf.read(byte_stream)
        logger.info("[INFO] EdgeTTS audio stream %s: %s", sample_rate, getattr(stream, "shape", None))
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info("[WARN] audio has %s channels, only use the first.", stream.shape[1])
            stream = stream[:, 0]

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info("[WARN] audio sample rate is %s, resampling into %s.", sample_rate, self.sample_rate)
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    async def _stream_edge_tts(self, voicename: str, text: str) -> None:
        try:
            communicate = edge_tts.Communicate(text, voicename)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and self.state == State.RUNNING:
                    self.input_stream.write(chunk["data"])
        except Exception:
            logger.exception("EdgeTTS synth failed")
