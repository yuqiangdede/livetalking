from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..utils.app_logger import logger

try:  # pragma: no cover - optional dependency
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None


_SEGMENT_SIZE = 20
_MAX_LEN = 200


def _normalize_text(text: str) -> str:
    current = re.sub(r"\s+", " ", str(text or "")).strip()
    if not current:
        return ""
    current = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", current)
    current = re.sub(r"\s+([，。！？!?、,.;:])", r"\1", current)
    return current


def _split_words(text: str) -> list[str]:
    words: list[str] = []
    for raw_word in _normalize_text(text).split():
        buffer = ""
        for char in raw_word:
            if len(char.encode("utf-8")) > 1:
                if not buffer:
                    buffer = char
                elif len(buffer[-1].encode("utf-8")) > 1:
                    buffer += char
                else:
                    words.append(buffer)
                    buffer = char
            else:
                if not buffer:
                    buffer = char
                elif len(buffer[-1].encode("utf-8")) > 1:
                    words.append(buffer)
                    buffer = char
                else:
                    buffer += char
        if buffer:
            words.append(buffer)
    return words


def _build_token_ids(words: list[str], token2id: dict[str, int], unk_id: int) -> list[int]:
    ids: list[int] = []
    for word in words:
        if not word:
            continue
        if len(word[0].encode("utf-8")) > 1:
            ids.extend(token2id.get(char, unk_id) for char in word)
        else:
            ids.append(token2id.get(word, unk_id))
    return ids


def _to_text(
    ids: list[int],
    punctuations: list[int],
    id2token: list[str],
    id2punct: list[str],
    underscore_id: int,
) -> str:
    if not ids:
        return ""

    ans: list[str] = []
    for index, punctuation_id in enumerate(punctuations[: len(ids)]):
        token = id2token[ids[index]] if 0 <= ids[index] < len(id2token) else ""
        if not token:
            continue
        if ans and len(ans[-1].encode("utf-8")) == 1 and len(token.encode("utf-8")) == 1:
            ans.append(" ")
        ans.append(token)
        if punctuation_id != underscore_id and 0 <= punctuation_id < len(id2punct):
            punct = id2punct[punctuation_id]
            if punct:
                ans.append(punct)
    return "".join(ans).strip()


class CTTransformerPunctuationRestorer:
    def __init__(self, model_dir: str, session_factory: Any | None = None) -> None:
        self.model_dir = Path(model_dir)
        self._session_factory = session_factory
        self._session = None
        self._load_lock = threading.Lock()
        self._input_name = ""
        self._length_name = ""
        self._output_name = ""
        self._token2id: dict[str, int] = {}
        self._id2token: list[str] = []
        self._punct2id: dict[str, int] = {}
        self._id2punct: list[str] = []
        self._unk_id = 0
        self._underscore_id = 1
        self._comma_id = 2
        self._dot_id = 3
        self._quest_id = 4
        self._validate_model_dir()
        self._ensure_loaded()

    def _validate_model_dir(self) -> None:
        required = ["model.int8.onnx", "config.yaml", "tokens.json"]
        missing = [name for name in required if not (self.model_dir / name).exists()]
        if missing:
            raise RuntimeError(
                "punctuation model directory is incomplete: "
                f"model_dir={self.model_dir} missing={missing}"
            )

    def _load_config(self) -> dict[str, Any]:
        config_path = self.model_dir / "config.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _load_tokens(self) -> list[str]:
        token_path = self.model_dir / "tokens.json"
        with token_path.open("r", encoding="utf-8") as handle:
            tokens = json.load(handle)
        if not isinstance(tokens, list):
            raise RuntimeError(f"invalid tokens.json in {self.model_dir}")
        return [str(item) for item in tokens]

    def _build_session(self):
        if self._session_factory is not None:
            return self._session_factory(self.model_dir)
        if ort is None:
            raise RuntimeError("onnxruntime is not installed, cannot use punctuation model")

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        model_path = self.model_dir / "model.int8.onnx"
        try:
            return ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception:
            return ort.InferenceSession(str(model_path), sess_options=session_options)

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return

            session = self._build_session()
            config = self._load_config()
            tokens = self._load_tokens()

            punctuation_list = list(config.get("model_conf", {}).get("punc_list", []))
            if not punctuation_list:
                try:
                    metadata = dict(getattr(session.get_modelmeta(), "custom_metadata_map", {}) or {})
                except Exception:
                    metadata = {}
                punc_list = metadata.get("punctuations")
                if isinstance(punc_list, str) and punc_list:
                    punctuation_list = punc_list.split("|")
            if not punctuation_list:
                punctuation_list = ["<unk>", "_", "，", "。", "？", "、"]

            token_list = tokens
            unk_symbol = str(config.get("tokenizer_conf", {}).get("unk_symbol", "<unk>"))

            self._id2token = [str(item) for item in token_list]
            self._token2id = {token: index for index, token in enumerate(self._id2token)}
            self._id2punct = [str(item) for item in punctuation_list]
            self._punct2id = {punct: index for index, punct in enumerate(self._id2punct)}
            self._unk_id = self._token2id.get(unk_symbol, 0)
            self._underscore_id = self._punct2id.get("_", 1 if len(self._id2punct) > 1 else 0)
            self._comma_id = self._punct2id.get("，", self._punct2id.get(",", 0))
            self._dot_id = self._punct2id.get("。", int(config.get("model_conf", {}).get("sentence_end_id", 3)))
            self._quest_id = self._punct2id.get("？", self._punct2id.get("?", self._dot_id))
            self._input_name = session.get_inputs()[0].name
            self._length_name = session.get_inputs()[1].name
            self._output_name = session.get_outputs()[0].name
            self._session = session

    def _run_segment(self, inputs: list[int]) -> list[int]:
        self._ensure_loaded()
        assert self._session is not None
        logits = self._session.run(
            [self._output_name],
            {
                self._input_name: np.array(inputs, dtype=np.int32).reshape(1, -1),
                self._length_name: np.array([len(inputs)], dtype=np.int32),
            },
        )[0]
        return logits[0].argmax(axis=-1).tolist()

    def punctuate_text(self, text: str) -> str:
        normalized = _normalize_text(text)
        if not normalized:
            return ""

        words = _split_words(normalized)
        ids = _build_token_ids(words, self._token2id, self._unk_id) if self._session is not None else []
        if not ids:
            self._ensure_loaded()
            ids = _build_token_ids(words, self._token2id, self._unk_id)
        if not ids:
            return normalized

        punctuations: list[int] = []
        last = -1
        num_segments = (len(ids) + _SEGMENT_SIZE - 1) // _SEGMENT_SIZE

        for index in range(num_segments):
            this_start = index * _SEGMENT_SIZE
            this_end = min(this_start + _SEGMENT_SIZE, len(ids))
            if last != -1:
                this_start = last
            segment = ids[this_start:this_end]
            if not segment:
                continue

            output = self._run_segment(segment)
            dot_index = -1
            comma_index = -1
            for output_index in range(len(output) - 1, 1, -1):
                value = output[output_index]
                if value in (self._dot_id, self._quest_id):
                    dot_index = output_index
                    break
                if comma_index == -1 and value == self._comma_id:
                    comma_index = output_index

            if dot_index == -1 and len(segment) >= _MAX_LEN and comma_index != -1:
                dot_index = comma_index
                output[dot_index] = self._dot_id

            if dot_index == -1:
                if last == -1:
                    last = this_start
                if index == num_segments - 1:
                    dot_index = len(segment) - 1
            else:
                last = this_start + dot_index + 1

            if dot_index != -1:
                punctuations.extend(output[: dot_index + 1])

        if len(punctuations) < len(ids):
            punctuations.extend([self._underscore_id] * (len(ids) - len(punctuations)))

        return _to_text(
            ids,
            punctuations,
            self._id2token,
            self._id2punct,
            self._underscore_id,
        )

    def punctuate(self, text: str) -> str:
        try:
            return self.punctuate_text(text)
        except Exception as exc:
            logger.warning("punctuation restoration failed, fallback to plain text: %s", exc)
            return _normalize_text(text)
