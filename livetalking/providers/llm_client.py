from __future__ import annotations

import json
import time
import uuid

import requests
from requests import Response
from requests.exceptions import RequestException

from ..config.app_config import clean_base_url
from ..core.base_real import BaseReal
from ..utils.app_logger import logger


_llm_config = {
    "mode": "openai_chat",
    "base_url": "https://api.openai.com/v1",
    "api_key": "",
    "model": "",
    "timeout_s": 60,
    "system_prompt_zh": "\u4f60\u662f\u4e00\u4e2a\u79bb\u7ebf\u6570\u5b57\u4eba\u52a9\u624b\u3002\u8bf7\u5728\u9700\u8981\u65f6\u4f7f\u7528\u4e2d\u6587\u7b80\u6d01\u51c6\u786e\u5730\u56de\u7b54\u3002\u6700\u591a\u4e09\u53e5\u8bdd\uff0c\u4e0d\u8981\u4f7f\u7528\u8868\u60c5\u3002",
}

MAX_DIALOG_CONTEXT = 10
RETRYABLE_STATUS_CODES = {502, 503, 504}
MAX_ERROR_BODY_PREVIEW = 1000


def _repair_mojibake(text: str) -> str:
    if not text:
        return text

    suspicious_markers = ("脙", "脗", "盲", "氓", "忙", "莽", "茂", "垄", "鈧", "鈩")
    if not any(marker in text for marker in suspicious_markers):
        return text

    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return repaired


def configure_llm(config, mode: str = "openai_chat") -> None:
    _llm_config["mode"] = mode
    _llm_config["base_url"] = clean_base_url(config.base_url).rstrip("/")
    _llm_config["api_key"] = config.api_key
    _llm_config["model"] = config.model
    _llm_config["timeout_s"] = int(config.timeout_s)
    _llm_config["system_prompt_zh"] = config.system_prompt_zh


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if _llm_config["api_key"]:
        headers["Authorization"] = f"Bearer {_llm_config['api_key']}"
    return headers


def _join_endpoint(base_url: str, endpoint: str) -> str:
    clean_url = clean_base_url(base_url).rstrip("/")
    endpoint = endpoint.strip()
    if not clean_url:
        return endpoint
    if clean_url.endswith(endpoint):
        return clean_url
    return f"{clean_url}/{endpoint.lstrip('/')}"


def _normalize_dialog_role(role: str | None) -> str:
    normalized = str(role or "user").strip().lower()
    return "assistant" if normalized == "assistant" else "user"


def _build_dialog_messages(
    nerfreal: BaseReal,
    message: str,
    continuous_dialogue: bool,
    max_context: int = MAX_DIALOG_CONTEXT,
) -> list[dict[str, str]]:
    content = (message or "").strip()
    if not continuous_dialogue:
        return [{"role": "user", "content": content}]

    history: list[dict] = []
    if hasattr(nerfreal, "get_dialog_history"):
        try:
            history = nerfreal.get_dialog_history(max_context)
        except Exception:
            logger.exception("failed to load dialog history for llm context")
            history = []

    messages: list[dict[str, str]] = []
    for item in history:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        messages.append({
            "role": _normalize_dialog_role(item.get("role")),
            "content": text,
        })

    if not messages or messages[-1]["role"] != "user" or messages[-1]["content"] != content:
        messages.append({"role": "user", "content": content})

    if len(messages) > max_context:
        messages = messages[-max_context:]
    return messages


def _build_responses_input_messages(
    nerfreal: BaseReal,
    message: str,
    continuous_dialogue: bool,
) -> list[dict]:
    dialog_messages = _build_dialog_messages(nerfreal, message, continuous_dialogue)
    input_messages: list[dict] = []
    for item in dialog_messages:
        text = str(item.get("content", "")).strip()
        if not text:
            continue
        input_messages.append({
            "role": item.get("role", "user"),
            "content": [
                {
                    "type": "input_text",
                    "text": text,
                }
            ],
        })
    return input_messages


def _latest_lm_studio_response_id(nerfreal: BaseReal) -> str:
    if not hasattr(nerfreal, "get_dialog_history"):
        return ""

    try:
        history = nerfreal.get_dialog_history(MAX_DIALOG_CONTEXT)
    except Exception:
        logger.exception("failed to load dialog history for lm studio context")
        return ""

    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")).strip().lower() != "assistant":
            continue
        meta = item.get("meta")
        if isinstance(meta, dict):
            response_id = str(meta.get("response_id", "")).strip()
            if response_id:
                return response_id
    return ""


def _payload(message: str, nerfreal: BaseReal, continuous_dialogue: bool) -> dict:
    if _llm_config["mode"] == "chat_completions_api":
        return {
            "model": _llm_config["model"],
            "messages": [
                {
                    "role": "system",
                    "content": _llm_config["system_prompt_zh"],
                },
                *_build_dialog_messages(nerfreal, message, continuous_dialogue),
            ],
            "stream": True,
        }

    if _llm_config["mode"] == "lm_studio_api":
        payload = {
            "model": _llm_config["model"],
            "input": (message or "").strip(),
            "stream": True,
        }
        system_prompt = str(_llm_config["system_prompt_zh"] or "").strip()
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if continuous_dialogue:
            previous_response_id = _latest_lm_studio_response_id(nerfreal)
            if previous_response_id:
                payload["previous_response_id"] = previous_response_id
            payload["store"] = True
        else:
            payload["store"] = False
        return payload

    payload = {
        "model": _llm_config["model"],
        "input": _build_responses_input_messages(nerfreal, message, continuous_dialogue),
        "instructions": _llm_config["system_prompt_zh"],
        "stream": True,
    }
    return payload


def _extract_text(data: dict) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            delta = first_choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                return _repair_mojibake(delta["content"])
            message = first_choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return _repair_mojibake(message["content"])

    if isinstance(data.get("content"), str):
        return _repair_mojibake(data["content"])
    if isinstance(data.get("delta"), str):
        return _repair_mojibake(data["delta"])
    if isinstance(data.get("output_text"), str):
        return _repair_mojibake(data["output_text"])

    output = data.get("output")
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                texts.append(content)
                continue
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    texts.append(block["text"])
        if texts:
            return _repair_mojibake("".join(texts))
    return ""


def _extract_response_id(data: dict) -> str:
    response_id = data.get("response_id")
    if isinstance(response_id, str) and response_id.strip():
        return response_id.strip()
    response = data.get("response")
    if isinstance(response, dict):
        response_id = response.get("response_id")
        if isinstance(response_id, str) and response_id.strip():
            return response_id.strip()
    return ""


def _consume_text(
    message: str,
    result: str,
    sentence: str,
    sentence_index: int,
    sentence_start: float,
    first: bool,
    start: float,
):
    if not message:
        return result, sentence, sentence_index, sentence_start, first

    if first:
        logger.info("llm Time to first chunk: %ss", time.perf_counter() - start)
        first = False

    for char in message:
        result += char
        sentence += char
        if char in ",.!;:锛屻€傦紒锛燂紱锛?":
            text = sentence.strip()
            if text:
                logger.info(
                    "llm sentence %s elapsed: %.3fs text=%s",
                    sentence_index,
                    time.perf_counter() - sentence_start,
                    text,
                )
            sentence_index += 1
            sentence = ""
            sentence_start = time.perf_counter()

    return result, sentence, sentence_index, sentence_start, first


def _response_preview(response: Response) -> str:
    try:
        body = response.text
    except Exception:
        return "<failed to read response body>"

    body = (body or "").strip()
    if not body:
        return "<empty response body>"
    if len(body) > MAX_ERROR_BODY_PREVIEW:
        return body[:MAX_ERROR_BODY_PREVIEW] + "...(truncated)"
    return body


def _post_llm_request(url: str, payload: dict, stream: bool) -> Response | None:
    session = requests.Session()
    session.trust_env = False

    last_error: RequestException | None = None
    for attempt in range(2):
        try:
            response = session.post(
                url,
                headers=_headers(),
                json=payload,
                timeout=_llm_config["timeout_s"],
                stream=stream,
            )
            if response.status_code in RETRYABLE_STATUS_CODES and attempt == 0:
                logger.warning(
                    "llm request returned %s, retrying once: %s body=%s",
                    response.status_code,
                    url,
                    _response_preview(response),
                )
                response.close()
                time.sleep(0.5)
                continue
            if response.status_code >= 400:
                logger.warning(
                    "llm request failed with %s: %s body=%s",
                    response.status_code,
                    url,
                    _response_preview(response),
                )
                response.close()
                return None
            response.raise_for_status()
            return response
        except RequestException as exc:
            last_error = exc
            if attempt == 0:
                logger.warning("llm request failed, retrying once: %s", exc)
                time.sleep(0.5)
                continue
            break

    if last_error is not None:
        logger.warning("llm request failed after retry: %s", last_error)
    return None


def llm_response(
    message: str,
    nerfreal: BaseReal,
    continuous_dialogue: bool = False,
    dialog_start_ts: float | None = None,
    response_token: int | None = None,
):
    start = time.perf_counter()
    dialog_start_ts = start if dialog_start_ts is None else dialog_start_ts
    dialog_id = str(uuid.uuid4())
    if hasattr(nerfreal, "is_response_active") and not nerfreal.is_response_active(response_token):
        logger.info("llm response aborted before request because session is inactive or superseded")
        return
    if _llm_config["mode"] == "chat_completions_api":
        url = _join_endpoint(_llm_config["base_url"], "/chat/completions")
    elif _llm_config["mode"] == "lm_studio_api":
        url = _join_endpoint(_llm_config["base_url"], "/api/v1/chat")
    else:
        url = _join_endpoint(_llm_config["base_url"], "/responses")

    payload = _payload(message, nerfreal, continuous_dialogue)
    response = _post_llm_request(url, payload, bool(payload.get("stream", False)))
    if response is None:
        return

    response.encoding = "utf-8"

    result = ""
    sentence = ""
    sentence_start = start
    sentence_index = 1
    first = True
    response_id = ""

    if not bool(payload.get("stream", False)):
        try:
            data = response.json()
        except ValueError:
            data = {}
        response_id = _extract_response_id(data)
        msg = _extract_text(data)
        result, sentence, sentence_index, sentence_start, first = _consume_text(
            msg,
            result,
            sentence,
            sentence_index,
            sentence_start,
            first,
            start,
        )
    else:
        for raw_line in response.iter_lines(decode_unicode=False):
            if hasattr(nerfreal, "is_response_active") and not nerfreal.is_response_active(response_token):
                logger.info("llm response dropped while streaming because session is inactive or superseded")
                response.close()
                return
            if not raw_line:
                continue
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace").strip()
            else:
                line = str(raw_line).strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            extracted_response_id = _extract_response_id(data)
            if extracted_response_id:
                response_id = extracted_response_id

            msg = _extract_text(data)
            result, sentence, sentence_index, sentence_start, first = _consume_text(
                msg,
                result,
                sentence,
                sentence_index,
                sentence_start,
                first,
                start,
            )

    if sentence.strip():
        logger.info(
            "llm sentence %s elapsed: %.3fs text=%s",
            sentence_index,
            time.perf_counter() - sentence_start,
            sentence.strip(),
        )

    elapsed = time.perf_counter() - start
    logger.info("llm Time to last chunk: %ss", elapsed)
    if hasattr(nerfreal, "is_response_active") and not nerfreal.is_response_active(response_token):
        logger.info("llm response dropped after completion because session is inactive or superseded")
        return
    if result.strip():
        if hasattr(nerfreal, "append_dialog"):
            meta = {"llm_elapsed": elapsed, "dialog_id": dialog_id}
            if response_id:
                meta["response_id"] = response_id
            nerfreal.append_dialog("assistant", result, "llm", meta, dialog_id=dialog_id)
        logger.info(result)
        nerfreal.put_msg_txt(
            result,
            {
                "llm_elapsed": elapsed,
                "llm_start_ts": start,
                "dialog_start_ts": dialog_start_ts,
                "llm_sentence_count": sentence_index - 1 if not sentence.strip() else sentence_index,
                "dialog_id": dialog_id,
            },
        )
