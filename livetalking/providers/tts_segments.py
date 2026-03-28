from __future__ import annotations

import re
import unicodedata


_SENTENCE_BREAK_RE = re.compile("([\u3002\uff1f\uff01.!?;\uff1b:\uff1a\n])")
_SPACE_RE = re.compile(r"\s+")


def normalize_tts_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).replace("\uFFFD", " ")
    return _SPACE_RE.sub(" ", normalized).strip()


def _split_long_segment(segment: str, max_chars: int) -> list[str]:
    current = segment.strip()
    if not current:
        return []

    pieces: list[str] = []
    while len(current) > max_chars:
        cut = -1
        for marker in (" ", ",", ";", ":", "\u3001", "\uff0c"):
            index = current.rfind(marker, 0, max_chars + 1)
            if index > cut:
                cut = index
        if cut <= 0:
            cut = max_chars
        piece = current[:cut].strip()
        if piece:
            pieces.append(piece)
        current = current[cut:].strip()
    if current:
        pieces.append(current)
    return pieces


def split_tts_segments(text: str, *, min_chars: int = 12, max_chars: int = 80) -> list[str]:
    cleaned = normalize_tts_text(text)
    if not cleaned:
        return []

    raw_segments: list[str] = []
    start = 0
    for match in _SENTENCE_BREAK_RE.finditer(cleaned):
        end = match.end()
        piece = cleaned[start:end].strip()
        if piece:
            raw_segments.append(piece)
        start = end
    tail = cleaned[start:].strip()
    if tail:
        raw_segments.append(tail)
    if not raw_segments:
        raw_segments = [cleaned]

    segments: list[str] = []
    for segment in raw_segments:
        if len(segment) <= max_chars:
            segments.append(segment)
            continue

        pieces = _split_long_segment(segment, max_chars)
        if not pieces:
            continue

        merged_pieces: list[str] = []
        buffer = ""
        for piece in pieces:
            if not buffer:
                buffer = piece
                continue
            if len(buffer) < min_chars and len(buffer) + len(piece) <= max_chars:
                buffer += piece
                continue
            merged_pieces.append(buffer)
            buffer = piece
        if buffer:
            merged_pieces.append(buffer)
        segments.extend(merged_pieces)

    return [segment for segment in segments if segment]
