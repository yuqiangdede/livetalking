from __future__ import annotations

import re
from typing import Any


_COMMENT_PREFIXES = ("#", ";", "//")


def _strip_inline_comment(line: str) -> str:
    current = str(line or "")
    for marker in (" #", "\t#", " ;", "\t;", " //", "\t//"):
        index = current.find(marker)
        if index >= 0:
            current = current[:index]
    return current.strip()


def _is_comment_line(line: str) -> bool:
    stripped = str(line or "").lstrip()
    if not stripped:
        return True
    return any(stripped.startswith(prefix) for prefix in _COMMENT_PREFIXES)


def parse_hotword_lines(text: str) -> list[str]:
    hotwords: list[str] = []
    seen: set[str] = set()
    for raw_line in str(text or "").splitlines():
        if _is_comment_line(raw_line):
            continue
        line = _strip_inline_comment(raw_line)
        if not line or line in seen:
            continue
        seen.add(line)
        hotwords.append(line)
    return hotwords


def _split_replacement_line(line: str) -> tuple[str, str] | None:
    current = _strip_inline_comment(line)
    if not current:
        return None

    for marker in ("=>", "->", "→", "=", "\t"):
        if marker in current:
            left, right = current.split(marker, 1)
            source = left.strip()
            target = right.strip()
            if source and target:
                return source, target

    parts = current.split(None, 1)
    if len(parts) == 2:
        source, target = parts[0].strip(), parts[1].strip()
        if source and target:
            return source, target
    return None


def parse_phonetic_replacements(text: str) -> list[tuple[str, str]]:
    rules_with_index: list[tuple[int, str, str]] = []
    seen_sources: set[str] = set()
    for index, raw_line in enumerate(str(text or "").splitlines()):
        if _is_comment_line(raw_line):
            continue
        parsed = _split_replacement_line(raw_line)
        if parsed is None:
            continue
        source, target = parsed
        if source == target or source in seen_sources:
            continue
        seen_sources.add(source)
        rules_with_index.append((index, source, target))

    rules_with_index.sort(key=lambda item: (-len(item[1]), item[0]))
    return [(source, target) for _, source, target in rules_with_index]


def apply_phonetic_replacements(text: str, rules: list[tuple[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    current = str(text or "")
    applied: list[dict[str, Any]] = []
    for source, target in rules:
        if not source or not target or source not in current:
            continue
        before = current
        current = current.replace(source, target)
        if current != before:
            applied.append(
                {
                    "stage": "phonetic",
                    "rule": f"{source}=>{target}",
                    "pattern": source,
                    "replacement": target,
                    "before": before,
                    "after": current,
                }
            )
    return current, applied
