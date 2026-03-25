from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PORTABLE_PYTHON_DIR = PROJECT_ROOT / "python"
PORTABLE_PYTHON_EXE = PORTABLE_PYTHON_DIR / "python.exe"
PORTABLE_PYTHON_SCRIPTS_DIR = PORTABLE_PYTHON_DIR / "Scripts"
FFMPEG_BIN_DIR = PROJECT_ROOT / "tools" / "ffmpeg" / "bin"


def get_runtime_path_entries() -> list[str]:
    entries: list[str] = []
    for path in (PORTABLE_PYTHON_DIR, PORTABLE_PYTHON_SCRIPTS_DIR, FFMPEG_BIN_DIR):
        if path.is_dir():
            entries.append(str(path))
    return entries


def resolve_runtime_executable(name: str, relative_path: str | None = None) -> str | None:
    if relative_path:
        bundled_path = PROJECT_ROOT / relative_path
        if bundled_path.is_file():
            return str(bundled_path)
    resolved = shutil.which(name)
    if resolved:
        return resolved
    return None


def require_runtime_executable(name: str, relative_path: str | None = None) -> str:
    resolved = resolve_runtime_executable(name, relative_path=relative_path)
    if resolved:
        return resolved
    if relative_path:
        expected = PROJECT_ROOT / relative_path
        raise FileNotFoundError(f"{name} not found. Expected bundled executable at {expected} or an entry in PATH.")
    raise FileNotFoundError(f"{name} not found in PATH.")


def resolve_ffmpeg_executable(required: bool = True) -> str | None:
    if required:
        return require_runtime_executable("ffmpeg", relative_path=r"tools\ffmpeg\bin\ffmpeg.exe")
    return resolve_runtime_executable("ffmpeg", relative_path=r"tools\ffmpeg\bin\ffmpeg.exe")


def resolve_ffprobe_executable(required: bool = True) -> str | None:
    if required:
        return require_runtime_executable("ffprobe", relative_path=r"tools\ffmpeg\bin\ffprobe.exe")
    return resolve_runtime_executable("ffprobe", relative_path=r"tools\ffmpeg\bin\ffprobe.exe")
