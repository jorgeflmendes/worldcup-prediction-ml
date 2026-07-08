"""Filesystem configuration."""

from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


PROJECT_ROOT = _path_from_env(
    "WORLDCUP2026_ROOT",
    Path(__file__).resolve().parents[2],
)
DATA_DIR = _path_from_env("WORLDCUP2026_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = _path_from_env("WORLDCUP2026_RAW_DIR", DATA_DIR / "raw")
EXTERNAL_DIR = _path_from_env(
    "WORLDCUP2026_EXTERNAL_DIR",
    DATA_DIR / "external",
)
ARTIFACT_DIR = _path_from_env(
    "WORLDCUP2026_ARTIFACT_DIR",
    PROJECT_ROOT / "artifacts",
)
OUTPUT_DIR = _path_from_env("WORLDCUP2026_OUTPUT_DIR", PROJECT_ROOT / "output")
