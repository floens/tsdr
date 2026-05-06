from __future__ import annotations

import json
import logging
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from platformdirs import user_config_dir

logger = logging.getLogger(__name__)


def config_dir() -> Path:
    return Path(user_config_dir("tsdr"))


def load_toml(filename: str) -> dict[str, Any]:
    path = config_dir() / filename
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        logger.warning("Failed to load %s: %s", path, e)
        return {}


def save_toml(filename: str, data: dict[str, Any]) -> None:
    path = config_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "wb") as f:
            tomli_w.dump(data, f)
    except OSError as e:
        logger.warning("Failed to save %s: %s", path, e)


def read_text(filename: str) -> str:
    path = config_dir() / filename
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to read %s: %s", path, e)
        return ""


def write_text(filename: str, content: str) -> None:
    path = config_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to write %s: %s", path, e)


def list_files(subdir: str, suffix: str) -> list[Path]:
    """Return sorted files in config_dir()/subdir with the given suffix.

    `suffix` includes the dot (e.g. ".json"). Missing subdir: []; OSError
    is logged and [] returned.
    """
    path = config_dir() / subdir
    if not path.exists():
        return []
    try:
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix == suffix)
    except OSError as e:
        logger.warning("Failed to list %s: %s", path, e)
        return []


def load_json(relpath: str) -> Any:
    """Load a JSON file relative to config_dir().

    Missing file: None. OSError / JSONDecodeError are logged and None returned.
    `relpath` may be a nested path like "bandplans/usa.json".
    """
    path = config_dir() / relpath
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None
