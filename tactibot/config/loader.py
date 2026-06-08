"""Carga de configuración YAML."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Raíz del repo CASTERBOT (contiene configs/, data/, tactibot/)
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    root = base or REPO_ROOT
    return (root / p).resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config no encontrado: {path}")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(REPO_ROOT)
    if "video_path" in cfg:
        cfg["video_path"] = str(resolve_path(cfg["video_path"]))
    return cfg


def snapshot_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.pop("_config_path", None)
    return out
