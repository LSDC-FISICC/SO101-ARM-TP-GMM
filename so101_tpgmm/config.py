"""YAML configuration loading.

The configuration is kept as a plain nested ``dict`` (with a thin attribute
accessor) rather than a rigid dataclass schema: phases add sections
incrementally and scripts accept ``--set key.sub=value`` overrides.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config, optionally applying ``key.sub=value`` overrides.

    Args:
        path: YAML file. Defaults to ``configs/default.yaml``.
        overrides: Strings like ``"gripper.open_threshold=30"``. Values are parsed
            with YAML so numbers/lists/bools keep their types.

    Returns:
        The configuration dictionary.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        if not _:
            raise ValueError(f"Override must look like key.sub=value, got {ov!r}")
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def set_by_path(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``cfg[a][b][c] = value`` for ``dotted == "a.b.c"`` (creating dicts)."""
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def merged(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-merged copy of two config dicts (``extra`` wins)."""
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merged(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
